"""Código compartido por las tres familias de modelos: arma la matriz de features y entrena una corrida.

Cada familia agrega features a la anterior:
- historia: features_perfiles (155 seleccionadas) + lags y deltas de 1 y 2 meses de todas las variables + tendencia,
  mínimo, máximo y ratios de 6 meses para 40 variables + ranking con cero fijo de los montos.
- historia_comportamiento: lo anterior + features_comportamiento.
- historia_comportamiento_flujos: lo anterior + features_flujos + ranking con cero fijo de los flujos en pesos.

Una corrida construye el dataset de LightGBM una sola vez y entrena las semillas en el orden dado. LightGBM arma los
bins del dataset con la semilla del primer entrenamiento, así que para obtener las mismas probabilidades hay que repetir
las mismas corridas con las semillas en el mismo orden.
"""
import gc
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
from features_perfiles import agregar_features
import features_comportamiento
import features_flujos
from listas_de_variables import FEATURES_BASE_155, FEATURES_HISTORIA_40

DATOS = os.path.join(os.path.dirname(os.path.dirname(ENTREGA)), "data", "competencia_01.csv")
HISTORIA = os.path.join(ENTREGA, "trabajo", "historia")
PROBABILIDADES = os.path.join(ENTREGA, "probabilidades")

FAMILIAS = ("historia", "historia_comportamiento", "historia_comportamiento_flujos")
MESES = [202103, 202104, 202105, 202106, 202108]
MESES_ENTRENAMIENTO = [202103, 202104, 202105, 202106]
EXTRAS = ("tend", "min", "max", "ratioavg", "ratiomax")
PARAMETROS = dict(objective="binary", max_bin=31, min_data_in_leaf=0, feature_fraction=0.5, feature_fraction_bynode=0.2,
                  learning_rate=0.005, num_leaves=63, min_sum_hessian_in_leaf=10.0, verbose=-1, num_threads=4)
ARBOLES = 2000


def rankear_con_cero_fijo(x, mes):
    """Posición de cada valor dentro de su mes: positivos en (0, 1], negativos en [-1, 0), cero y NA sin cambios."""
    x = pd.Series(x, dtype=np.float64)
    pos = x.where(x > 0).groupby(mes).rank(pct=True).values
    neg = -(-x.where(x < 0)).groupby(mes).rank(pct=True).values
    return np.where(x > 0, pos, np.where(x < 0, neg, x.values)).astype(np.float32)


def preparar(familia, carpeta_historia=HISTORIA):
    """Devuelve el dataset de entrenamiento, la matriz de agosto y los numero_de_cliente de agosto."""
    assert familia in FAMILIAS
    crudo = pd.read_csv(DATOS)
    crudo.loc[crudo.foto_mes == 202105, "ccajas_depositos"] = np.nan   # en 202105 viene toda en cero
    montos = [c for c in crudo.columns if c.startswith(("m", "Visa_m", "Master_m")) and pd.api.types.is_numeric_dtype(crudo[c])]

    extra, cols_extra, cols_extra_rk = None, [], []
    if familia != "historia":
        extra = features_comportamiento.construir(crudo)
        extra = extra[extra.foto_mes.isin(MESES)].reset_index(drop=True)
        if familia == "historia_comportamiento_flujos":
            flujos = features_flujos.construir2(crudo)
            flujos = flujos[flujos.foto_mes.isin(MESES)].drop(columns=["x2_entradas", "x2_salidas"]).reset_index(drop=True)
            extra = extra.merge(flujos, on=["numero_de_cliente", "foto_mes"], how="left")
            del flujos
        gc.collect()
        cols_extra = [c for c in extra.columns if c.startswith(("x_", "x2_"))]
        if familia == "historia_comportamiento_flujos":
            cols_extra_rk = [c for c in cols_extra if c in features_flujos.MONTOS_X2 or c.startswith(("x_entradas", "x_salidas"))
                             or (c.endswith("_tend4") and not c.startswith("x2_productos"))]
            cols_extra_rk = [c for c in cols_extra_rk if "sobre" not in c]

    base = FEATURES_BASE_155
    full, _ = agregar_features(crudo)
    del crudo
    full = full[full.foto_mes.isin(MESES)][["numero_de_cliente", "foto_mes", "clase_ternaria"] + base].reset_index(drop=True)
    full[base] = full[base].astype(np.float32)
    full = full.sort_values("foto_mes", kind="stable").reset_index(drop=True)

    # columnas de historia: lags y deltas de todas las variables + ventana de 6 meses para las 40 elegidas
    bloques = sorted(f for f in os.listdir(carpeta_historia) if f.startswith("bloque_"))
    esquema = {f: pq.read_schema(os.path.join(carpeta_historia, f)).names for f in bloques}
    todas_h = [c for f in bloques for c in esquema[f]]
    cols_h = [c for c in todas_h if c.endswith(("_lag1", "_lag2", "_delta1", "_delta2"))]
    cols_h += [f"h_{c}_{s}" for c in FEATURES_HISTORIA_40 for s in EXTRAS if f"h_{c}_{s}" in todas_h]

    def es_monto(c):
        if "ratio" in c:
            return False
        if c.startswith("h_"):
            return c.endswith(("_lag1", "_lag2", "_delta1", "_delta2", "_tend", "_min", "_max")) and c[2:].rsplit("_", 1)[0] in montos
        return c.startswith(("m", "Visa_m", "Master_m"))

    feats = base + cols_h
    feats = feats + [f"rk_{c}" for c in feats if es_monto(c)]
    assert len(feats) == 1395, len(feats)
    feats = feats + cols_extra + [f"rk_{c}" for c in cols_extra_rk]
    print(f"{familia}: {len(feats)} features", flush=True)

    # fila i de la matriz = fila pos[i] de los parquet de historia
    claves = pd.read_parquet(os.path.join(carpeta_historia, "claves.parquet"))
    claves["_fila"] = np.arange(len(claves))
    pos = full[["numero_de_cliente", "foto_mes"]].merge(claves, on=["numero_de_cliente", "foto_mes"], how="left")["_fila"]
    assert pos.notna().all() and len(pos) == len(full)
    pos = pos.values.astype(np.int64)
    mes = full.foto_mes.values

    X = np.empty((len(full), len(feats)), dtype=np.float32)
    col = {c: j for j, c in enumerate(feats)}
    for c in base:
        v = full[c].values
        X[:, col[c]] = v
        if f"rk_{c}" in col:
            X[:, col[f"rk_{c}"]] = rankear_con_cero_fijo(v, mes)
    pedidas = set(cols_h)
    for f in bloques:
        cs = [c for c in esquema[f] if c in pedidas]
        if not cs:
            continue
        t = pq.read_table(os.path.join(carpeta_historia, f), columns=cs).to_pandas()
        for c in cs:
            v = t[c].values.astype(np.float32)[pos]
            X[:, col[c]] = v
            if f"rk_{c}" in col:
                X[:, col[f"rk_{c}"]] = rankear_con_cero_fijo(v, mes)
        del t
        gc.collect()
    if extra is not None:
        ex = full[["numero_de_cliente", "foto_mes"]].merge(extra, on=["numero_de_cliente", "foto_mes"], how="left")
        del extra
        for c in cols_extra:
            v = ex[c].values.astype(np.float32)
            X[:, col[c]] = v
            if c in cols_extra_rk:
                X[:, col[f"rk_{c}"]] = rankear_con_cero_fijo(v, mes)
        del ex

    n_entrenamiento = int(full.foto_mes.isin(MESES_ENTRENAMIENTO).sum())
    assert full.foto_mes.values[:n_entrenamiento].max() <= 202106 and (full.foto_mes.values[n_entrenamiento:] == 202108).all()
    ids = full.numero_de_cliente.values[n_entrenamiento:].astype(np.int64)
    y = full.clase_ternaria.isin(["BAJA+1", "BAJA+2"]).astype(int).values[:n_entrenamiento]
    del full
    gc.collect()
    dataset = lgb.Dataset(X[:n_entrenamiento], label=y, feature_name=feats, params={"max_bin": 31}, free_raw_data=True)
    return dataset, X[n_entrenamiento:], ids


def entrenar_corrida(familia, nombre_corrida, semillas, carpeta_historia=HISTORIA):
    """Entrena las semillas en orden sobre un único dataset y guarda las probabilidades de agosto."""
    os.makedirs(PROBABILIDADES, exist_ok=True)
    salida = os.path.join(PROBABILIDADES, f"{familia}_{nombre_corrida}.npy")
    dataset, x_agosto, ids = preparar(familia, carpeta_historia)
    probabilidades = []
    for i, s in enumerate(semillas, 1):
        modelo = lgb.train({**PARAMETROS, "seed": s}, dataset, num_boost_round=ARBOLES)
        probabilidades.append(modelo.predict(x_agosto))
        print(f"semilla {s} lista ({i} de {len(semillas)})", flush=True)
    np.save(salida, np.array(probabilidades))
    np.save(os.path.join(PROBABILIDADES, "numero_de_cliente_agosto.npy"), ids)
    print(f"guardado: {salida}", flush=True)
