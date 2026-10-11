# Paso 1: importancia (gain) de las variables originales y de las features de perfiles.
# Entreno con 202103 y 202104 (BAJA+1 y BAJA+2 como positivos), 10 semillas, y guardo la importancia promedio en
# trabajo/seleccion/paso_1/importancia.csv.
# %%
import os, sys
ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semillas import SEMILLA

BASE = os.path.dirname(os.path.dirname(ENTREGA))   # raíz del repo: ahí está data/competencia_01.csv
SAL = os.path.join(ENTREGA, "trabajo", "seleccion")
OUT = os.path.join(SAL, "paso_1")
os.makedirs(OUT, exist_ok=True)
GAN, COSTO = 1_072_500, 27_500
semillas = np.random.default_rng(SEMILLA).choice(1_000_000, size=10, replace=False).tolist()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, max_bin=31,
              verbose=-1, num_threads=4)
CORTES = list(range(6_000, 20_001, 500))

VARS_LAG = ["ctrx_quarter", "mcuentas_saldo", "mcaja_ahorro", "cpayroll_trx", "mpayroll",
            "mtarjeta_visa_consumo", "mtarjeta_master_consumo", "ctarjeta_debito_transacciones",
            "mcomisiones_mantenimiento", "mprestamos_personales", "mplazo_fijo_pesos",
            "mplazo_fijo_dolares", "chomebanking_transacciones", "mtransferencias_recibidas",
            "ctransferencias_recibidas", "mrentabilidad", "mactivos_margen", "mpasivos_margen",
            "cproductos", "Visa_msaldototal", "Master_msaldototal"]

# %%
df = pd.read_csv(os.path.join(BASE, "data", "competencia_01.csv"))
originales = [c for c in df.columns if c not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
df["idx_mes"] = (df.foto_mes // 100) * 12 + df.foto_mes % 100
base = df[["numero_de_cliente", "idx_mes"] + VARS_LAG]
nuevas = {}
for k in (1, 2):
    lag = base.assign(idx_mes=base.idx_mes + k).rename(columns={v: f"{v}_lag{k}" for v in VARS_LAG})
    m = df[["numero_de_cliente", "idx_mes"]].merge(lag, on=["numero_de_cliente", "idx_mes"], how="left")
    assert len(m) == len(df)
    for v in VARS_LAG:
        nuevas[f"{v}_lag{k}"] = m[f"{v}_lag{k}"].values
        nuevas[f"{v}_delta{k}"] = df[v].values - m[f"{v}_lag{k}"].values
for v in VARS_LAG:  # variación relativa contra el mes anterior (menos sensible a la inflación)
    l1 = nuevas[f"{v}_lag1"]
    nuevas[f"{v}_ratio1"] = np.where(np.abs(l1) > 0, df[v].values / np.where(l1 == 0, np.nan, l1), np.nan)

l1 = lambda v: nuevas[f"{v}_lag1"]
cero = lambda x: np.nan_to_num(x, nan=0.0)
pf = cero(df.mplazo_fijo_pesos.values) + cero(df.mplazo_fijo_dolares.values)
pf1 = cero(l1("mplazo_fijo_pesos")) + cero(l1("mplazo_fijo_dolares"))
sig = {
    "sen_corte_sueldo": (cero(df.cpayroll_trx.values) == 0) & (cero(l1("cpayroll_trx")) > 0),
    "sen_inicio_mantenimiento": (cero(df.mcomisiones_mantenimiento.values) > 0) & (cero(l1("mcomisiones_mantenimiento")) == 0) & ~np.isnan(l1("mcomisiones_mantenimiento")),
    "sen_desarme_pf": (pf == 0) & (pf1 > 0),
    "sen_fin_prestamo": (cero(df.mprestamos_personales.values) == 0) & (cero(l1("mprestamos_personales")) > 0),
    "sen_caida_mov10": nuevas["ctrx_quarter_ratio1"] <= 0.9,
    "sen_caida_mov30": nuevas["ctrx_quarter_ratio1"] <= 0.7,
    "sen_rojo_sin_mov": (df.mcuentas_saldo.values < 0) & (cero(df.ctrx_quarter.values) <= 10),
    "sen_tarjeta_cierre": (cero(df.Visa_status.values) > 0) | (cero(df.Master_status.values) > 0),
}
for k, v in sig.items():
    nuevas[k] = np.asarray(v, dtype=np.int8)
vinc = [df.cpayroll_trx > 0, df.ccuenta_debitos_automaticos > 0,
        (df.ctarjeta_visa_debitos_automaticos > 0) | (df.ctarjeta_master_debitos_automaticos > 0),
        (df.cpagomiscuentas > 0) | (df.cpagodeservicios > 0), df.cplazo_fijo > 0,
        (df[["cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"]].fillna(0).sum(1) > 0),
        (df[["cprestamos_personales", "cprestamos_prendarios", "cprestamos_hipotecarios"]].fillna(0).sum(1) > 0)]
nuevas["vinculos"] = np.sum([v.fillna(False).values for v in vinc], axis=0).astype(np.int8)
df = pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)
print("variables nuevas:", len(nuevas), "| total features:", len(originales) + len(nuevas))

# chequeos de no-fuga: el lag de 202103 tiene que estar vacío y el de un cliente debe ser su mes anterior
assert df.loc[df.foto_mes == 202103, "ctrx_quarter_lag1"].isna().all()
c = df.loc[df.foto_mes == 202105, "numero_de_cliente"].iloc[0]
a = df[(df.numero_de_cliente == c)].set_index("foto_mes")
assert a.loc[202105, "ctrx_quarter_lag1"] == a.loc[202104, "ctrx_quarter"]
print("señales, % en 202106:", {k: round(df.loc[df.foto_mes == 202106, k].mean() * 100, 2) for k in sig})

# %% validación
df["y"] = df["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).astype(int)
features = originales + list(nuevas)
tr = df[df.foto_mes.isin([202103, 202104])]
va = df[df.foto_mes == 202106]
clase_va = va["clase_ternaria"].values
dtrain = lgb.Dataset(tr[features], label=tr["y"], free_raw_data=False)


def gan_cortes(p):
    acum = np.cumsum(np.where(clase_va[np.argsort(-p)] == "BAJA+2", GAN, -COSTO))
    return {c: acum[c - 1] for c in CORTES}


filas, probs, imps = [], [], []
for s in semillas:
    m = lgb.train({**PARAMS, "seed": s}, dtrain, num_boost_round=400)
    p = m.predict(va[features]); probs.append(p)
    imps.append(pd.Series(m.feature_importance("gain"), index=features))
    filas += [{"semilla": s, "envios": c, "ganancia": g} for c, g in gan_cortes(p).items()]
    print("semilla", s, "lista", flush=True)
res = pd.DataFrame(filas); res.to_csv(os.path.join(OUT, "validacion_202106.csv"), index=False)
imp = pd.concat(imps, axis=1).mean(1).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT, "importancia.csv"))
