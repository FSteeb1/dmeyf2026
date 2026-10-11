# Paso 4: importancia de las features de historia, para elegir las 40 variables a las que agrego la ventana de 6 meses.
# Uso las 155 del paso 3 + lags y deltas de 1 y 2 meses de todas las variables (1_datos/construir_historia.py),
# más reglas y hojas de un random forest que probé en ese momento. Entreno el modelo de la familia historia con la
# primera semilla y guardo la importancia en trabajo/seleccion/paso_4/importancia_semilla1.csv.
# FEATURES_HISTORIA_40 (listas_de_variables.py) son las 40 variables cuyas columnas h_ suman más importancia.
# %%
import os, sys
ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
import os, sys, json, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from semillas import SEMILLA
from features_perfiles import agregar_features
import features_historia

BASE = os.path.dirname(os.path.dirname(ENTREGA))   # raíz del repo: ahí está data/competencia_01.csv
SAL = os.path.join(ENTREGA, "trabajo", "seleccion")
OUT = os.path.join(SAL, "paso_4"); os.makedirs(OUT, exist_ok=True)
semillas = np.random.default_rng(SEMILLA).choice(1_000_000, size=20, replace=False).tolist()[:10]
CORTES = [10_000, 11_000, 12_500, 14_000]
LENTO = dict(objective="binary", max_bin=31, min_data_in_leaf=0, feature_fraction=0.5, feature_fraction_bynode=0.2,
             learning_rate=0.005, num_leaves=63, min_sum_hessian_in_leaf=10.0, verbose=-1, num_threads=4)
RF = dict(objective="binary", boosting="rf", num_leaves=16, min_data_in_leaf=1000, bagging_fraction=0.632, bagging_freq=1,
          feature_fraction=0.3, max_bin=31, verbose=-1, num_threads=4, seed=SEMILLA)
MESES = [202103, 202104, 202105, 202106, 202108]

full = pd.read_csv(os.path.join(BASE, "data", "competencia_01.csv"))
full.loc[full.foto_mes == 202105, "ccajas_depositos"] = np.nan
full, _ = agregar_features(full)
sel = json.load(open(os.path.join(SAL, "paso_2", "seleccion.json"), encoding="utf-8"))["quedan"]
PREST = [f for f in sel if f.startswith(("mprestamos_personales", "cprestamos_personales"))]
t150 = pd.read_csv(os.path.join(SAL, "paso_3", "importancia_gain.csv"), index_col=0).index[:150].tolist()
f155 = t150 + [f for f in PREST if f not in t150]
AUX = ["active_quarter", "ctarjeta_master", "ccomisiones_mantenimiento", "Master_status", "sen_rojo_sin_mov", "vinculos",
       "cprestamos_personales", "mpasivos_margen", "ctrx_quarter", "cmobile_app_trx", "mcuentas_saldo", "cpayroll_trx",
       "mtarjeta_visa_consumo", "mcaja_ahorro", "mpayroll"]
full = full[full.foto_mes.isin(MESES)][["numero_de_cliente", "foto_mes", "clase_ternaria"] + list(dict.fromkeys(f155 + AUX))].reset_index(drop=True)
cl, H = features_historia.cargar(os.path.join(ENTREGA, "trabajo", "historia"), MESES)
H = H[[c for c in H.columns if c.endswith(("_lag1", "_lag2", "_delta1", "_delta2"))]]
full = full.merge(pd.concat([cl, H], axis=1), on=["numero_de_cliente", "foto_mes"], how="left"); del H, cl; gc.collect()
f755 = f155 + [c for c in full.columns if c.startswith("h_")]

# --- reglas
g = lambda c: pd.to_numeric(full[c], errors="coerce")
mov, sal, ah = g("ctrx_quarter"), g("mcuentas_saldo"), g("mcaja_ahorro")
su = g("cpayroll_trx").fillna(0) > 0; visa = g("mtarjeta_visa_consumo").fillna(0) > 0
sin_master = g("Master_status").isna(); rojo = g("sen_rojo_sin_mov") == 1
B1 = (mov <= 30) & (sal >= 0) & ~visa & ~su
B2 = (mov <= 40) & visa & (ah < 1000) & ~su
pay2, sal2, cpay2 = g("h_mpayroll_lag2"), g("h_mcuentas_saldo_lag2"), g("h_cpayroll_trx_lag2")
REGLAS = {
    "regla_R1": (g("active_quarter") == 0) & (g("ctarjeta_master").fillna(0) == 0) & (g("ccomisiones_mantenimiento") > 0),
    "regla_R2": sin_master & rojo & (g("vinculos") == 0),
    "regla_R3": (g("cprestamos_personales").fillna(0) == 0) & sin_master & rojo,
    "regla_R4": (g("mpasivos_margen") == 0) & (mov <= 27) & (g("cmobile_app_trx").fillna(0) == 0),
    "regla_perfil_A": (mov <= 10) & (sal < 0) & ~su,
    "regla_perfil_B1": B1, "regla_perfil_B2": B2, "regla_perfil_B": B1 | B2,
    "regla_sueldo_mitad_2m": (pay2 > 0) & (g("mpayroll").fillna(0) <= 0.5 * pay2),
    "regla_perdio_sueldo_y_mitad_saldo_2m": (cpay2 > 0) & ~su & (sal2 > 0) & (sal <= 0.5 * sal2),
}
for k, v in REGLAS.items():
    full[k] = v.fillna(False).astype(np.float32)
reglas = list(REGLAS)
tasa = full[full.foto_mes.isin([202103, 202104, 202105, 202106])].assign(b2=lambda d: d.clase_ternaria.eq("BAJA+2"))
print("Reglas (clientes y lift de BAJA+2 por mes de entrenamiento):", flush=True)
for k in reglas:
    print(f"  {k}: " + " | ".join(f"{m}: {int(t[k].sum())} cl, lift {t.loc[t[k] == 1, 'b2'].mean() / t.b2.mean():.1f}"
                                   for m, t in tasa.groupby("foto_mes")), flush=True)

# --- hojas de Random Forest
es_tr = full.foto_mes.isin([202103, 202104, 202105, 202106]).values
y = full.clase_ternaria.isin(["BAJA+1", "BAJA+2"]).astype(int).values
rf = lgb.train(RF, lgb.Dataset(full.loc[es_tr, f755], label=y[es_tr], params={"max_bin": 31}), num_boost_round=20)
hojas = rf.predict(full[f755], pred_leaf=True)          # (filas, 20) índice de hoja por árbol
hcols = {}
for t in range(hojas.shape[1]):
    for h in np.unique(hojas[:, t]):
        hcols[f"rfhoja_{t:02d}_{h:02d}"] = (hojas[:, t] == h).astype(np.float32)
full = pd.concat([full, pd.DataFrame(hcols)], axis=1); del hojas; gc.collect()
print("hojas de RF:", len(hcols), flush=True)

feats = f755 + reglas + list(hcols)
print("variables totales:", len(feats), flush=True)
pr = full[full.foto_mes == 202108]; ids = pr.numero_de_cliente.values.astype(np.int64); Xpr = pr[feats]
d = lgb.Dataset(full.loc[es_tr, feats], label=y[es_tr], params={"max_bin": 31}, free_raw_data=True)
del full; gc.collect()

m = lgb.train({**LENTO, "seed": semillas[0]}, d, num_boost_round=2000)
imp = pd.Series(m.feature_importance("gain"), index=feats); imp = imp / imp.sum()
imp.sort_values(ascending=False).to_csv(os.path.join(OUT, "importancia_semilla1.csv"))
