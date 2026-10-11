# Paso 3: ranking de importancia de las features del paso 2 (sin las de préstamos personales). Entreno con 202104,
# 20 semillas, y guardo la importancia promedio en trabajo/seleccion/paso_3/importancia_gain.csv.
# FEATURES_BASE_155 (listas_de_variables.py) son las 150 primeras de este ranking más las 5 de préstamos personales.
# %%
import os, sys
ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
import os, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semillas import SEMILLA
from features_perfiles import agregar_features

BASE = os.path.dirname(os.path.dirname(ENTREGA))   # raíz del repo: ahí está data/competencia_01.csv
SAL = os.path.join(ENTREGA, "trabajo", "seleccion")
OUT = os.path.join(SAL, "paso_3"); os.makedirs(OUT, exist_ok=True)
GAN, COSTO = 1_072_500, 27_500
semillas = np.random.default_rng(SEMILLA).choice(1_000_000, size=20, replace=False).tolist()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, max_bin=31,
              verbose=-1, num_threads=4)
CORTES = list(range(8_000, 17_001, 500))
TAMANOS = [25, 50, 75, 100, 150]

df = pd.read_csv(os.path.join(BASE, "data", "competencia_01.csv"))
df, nuevas = agregar_features(df)
df = df.copy()
df["y"] = df["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).astype(int)
sel = json.load(open(os.path.join(SAL, "paso_2", "seleccion.json"), encoding="utf-8"))["quedan"]
base = [f for f in sel if f not in ("Master_fultimo_cierre", "Visa_fultimo_cierre")]
sinp = [f for f in base if not f.startswith(("mprestamos_personales", "cprestamos_personales"))]
tr = df[df.foto_mes == 202104]; va = df[df.foto_mes == 202106]; clase = va["clase_ternaria"].values


def gan(p):
    acum = np.cumsum(np.where(clase[np.argsort(-p)] == "BAJA+2", GAN, -COSTO))
    return {c: acum[c - 1] for c in CORTES}


res, imp = [], []
def correr(nombre, feats, guardar_imp=False):
    d = lgb.Dataset(tr[feats], label=tr["y"], free_raw_data=False)
    for s in semillas:
        m = lgb.train({**PARAMS, "seed": s}, d, num_boost_round=400)
        if guardar_imp: imp.append(pd.Series(m.feature_importance("gain"), index=feats))
        res.extend({"n": nombre, "semilla": s, "envios": c, "ganancia": g} for c, g in gan(m.predict(va[feats])).items())
    print(nombre, "listo", flush=True)


correr(f"todas ({len(sinp)})", sinp, guardar_imp=True)
orden = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)
orden.to_csv(os.path.join(OUT, "importancia_gain.csv"))
print("variables con importancia 0:", int((orden == 0).sum()))
