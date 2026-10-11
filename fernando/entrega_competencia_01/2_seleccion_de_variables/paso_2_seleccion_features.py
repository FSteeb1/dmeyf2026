# Paso 2: saco las features que no usó ningún modelo del paso 1 y, de cada par con correlación de Spearman mayor a
# 0,95, me quedo con una. Guarda la lista en trabajo/seleccion/paso_2/seleccion.json.
# %%
import os, sys
ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
import os, sys, json
import numpy as np
import pandas as pd

from semillas import SEMILLA
from features_perfiles import agregar_features

BASE = os.path.dirname(os.path.dirname(ENTREGA))   # raíz del repo: ahí está data/competencia_01.csv
SAL = os.path.join(ENTREGA, "trabajo", "seleccion")
OUT = os.path.join(SAL, "paso_2"); os.makedirs(OUT, exist_ok=True)
UMBRAL = 0.95

df = pd.read_csv(os.path.join(BASE, "data", "competencia_01.csv"))
originales = [c for c in df.columns if c not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]]
df, nuevas = agregar_features(df)
features = originales + nuevas
imp = pd.read_csv(os.path.join(SAL, "paso_1", "importancia.csv"), index_col=0).iloc[:, 0].reindex(features).fillna(0)


def tipo(f):
    if f in originales: return 0
    if f.startswith("sen_") or f == "vinculos": return 1
    if "_lag" in f: return 2
    if "_delta" in f: return 3
    return 4  # ratio


sin_uso = [f for f in features if imp[f] == 0]
cand = [f for f in features if f not in sin_uso]
muestra = df[df.foto_mes.isin([202103, 202104])].sample(100_000, random_state=SEMILLA)[cand]
const = [c for c in cand if muestra[c].nunique(dropna=True) <= 1]
cand = [c for c in cand if c not in const]
corr = muestra[cand].rank().corr(method="pearson").abs()  
orden = sorted(cand, key=lambda f: (tipo(f), -imp[f]))
def r_no_ceros(a, b):
    """Spearman solo en filas donde al menos una de las dos es distinta de 0 (y ninguna NaN).
    Evita que dos variables 'casi todo cero' parezcan redundantes solo por compartir ceros."""
    x, y = muestra[a], muestra[b]
    m = x.notna() & y.notna() & ((x != 0) | (y != 0))
    if m.sum() < 500: return 0.0
    return abs(x[m].rank().corr(y[m].rank()))


quedan, sacadas, descartes_por_ceros = [], [], []
for f in orden:
    par, r = None, None
    for q in quedan:
        if corr.loc[f, q] > UMBRAL:            
            rr = r_no_ceros(f, q)             
            if rr > UMBRAL: par, r = q, rr; break
            descartes_por_ceros.append((f, q, round(corr.loc[f, q], 3), round(rr, 3)))
    if par is None: quedan.append(f)
    else: sacadas.append({"variable": f, "redundante_con": par, "r_sin_ceros": round(r, 3)})

red = pd.DataFrame(sacadas)
print(f"Total variables: {len(features)} ({len(originales)} originales + {len(nuevas)} nuevas)")
print(f"Sin uso en ninguna semilla: {len(sin_uso)} -> {sin_uso}")
print(f"Constantes en entrenamiento: {len(const)} -> {const}")
print(f"Redundantes (|r|>{UMBRAL}): {len(red)}")
print(red.to_string(index=False))
print(f"\nPares que parecían redundantes solo por los ceros (se conservan ambas): {len(descartes_por_ceros)}")
for d in descartes_por_ceros: print("  ", d)
print(f"QUEDAN: {len(quedan)} | de ellas nuevas: {sum(f in nuevas for f in quedan)} | originales: {sum(f in originales for f in quedan)}")
json.dump({"quedan": quedan, "sin_uso": sin_uso, "constantes": const, "redundantes": sacadas},
          open(os.path.join(OUT, "seleccion.json"), "w"), indent=1, ensure_ascii=False)
