"""Arma la entrega final: promedio de las probabilidades de los 27 modelos y los 11.000 clientes con mayor probabilidad.

Lee las probabilidades de ../probabilidades (las que dejan los entrenar.py de 3_modelos), escribe
salida/prediccion_final.csv, valida el formato y compara su SHA-256 con el del archivo que entregué.
Uso: python armar_ensamble.py
"""
import hashlib
import json
import os
import platform

import lightgbm
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
PROBABILIDADES = os.path.join(os.path.dirname(AQUI), "probabilidades")
SALIDA = os.path.join(AQUI, "salida")
CORTE = 11_000
SHA256_ENTREGADO = "2a5a508683301735e4f4b5ab07ccc50c102459479691baf9211e68c3e496663a"

# en este orden se promediaron para la entrega (el orden cambia los últimos decimales del promedio)
CORRIDAS = [
    "historia_comportamiento_corrida_1",
    "historia_comportamiento_corrida_2",
    "historia_comportamiento_corrida_3",
    "historia_comportamiento_corrida_4",
    "historia_comportamiento_flujos_corrida_1",
    "historia_comportamiento_flujos_corrida_2",
    "historia_comportamiento_flujos_corrida_3",
    "historia_corrida_1",
]

ids = np.load(os.path.join(PROBABILIDADES, "numero_de_cliente_agosto.npy"))
modelos = np.vstack([np.load(os.path.join(PROBABILIDADES, f"{c}.npy")) for c in CORRIDAS])
assert modelos.shape == (27, len(ids)), modelos.shape
promedio = modelos.mean(0)
orden = np.lexsort((ids, -promedio))   # mayor probabilidad primero, empate por numero_de_cliente
elegidos = ids[orden][:CORTE]

os.makedirs(SALIDA, exist_ok=True)
archivo = os.path.join(SALIDA, "prediccion_final.csv")
pd.Series(elegidos).to_csv(archivo, index=False, header=False)

texto = open(archivo, encoding="utf-8").read()
lineas = texto.split()
assert len(lineas) == CORTE and len(set(lineas)) == CORTE, "cantidad de clientes o repetidos"
assert all(x.isdigit() for x in lineas), "hay valores que no son enteros"
assert set(map(int, lineas)) <= set(ids.tolist()), "hay clientes que no son de 202108"

sha = hashlib.sha256(open(archivo, "rb").read()).hexdigest()
json.dump({"modelos": int(modelos.shape[0]), "corte": CORTE, "sha256": sha, "python": platform.python_version(),
           "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "sistema": platform.platform(), "nucleos": os.cpu_count()},
          open(os.path.join(SALIDA, "meta.json"), "w", encoding="utf-8"), indent=1)

print(f"archivo generado: {archivo} ({CORTE} clientes)")
print(f"SHA-256 generado:  {sha}")
print(f"SHA-256 entregado: {SHA256_ENTREGADO}")
print(f"coinciden: {'sí' if sha == SHA256_ENTREGADO else 'NO'}")
