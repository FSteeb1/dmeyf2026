"""Reentrena la primera semilla de una corrida y la compara con las probabilidades guardadas en ../probabilidades.

Sirve para comprobar que el código de esta carpeta reproduce los modelos sin tener que reentrenar los 27.
Uso: python verificar_semilla.py historia corrida_1 [--historia CARPETA]
"""
import argparse
import importlib.util
import os
import sys

import lightgbm as lgb
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from comun import ARBOLES, HISTORIA, PARAMETROS, PROBABILIDADES, preparar

p = argparse.ArgumentParser()
p.add_argument("familia")
p.add_argument("corrida")
p.add_argument("--historia", default=HISTORIA)
a = p.parse_args()

spec = importlib.util.spec_from_file_location("entrenar", os.path.join(AQUI, a.familia, "entrenar.py"))
entrenar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrenar)
semilla = entrenar.CORRIDAS[a.corrida][0]

dataset, x_agosto, ids = preparar(a.familia, a.historia)
nueva = lgb.train({**PARAMETROS, "seed": semilla}, dataset, num_boost_round=ARBOLES).predict(x_agosto)
guardada = np.load(os.path.join(PROBABILIDADES, f"{a.familia}_{a.corrida}.npy"))[0]
assert np.array_equal(ids, np.load(os.path.join(PROBABILIDADES, "numero_de_cliente_agosto.npy")))
print(f"{a.familia} {a.corrida} semilla {semilla}")
print(f"diferencia máxima con la guardada: {np.abs(nueva - guardada).max():.3e}")
print(f"idénticas: {'sí' if np.array_equal(nueva, guardada) else 'NO'}")
