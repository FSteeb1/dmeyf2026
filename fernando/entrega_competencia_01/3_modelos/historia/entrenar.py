"""Entrena los modelos de la familia historia y guarda sus probabilidades para agosto en ../../probabilidades.

Uso:
    python entrenar.py                 corre todas las corridas, en orden
    python entrenar.py --corrida corrida_2    corre solo esa corrida
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comun import HISTORIA, entrenar_corrida

FAMILIA = "historia"
CORRIDAS = {
    "corrida_1": [975935, 98956, 619348, 491242, 983565],
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corrida", choices=list(CORRIDAS))
    p.add_argument("--historia", default=HISTORIA)
    a = p.parse_args()
    for nombre, semillas in CORRIDAS.items():
        if a.corrida in (None, nombre):
            entrenar_corrida(FAMILIA, nombre, semillas, a.historia)
