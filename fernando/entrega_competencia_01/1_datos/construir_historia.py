"""Construye las features de historia (h_*) para todas las variables y las guarda en ../trabajo/historia.

Necesita data/competencia_01.csv en la raíz del repo (lo genera construccion_target.ipynb).
Uso: python construir_historia.py
"""
import os
import sys

ENTREGA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENTREGA)
from features_historia import construir

DATOS = os.path.join(os.path.dirname(os.path.dirname(ENTREGA)), "data", "competencia_01.csv")
construir(DATOS, os.path.join(ENTREGA, "trabajo", "historia"))
