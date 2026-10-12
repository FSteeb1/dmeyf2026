"""Deja las filas de data/competencia_01.csv en el orden con el que entrené los modelos.

construccion_target.ipynb escribe siempre las mismas filas, pero DuckDB las escribe en paralelo y el orden cambia de una
corrida a otra. Dentro de cada mes, los modelos reciben las filas en el orden del archivo, y LightGBM da probabilidades
distintas si cambia ese orden. orden_filas.csv.gz tiene numero_de_cliente y foto_mes de cada fila, en el orden del
archivo que usé. Este script reescribe el archivo en ese orden y verifica que quede idéntico al original.

Necesita data/competencia_01.csv en la raíz del repo (lo genera construccion_target.ipynb).
Uso: python ordenar_filas.py
"""
import gzip
import hashlib
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(AQUI))), "data", "competencia_01.csv")
ORDEN = os.path.join(AQUI, "orden_filas.csv.gz")
SHA256_ESPERADO = "5a33cf43d805f411bde1a65bc86e1df8311d2660c2b162e47f2a78985155b87d"


def clave(linea):
    """numero_de_cliente y foto_mes, las dos primeras columnas de la fila."""
    return b",".join(linea.split(b",", 2)[:2])


with open(DATOS, "rb") as f:
    encabezado = f.readline()
    filas = {clave(linea): linea for linea in f}

with gzip.open(ORDEN, "rb") as f:
    assert f.readline().strip() == b"numero_de_cliente,foto_mes"
    orden = [linea.strip() for linea in f]

assert len(orden) == len(filas) and set(orden) == set(filas), "las filas no coinciden con orden_filas.csv.gz"

with open(DATOS, "wb") as f:
    f.write(encabezado)
    f.writelines(filas[k] for k in orden)

sha = hashlib.sha256(open(DATOS, "rb").read()).hexdigest()
print(f"SHA-256 de competencia_01.csv: {sha}")
print(f"coincide con el que usé: {'sí' if sha == SHA256_ESPERADO else 'NO'}")
if sha != SHA256_ESPERADO:
    raise SystemExit("competencia_01.csv no quedó idéntico al que usé")
