# Primera Competencia DMEyF 2026 — Fernando Steeb

Entrega final para la competencia 1: un ensamble a partir del promedio de las probabilidades de 27 modelos LightGBM, tomando como punto de corte los 11.000 clientes con mayor probabilidad.

| | |
|---|---|
| Archivo | `4_ensamble/salida/prediccion_final.csv` (lo genera `armar_ensamble.py`) |
| Clientes | 11.000, sin encabezado, uno por línea |
| SHA-256 | `2a5a508683301735e4f4b5ab07ccc50c102459479691baf9211e68c3e496663a` |

## Descripción del ensamble final

Entreno con marzo, abril, mayo y junio, con BAJA+1 y BAJA+2 como positivos. Todos los modelos usan los mismos hiperparámetros (están en `3_modelos/comun.py`). Solo cambia la semilla. 

Los 27 modelos se reparten en tres familias. Cada una agrega features a la anterior:

- **historia** (5 modelos): las features de perfiles (`features_perfiles.py`), los lags y deltas de 1 y 2 meses de todas las variables, el mínimo, el máximo y la tendencia de los últimos 6 meses para 40 variables, y el ranking de los montos dentro de cada mes.
- **historia_comportamiento** (9 modelos): lo anterior más features de cada cliente contra su propio pasado, de entradas y salidas de dinero, de cantidad de productos y canales, y relaciones entre productos del mismo mes (`features_comportamiento.py`).
- **historia_comportamiento_flujos** (13 modelos): lo anterior más las entradas y salidas desglosadas por tipo, el saldo neto y qué producto dejó de tener (`features_flujos.py`).

Para facilitar la comparación entre meses y controlar el drift producto de la inflación y el aguinaldo de junio, agregué una copia de cada monto con su posición dentro del mes: los positivos entre 0 y 1, los negativos entre −1 y 0, y el cero queda en cero.

## Cómo replicar la entrega

Usé Python 3.12.10 en Windows, con 4 núcleos (`num_threads = 4`) y las versiones de `requirements.txt`:

```
pip install -r requirements.txt
```

1. **Datos.** Correr `1_datos/construccion_target.ipynb`. Genera `data/competencia_01.csv` con `clase_ternaria`. El archivo que usé tiene este SHA-256: `5a33cf43d805f411bde1a65bc86e1df8311d2660c2b162e47f2a78985155b87d`.
2. **Features de historia.** `python 1_datos/construir_historia.py`. Deja los parquet en `trabajo/historia/`.
3. **Modelos.** En cada carpeta de `3_modelos/`, `python entrenar.py`. Cada familia de modelos se entrenó en varias corridas (están en `CORRIDAS`, dentro de cada `entrenar.py`). Cada corrida arma el dataset de LightGBM con la primera semilla que entrena, así que para obtener exactamente las mismas probabilidades hay que respetar las corridas y el orden de las semillas. Las probabilidades quedan en `probabilidades/`.
4. **Ensamble.** `python 4_ensamble/armar_ensamble.py`. Promedia los 27 modelos, toma los 11.000 clientes con mayor probabilidad (ante empate, por `numero_de_cliente`), valida el formato y compara el SHA-256 con el de la entrega.

Las probabilidades de los 27 modelos ya están en `probabilidades/`, así que el paso 4 se puede correr directamente. Para comprobar que el código reproduce uno de los modelos sin reentrenar los 27, `3_modelos/verificar_semilla.py` reentrena la primera semilla de una corrida y la compara con la guardada:

```
python 3_modelos/verificar_semilla.py historia corrida_1
```

## Aclaración sobre las listas de features

`listas_de_variables.py` tiene las dos listas fijas que usan los modelos: las 155 features base y las 40 variables con historia de 6 meses. En `2_seleccion_de_variables/` están ordenados los pasos con los que llegué a esas listas (`paso_1` a `paso_4`). Los dejé por si se quiere revisar cómo fue el proceso, pero no llegué a chequear si se reproducen antes de la entrega así que aún no puedo garantizar que produzcan exactamente el mismo resultado. 

## Carpetas

```
1_datos/                      clase_ternaria y features de historia
2_seleccion_de_variables/     pasos con los que armé las listas de features
3_modelos/                    comun.py y una carpeta por familia, con su entrenar.py
4_ensamble/                   promedio de los 27 modelos, corte y verificación
probabilidades/               probabilidades de agosto de cada corrida
features_*.py                 construcción de las features
listas_de_variables.py        listas fijas de features y semillas
semillas.py                   semilla general
```
