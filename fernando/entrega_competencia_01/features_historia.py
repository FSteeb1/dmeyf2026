"""Features de historia para todas las variables numéricas (prefijo h_).

- lag1, lag2: valor de hace 1 y 2 meses (mismo cliente, mes calendario exacto; si no estaba, queda NA)
- delta1, delta2: valor actual menos el lag
- en la ventana del mes actual y hasta 5 anteriores: tend (pendiente), min, max, ratioavg (actual / promedio),
  ratiomax (actual / máximo)
Solo mira hacia atrás. Se calcula por bloques de columnas y se guarda en parquet (float32) para no agotar la memoria.
Las divisiones por cero quedan NA.
"""
import os
import numpy as np
import pandas as pd

BLOQUE = 15
EXCLUIR = {"numero_de_cliente", "foto_mes", "clase_ternaria", "Master_fultimo_cierre", "Visa_fultimo_cierre"}


def construir(ruta_csv, carpeta_out):
    os.makedirs(carpeta_out, exist_ok=True)
    df = pd.read_csv(ruta_csv)
    df.loc[df.foto_mes == 202105, "ccajas_depositos"] = np.nan   # en 202105 viene toda en cero
    cols = [c for c in df.columns if c not in EXCLUIR and pd.api.types.is_numeric_dtype(df[c])]
    df[cols] = df[cols].astype(np.float32)
    idx = ((df.foto_mes // 100) * 12 + df.foto_mes % 100).values
    claves = pd.DataFrame({"numero_de_cliente": df.numero_de_cliente.values, "idx": idx})
    claves[["numero_de_cliente", "foto_mes"]] = df[["numero_de_cliente", "foto_mes"]].values
    claves[["numero_de_cliente", "foto_mes"]].to_parquet(os.path.join(carpeta_out, "claves.parquet"))
    x = -np.arange(6, dtype=np.float32)          # posición de cada mes de la ventana: 0, -1, ..., -5
    for b in range(0, len(cols), BLOQUE):
        bc = cols[b:b + BLOQUE]
        base = df[["numero_de_cliente"] + bc].assign(idx=idx)
        V = [df[bc].values]                       # V[k] = valor de hace k meses
        for k in range(1, 6):
            m = claves[["numero_de_cliente", "idx"]].merge(base.assign(idx=base.idx + k), on=["numero_de_cliente", "idx"], how="left")
            V.append(m[bc].values.astype(np.float32))
        V = np.stack(V)                           # (6, filas, columnas)
        nuevo = {}
        with np.errstate(all="ignore"):
            cur = V[0]
            for k in (1, 2):
                for j, c in enumerate(bc):
                    nuevo[f"h_{c}_lag{k}"] = V[k][:, j]; nuevo[f"h_{c}_delta{k}"] = cur[:, j] - V[k][:, j]
            ok = ~np.isnan(V); n = ok.sum(0)
            Vz = np.where(ok, V, 0.0)
            xs = np.where(ok, x[:, None, None], 0.0)
            mx = xs.sum(0) / np.maximum(n, 1); my = Vz.sum(0) / np.maximum(n, 1)
            cov = (np.where(ok, (x[:, None, None] - mx) * (V - my), 0.0)).sum(0)
            var = (np.where(ok, (x[:, None, None] - mx) ** 2, 0.0)).sum(0)
            tend = np.where((n >= 2) & (var > 0), cov / np.where(var > 0, var, 1), np.nan)
            vmin = np.nanmin(V, axis=0); vmax = np.nanmax(V, axis=0); vavg = np.nanmean(V, axis=0)
            ravg = np.where(np.abs(vavg) > 0, cur / np.where(vavg == 0, np.nan, vavg), np.nan)
            rmax = np.where(np.abs(vmax) > 0, cur / np.where(vmax == 0, np.nan, vmax), np.nan)
        for j, c in enumerate(bc):
            nuevo[f"h_{c}_tend"] = tend[:, j]; nuevo[f"h_{c}_min"] = vmin[:, j]; nuevo[f"h_{c}_max"] = vmax[:, j]
            nuevo[f"h_{c}_ratioavg"] = ravg[:, j]; nuevo[f"h_{c}_ratiomax"] = rmax[:, j]
        pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in nuevo.items()}).to_parquet(
            os.path.join(carpeta_out, f"bloque_{b // BLOQUE:03d}.parquet"))
        print(f"bloque {b // BLOQUE + 1}/{-(-len(cols) // BLOQUE)} listo ({len(nuevo)} columnas)", flush=True)
        del V, Vz, xs, nuevo
    return cols


def cargar(carpeta, meses, columnas=None):
    """Devuelve (claves, X) solo para los meses pedidos, concatenando los bloques en float32."""
    cl = pd.read_parquet(os.path.join(carpeta, "claves.parquet"))
    mask = cl.foto_mes.isin(meses).values
    partes = []
    for f in sorted(p for p in os.listdir(carpeta) if p.startswith("bloque_")):
        t = pd.read_parquet(os.path.join(carpeta, f), columns=columnas and [c for c in columnas if c in pd.read_parquet(os.path.join(carpeta, f)).columns])
        partes.append(t[mask].reset_index(drop=True))
    return cl[mask].reset_index(drop=True), pd.concat(partes, axis=1)
