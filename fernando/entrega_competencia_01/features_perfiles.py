"""Features de perfiles: lags, deltas y ratio de 1 y 2 meses para 21 variables, señales de baja y cantidad de vínculos.

agregar_features(df) devuelve (df con las features, lista de features nuevas).
Los lags se toman por mes calendario: si el cliente no estaba ese mes, quedan vacíos.
"""
import numpy as np
import pandas as pd

VARS_LAG = ["ctrx_quarter", "mcuentas_saldo", "mcaja_ahorro", "cpayroll_trx", "mpayroll",
            "mtarjeta_visa_consumo", "mtarjeta_master_consumo", "ctarjeta_debito_transacciones",
            "mcomisiones_mantenimiento", "mprestamos_personales", "mplazo_fijo_pesos",
            "mplazo_fijo_dolares", "chomebanking_transacciones", "mtransferencias_recibidas",
            "ctransferencias_recibidas", "mrentabilidad", "mactivos_margen", "mpasivos_margen",
            "cproductos", "Visa_msaldototal", "Master_msaldototal"]


def agregar_features(df):
    idx = (df.foto_mes // 100) * 12 + df.foto_mes % 100
    base = df[["numero_de_cliente"] + VARS_LAG].assign(idx_mes=idx.values)
    keys = pd.DataFrame({"numero_de_cliente": df.numero_de_cliente.values, "idx_mes": idx.values})
    nuevas = {}
    for k in (1, 2):
        lag = base.assign(idx_mes=base.idx_mes + k).rename(columns={v: f"{v}_lag{k}" for v in VARS_LAG})
        m = keys.merge(lag, on=["numero_de_cliente", "idx_mes"], how="left")
        assert len(m) == len(df)
        for v in VARS_LAG:
            nuevas[f"{v}_lag{k}"] = m[f"{v}_lag{k}"].values
            nuevas[f"{v}_delta{k}"] = df[v].values - m[f"{v}_lag{k}"].values
    for v in VARS_LAG:
        l1 = nuevas[f"{v}_lag1"]
        nuevas[f"{v}_ratio1"] = np.where(np.abs(l1) > 0, df[v].values / np.where(l1 == 0, np.nan, l1), np.nan)

    cero = lambda x: np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    l1 = lambda v: nuevas[f"{v}_lag1"]
    pf = cero(df.mplazo_fijo_pesos) + cero(df.mplazo_fijo_dolares)
    pf1 = cero(l1("mplazo_fijo_pesos")) + cero(l1("mplazo_fijo_dolares"))
    sig = {
        "sen_corte_sueldo": (cero(df.cpayroll_trx) == 0) & (cero(l1("cpayroll_trx")) > 0),
        "sen_inicio_mantenimiento": (cero(df.mcomisiones_mantenimiento) > 0) & (cero(l1("mcomisiones_mantenimiento")) == 0) & ~np.isnan(l1("mcomisiones_mantenimiento")),
        "sen_desarme_pf": (pf == 0) & (pf1 > 0),
        "sen_fin_prestamo": (cero(df.mprestamos_personales) == 0) & (cero(l1("mprestamos_personales")) > 0),
        "sen_caida_mov10": nuevas["ctrx_quarter_ratio1"] <= 0.9,
        "sen_caida_mov30": nuevas["ctrx_quarter_ratio1"] <= 0.7,
        "sen_rojo_sin_mov": (df.mcuentas_saldo.values < 0) & (cero(df.ctrx_quarter) <= 10),
        "sen_tarjeta_cierre": (cero(df.Visa_status) > 0) | (cero(df.Master_status) > 0),
    }
    for k, v in sig.items():
        nuevas[k] = np.asarray(v, dtype=np.int8)
    vinc = [df.cpayroll_trx > 0, df.ccuenta_debitos_automaticos > 0,
            (df.ctarjeta_visa_debitos_automaticos > 0) | (df.ctarjeta_master_debitos_automaticos > 0),
            (df.cpagomiscuentas > 0) | (df.cpagodeservicios > 0), df.cplazo_fijo > 0,
            df[["cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"]].fillna(0).sum(1) > 0,
            df[["cprestamos_personales", "cprestamos_prendarios", "cprestamos_hipotecarios"]].fillna(0).sum(1) > 0]
    nuevas["vinculos"] = np.sum([v.fillna(False).values for v in vinc], axis=0).astype(np.int8)
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(nuevas)], axis=1)
    return out, list(nuevas)
