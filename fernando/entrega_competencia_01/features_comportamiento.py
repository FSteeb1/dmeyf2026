"""Features de comportamiento del cliente (ventana de 4 meses, solo mira hacia atrás).

1. Contra su propio pasado, para todas las variables: meses desde el máximo, racha de caídas, coeficiente de variación.
2. Flujos de dinero del mes: entradas, salidas, salidas / entradas, saldo / entradas.
3. Cartera: productos activos, canales usados, productos perdidos respecto del máximo de 4 meses.
4. Relaciones entre productos del mismo mes: consumo / límite de tarjeta, comisiones / saldo, consumo / sueldo, etc.
Para 2, 3 y 4 también: valor del mes anterior, cambio, mínimo y máximo de la ventana.
"""
import numpy as np
import pandas as pd

VENTANA = 4
EXCLUIR = {"numero_de_cliente", "foto_mes", "clase_ternaria", "Master_fultimo_cierre", "Visa_fultimo_cierre",
           "Master_Fvencimiento", "Visa_Fvencimiento", "Master_fechaalta", "Visa_fechaalta", "cliente_edad", "cliente_antiguedad"}


def _z(df, c):
    return df[c].fillna(0).astype(np.float64) if c in df else pd.Series(0.0, index=df.index)


def _div(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(np.abs(b) > 0, a / np.where(b == 0, np.nan, b), np.nan).astype(np.float32)


def intrames(df):
    """Variables del mes (ideas 2, 3 y 4). df = crudo con todas las filas."""
    z = lambda c: _z(df, c)
    entradas = z("mpayroll") + z("mpayroll2") + z("mtransferencias_recibidas") + z("mcheques_depositados")
    salidas = (z("mtransferencias_emitidas") + z("mextraccion_autoservicio") + z("matm") + z("matm_other") + z("mpagodeservicios")
               + z("mpagomiscuentas") + z("mcuenta_debitos_automaticos") + z("mttarjeta_visa_debitos_automaticos")
               + z("mttarjeta_master_debitos_automaticos") + z("mcheques_emitidos"))
    prods = [z("ctarjeta_visa") > 0, z("ctarjeta_master") > 0, z("cprestamos_personales") > 0, z("cprestamos_prendarios") > 0,
             z("cprestamos_hipotecarios") > 0, z("cplazo_fijo") > 0, (z("cinversion1") + z("cinversion2")) > 0,
             (z("cseguro_vida") + z("cseguro_auto") + z("cseguro_vivienda") + z("cseguro_accidentes_personales")) > 0,
             z("ccaja_seguridad") > 0, z("ccuenta_debitos_automaticos") > 0, z("cpayroll_trx") > 0, z("mcaja_ahorro_dolares") > 0]
    canales = [z("chomebanking_transacciones") > 0, z("cmobile_app_trx") > 0, (z("catm_trx") + z("catm_trx_other")) > 0,
               z("ccajas_transacciones") > 0, z("ccallcenter_transacciones") > 0, z("ctarjeta_debito_transacciones") > 0]
    cons_t = z("mtarjeta_visa_consumo") + z("mtarjeta_master_consumo")
    out = {
        "x_entradas": entradas.astype(np.float32), "x_salidas": salidas.astype(np.float32),
        "x_salidas_sobre_entradas": _div(salidas, entradas), "x_saldo_sobre_entradas": _div(z("mcuentas_saldo"), entradas),
        "x_productos_activos": np.sum(prods, axis=0).astype(np.float32), "x_canales_usados": np.sum(canales, axis=0).astype(np.float32),
        "x_visa_consumo_sobre_limite": _div(z("mtarjeta_visa_consumo"), z("Visa_mlimitecompra")),
        "x_master_consumo_sobre_limite": _div(z("mtarjeta_master_consumo"), z("Master_mlimitecompra")),
        "x_visa_saldo_sobre_limite": _div(z("Visa_msaldototal"), z("Visa_mlimitecompra")),
        "x_comisiones_sobre_saldo": _div(z("mcomisiones"), np.abs(z("mcuentas_saldo")) + 1),
        "x_comisiones_sobre_rentabilidad": _div(z("mcomisiones"), np.abs(z("mrentabilidad")) + 1),
        "x_mantenimiento_sobre_rentabilidad": _div(z("mcomisiones_mantenimiento"), np.abs(z("mrentabilidad")) + 1),
        "x_pasivos_sobre_activos_margen": _div(z("mpasivos_margen"), np.abs(z("mactivos_margen")) + 1),
        "x_consumo_tarjeta_sobre_sueldo": _div(cons_t, z("mpayroll") + z("mpayroll2")),
        "x_saldo_sobre_sueldo": _div(z("mcuentas_saldo"), z("mpayroll") + z("mpayroll2")),
        "x_trx_por_producto": _div(z("ctrx_quarter"), np.sum(prods, axis=0) + 1),
        "x_pagos_tarjeta_sobre_saldo_tarjeta": _div(z("Visa_mpagado") + z("Master_mpagado"), z("Visa_msaldototal") + z("Master_msaldototal")),
    }
    return pd.DataFrame(out, index=df.index)


def _ventana(df, cols):
    """V[k] = valor de hace k meses (k = 0..VENTANA-1), alineado a las filas de df."""
    idx = ((df.foto_mes // 100) * 12 + df.foto_mes % 100).values
    k0 = pd.DataFrame({"numero_de_cliente": df.numero_de_cliente.values, "idx": idx})
    base = df[["numero_de_cliente"] + cols].assign(idx=idx)
    V = [df[cols].values.astype(np.float32)]
    for k in range(1, VENTANA):
        V.append(k0.merge(base.assign(idx=base.idx + k), on=["numero_de_cliente", "idx"], how="left")[cols].values.astype(np.float32))
    return np.stack(V)                                            # (VENTANA, filas, columnas)


def construir(crudo):
    """Devuelve DataFrame (numero_de_cliente, foto_mes, x_* ...) para todas las filas del crudo."""
    crudo = crudo.reset_index(drop=True)
    base_cols = [c for c in crudo.columns if c not in EXCLUIR and pd.api.types.is_numeric_dtype(crudo[c])]
    out = {}
    # 1. contra el propio pico, para todas las variables del crudo (en bloques)
    for b in range(0, len(base_cols), 25):
        cs = base_cols[b:b + 25]
        V = _ventana(crudo, cs); cur = V[0]
        with np.errstate(all="ignore"):
            ok = ~np.isnan(V)
            Vm = np.where(ok, V, -np.inf)
            desde_max = np.argmax(Vm, axis=0).astype(np.float32)          # 0 = el máximo es hoy
            desde_max[~ok.any(0)] = np.nan
            baja = (V[:-1] < V[1:])                                       # V[k] < V[k+1]: bajó del mes k+1 al k
            racha = np.zeros(cur.shape, dtype=np.float32); sigue = np.ones(cur.shape, dtype=bool)
            for k in range(VENTANA - 1):
                sigue &= baja[k]; racha += sigue
            n = ok.sum(0); media = np.nanmean(V, axis=0); desv = np.nanstd(V, axis=0)
            cv = np.where((n >= 2) & (np.abs(media) > 0), desv / np.abs(np.where(media == 0, np.nan, media)), np.nan).astype(np.float32)
        for j, c in enumerate(cs):
            out[f"x_{c}_meses_desde_max"] = desde_max[:, j]; out[f"x_{c}_racha_caidas"] = racha[:, j]; out[f"x_{c}_cv"] = cv[:, j]
    # 2–4. variables intrames y su historia
    I = intrames(crudo)
    I = pd.concat([crudo[["numero_de_cliente", "foto_mes"]], I], axis=1)
    icols = [c for c in I.columns if c.startswith("x_")]
    V = _ventana(I, icols); cur = V[0]
    with np.errstate(all="ignore"):
        vmin, vmax = np.nanmin(V, axis=0), np.nanmax(V, axis=0)
    for j, c in enumerate(icols):
        out[c] = cur[:, j]; out[f"{c}_lag1"] = V[1][:, j]; out[f"{c}_delta1"] = cur[:, j] - V[1][:, j]
        out[f"{c}_min4"] = vmin[:, j]; out[f"{c}_max4"] = vmax[:, j]
    prod_max = np.nanmax(V[:, :, icols.index("x_productos_activos")], axis=0)
    out["x_productos_perdidos_vs_max4"] = (prod_max - cur[:, icols.index("x_productos_activos")]).astype(np.float32)
    res = pd.DataFrame(out); res.insert(0, "foto_mes", crudo.foto_mes.values); res.insert(0, "numero_de_cliente", crudo.numero_de_cliente.values)
    return res
