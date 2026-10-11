"""Features de flujos de dinero (4 meses).

1. Entradas por tipo (sueldo, transferencias, cheques) y salidas por tipo (transferencias, cajero, débitos automáticos,
   pagos de servicios, cheques). Neto = entradas - salidas, y neto acumulado en la ventana de tiempo.
2. Para entradas, salidas y neto: tendencia, actual / máximo y actual / promedio de la ventana de tiempo.
3. Qué producto dejó de tener (lo tuvo en la ventana y hoy no).
"""
import numpy as np
import pandas as pd

from features_comportamiento import VENTANA, _z, _div, _ventana


def _base(df):
    z = lambda c: _z(df, c)
    comp = {
        "x2_ent_sueldo": z("mpayroll") + z("mpayroll2"),
        "x2_ent_transf": z("mtransferencias_recibidas"),
        "x2_ent_cheques": z("mcheques_depositados"),
        "x2_sal_transf": z("mtransferencias_emitidas"),
        "x2_sal_cajero": z("mextraccion_autoservicio") + z("matm") + z("matm_other"),
        "x2_sal_debitos": z("mcuenta_debitos_automaticos") + z("mttarjeta_visa_debitos_automaticos") + z("mttarjeta_master_debitos_automaticos"),
        "x2_sal_servicios": z("mpagodeservicios") + z("mpagomiscuentas"),
        "x2_sal_cheques": z("mcheques_emitidos"),
    }
    ent = comp["x2_ent_sueldo"] + comp["x2_ent_transf"] + comp["x2_ent_cheques"]
    sal = sum(v for k, v in comp.items() if k.startswith("x2_sal_"))
    comp["x2_entradas"], comp["x2_salidas"], comp["x2_neto"] = ent, sal, ent - sal
    prods = {"visa": z("ctarjeta_visa") > 0, "master": z("ctarjeta_master") > 0, "prestamo": (z("cprestamos_personales") + z("cprestamos_prendarios") + z("cprestamos_hipotecarios")) > 0,
             "plazo_fijo": z("cplazo_fijo") > 0, "inversion": (z("cinversion1") + z("cinversion2")) > 0,
             "seguro": (z("cseguro_vida") + z("cseguro_auto") + z("cseguro_vivienda") + z("cseguro_accidentes_personales")) > 0,
             "caja_seguridad": z("ccaja_seguridad") > 0, "debito_auto": z("ccuenta_debitos_automaticos") > 0, "sueldo": z("cpayroll_trx") > 0,
             "dolares": z("mcaja_ahorro_dolares") > 0}
    for k, v in prods.items():
        comp[f"x2_tiene_{k}"] = v.astype(np.float64)
    comp["x2_productos"] = np.sum([v for v in prods.values()], axis=0).astype(np.float64)
    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in comp.items()})
    out.insert(0, "foto_mes", df.foto_mes.values); out.insert(0, "numero_de_cliente", df.numero_de_cliente.values)
    return out


def construir2(crudo):
    crudo = crudo.reset_index(drop=True)
    B = _base(crudo)
    montos = [c for c in B.columns if c.startswith(("x2_ent_", "x2_sal_")) or c in ("x2_entradas", "x2_salidas", "x2_neto")]
    tiene = [c for c in B.columns if c.startswith("x2_tiene_")]
    out = {c: B[c].values for c in montos + ["x2_productos"]}
    V = _ventana(B, montos + tiene + ["x2_productos"]); cur = V[0]
    cols = montos + tiene + ["x2_productos"]; j = {c: i for i, c in enumerate(cols)}
    x = -np.arange(VENTANA, dtype=np.float32)
    with np.errstate(all="ignore"):
        ok = ~np.isnan(V); n = ok.sum(0)
        Vz = np.where(ok, V, 0.0); xs = np.where(ok, x[:, None, None], 0.0)
        mx = xs.sum(0) / np.maximum(n, 1); my = Vz.sum(0) / np.maximum(n, 1)
        cov = (np.where(ok, (x[:, None, None] - mx) * (V - my), 0.0)).sum(0); var = (np.where(ok, (x[:, None, None] - mx) ** 2, 0.0)).sum(0)
        tend = np.where((n >= 2) & (var > 0), cov / np.where(var > 0, var, 1), np.nan).astype(np.float32)
        vmax = np.nanmax(V, axis=0); vavg = np.nanmean(V, axis=0); vsum = np.nansum(V, axis=0)
    for c in ("x2_entradas", "x2_salidas", "x2_neto", "x2_ent_sueldo", "x2_sal_transf", "x2_sal_cajero", "x2_productos"):
        k = j[c]
        out[f"{c}_tend4"] = tend[:, k]
        out[f"{c}_sobre_max4"] = _div(cur[:, k], vmax[:, k])
        out[f"{c}_sobre_prom4"] = _div(cur[:, k], vavg[:, k])
    out["x2_neto_acum4"] = vsum[:, j["x2_neto"]].astype(np.float32)
    for c in tiene:       # lo tuvo en la ventana y hoy no
        k = j[c]; out[c.replace("x2_tiene_", "x2_perdio_")] = ((vmax[:, k] > 0) & (cur[:, k] == 0)).astype(np.float32)
    res = pd.DataFrame(out); res.insert(0, "foto_mes", crudo.foto_mes.values); res.insert(0, "numero_de_cliente", crudo.numero_de_cliente.values)
    return res


MONTOS_X2 = ["x2_ent_sueldo", "x2_ent_transf", "x2_ent_cheques", "x2_sal_transf", "x2_sal_cajero", "x2_sal_debitos", "x2_sal_servicios",
             "x2_sal_cheques", "x2_entradas", "x2_salidas", "x2_neto", "x2_neto_acum4"]
