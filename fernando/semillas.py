"""Semillas elegidas.

Los cinco números primos en 100003 < p < 999983 que elegí y declaré en el
Check-In (Libro de la Asignatura, p. 33). Reemplazan al `semilla = 17` de los
notebooks de las clases. 

Cada script deriva de acá las semillas que necesite, como en z401:

    from semillas import SEMILLA

    np.random.seed(SEMILLA)
    semillas = np.random.choice(1000000, size=50, replace=False).tolist()
"""

SEMILLAS = (130399, 220391, 417577, 575753, 616261)

#: Semilla por defecto. Usar otra de SEMILLAS solo para repetir un
#: experimento variando unicamente la semilla.
SEMILLA = SEMILLAS[0]
