
from __future__ import annotations
import math
from functools import lru_cache

# Simple ISA 1976 subset: returns T [K], p [Pa], rho [kg/m^3], a [m/s]
@lru_cache(maxsize=None)
def isa_atm(alt_ft: float):
    h = alt_ft * 0.3048
    T0 = 288.15
    p0 = 101325.0
    L = 0.0065
    R = 287.05287
    g = 9.80665
    if h <= 11000.0:
        T = T0 - L*h
        p = p0 * (T/T0)**(g/(R*L))
        rho = p/(R*T)
    else:
        T11 = T0 - L*11000.0
        p11 = p0 * (T11/T0)**(g/(R*L))
        p = p11 * math.exp(-g*(h-11000.0)/(R*T11))
        rho = p/(R*T11)
        T = T11
    a = math.sqrt(1.4*287.05287*T)
    return T, p, rho, a
