
# Minimal smoke test for engine_f110.py (no Streamlit needed)
from engine_f110 import F110Deck

if __name__ == "__main__":
    eng = F110Deck()
    for alt in (0, 10000, 20000):
        for mach in (0.0, 0.6):
            t_mil = eng.thrust_lbf(alt, mach, "MIL")
            t_max = eng.thrust_lbf(alt, mach, "MAX")
            ff = eng.fuel_flow_pph(alt, mach, "MIL")
            rpm = eng.rpm_from_ff(ff)
            print(f"ALT {alt:5.0f} M {mach:0.1f}  MIL {t_mil:8.1f} lbf  MAX {t_max:8.1f} lbf  FF~{ff:7.0f} pph  RPM~{rpm:4.0f}%")
