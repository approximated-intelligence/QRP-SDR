```python
# ==============================================================================
# UNIFIED QRP TRANSCEIVER OPTIMIZER (20m JLCPCB Base -> 40m 2x Add-On)
# Pure Functional Architecture | Structural Pattern Matching (Python 3.10+)
# ==============================================================================
import itertools
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import make_interp_spline
from scipy.optimize import (
    brentq,
    differential_evolution,
    minimize,
    milp,
    LinearConstraint,
    Bounds,
)
from scipy.signal import firwin, freqz
from scipy.special import erfinv

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ==============================================================================
# LAYER 0: SHARED SYSTEM CONSTANTS, E-SERIES & PURE FORMATTERS
# ==============================================================================
F20 = 14.10e6
F40 = 7.05e6
BANDS = (
    {"name": "20m (14.10 MHz)", "f0": F20, "scale": 1.0, "q_l": 60.0},
    {"name": "40m (7.05 MHz)", "f0": F40, "scale": 2.0, "q_l": 45.0},
)

E24 = np.array(
    [
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.7,
        3.0,
        3.3,
        3.6,
        3.9,
        4.3,
        4.7,
        5.1,
        5.6,
        6.2,
        6.8,
        7.5,
        8.2,
        9.1,
    ]
)
E12 = E24[::2]
E6 = E24[::4]
E3 = E24[::8]
E96 = np.round(10 ** (np.arange(96) / 96), 2)
E48 = E96[::2]
E192 = np.round(10 ** (np.arange(192) / 192), 2)

SERIES = {
    "E3": E3,
    "E6": E6,
    "E12": E12,
    "E24": E24,
    "E48": E48,
    "E96": E96,
    "E192": E192,
}


def expand_series(
    series_name: str,
    decades: tuple[float, ...],
    vmin: float = 0.0,
    vmax: float = np.inf,
) -> np.ndarray:
    arr = np.sort(np.concatenate([SERIES[series_name] * d for d in decades]))
    return arr[(arr >= vmin) & (arr <= vmax)]


PA_CAP_POOL = expand_series("E24", (10e-12, 100e-12), 47e-12, 680e-12)
PA_IND_POOL = expand_series("E12", (10e-9, 100e-9), 47e-9, 680e-9)

# Global Trimmer Mid-Travel Capacitances (set to 0.0 to disable trimmer on that node)
TRIM_PA_CEXT_MID = 18e-12  # e.g., 6-30 pF SMD trimmer at mid-travel on PA drain (Cext)
TRIM_LPF_C1_MID = (
    18e-12  # e.g., 6-30 pF SMD trimmer at mid-travel on TX filter input (C1)
)


def fmt_eng(v: float, unit: str = "") -> str:
    match v:
        case x if x >= 1e6:
            return f"{x / 1e6:.4g}M{unit}"
        case x if x >= 1e3:
            return f"{x / 1e3:.4g}k{unit}"
        case x if x >= 1.0:
            return f"{x:.4g}{unit}"
        case x if x >= 1e-3:
            return f"{x * 1e3:.4g}m{unit}"
        case x if x >= 1e-6:
            return f"{x * 1e6:.4g}u{unit}"
        case x if x >= 1e-9:
            return f"{x * 1e9:.4g}n{unit}"
        case _:
            return f"{v * 1e12:.4g}p{unit}"


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def section(title: str, width: int = 86) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# ==============================================================================
# LAYER 1: CLASS-E EPC2037 GaN PA — DATASHEET, ANALYTICAL & COSS MODELS
# ==============================================================================
PA_SPEC = {
    "VDD": 4.85,
    "POUT_TARGET": 1.65,
    "F0": 14.0e6,
    "VDS_MAX": 100.0,
    "ID_MAX": 1.7,
    "RON_25": 0.400,
    "RON_MAX_25": 0.550,
    "QG": 115e-12,
    "VG": 5.0,
    "COSS_50": 6.5e-12,
    "QOSS_50": 600e-12,
    "COSS_TR": 12e-12,
    "COSS_ER": 9.5e-12,
    "COSS_50_MAX": 10e-12,
    "QOSS_50_MAX": 900e-12,
    "QGS": 32e-12,
    "QGD": 25e-12,
    "RG": 0.5,
    "CISS": 14e-12,
    "CRSS": 0.1e-12,
}

DS_FIG8_TJ = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0])
DS_FIG8_RDS_NORM = np.array([0.85, 1.00, 1.15, 1.30, 1.48, 1.65, 1.80])
_RDS_CURVE = PchipInterpolator(DS_FIG8_TJ, DS_FIG8_RDS_NORM, extrapolate=False)

DS_FIG5_VDS = np.array(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
)
DS_FIG5_COSS = 1e-12 * np.array(
    [22.0, 15.5, 11.5, 9.8, 8.5, 7.5, 7.0, 6.7, 6.5, 5.8, 5.1, 4.5]
)
DS_FIG5_CRSS = 1e-12 * np.array(
    [2.0, 0.7, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.05]
)

# Extend flat above 100 V up to 150 V for PCHIP clamping
_V_EXT = np.append(DS_FIG5_VDS, 150.0)
_C_EXT = np.append(DS_FIG5_COSS, DS_FIG5_COSS[-1])

_COSS_TAB = PchipInterpolator(_V_EXT, _C_EXT, extrapolate=True)
_QOSS_TAB = _COSS_TAB.antiderivative()
_QOSS_DATAPOINTS = _QOSS_TAB(_V_EXT)  # exakt die 13 Datenblatt-Stützpunkte
_EOSS_TAB = PchipInterpolator(
    _V_EXT, _V_EXT * _C_EXT, extrapolate=True
).antiderivative()

Q_RAW_50 = float(_QOSS_TAB(50.0))


def temperature_factor(tj: float) -> float:
    return float(_RDS_CURVE(tj))


def coss_const(v: np.ndarray | float) -> np.ndarray:
    return np.full_like(np.asarray(v, float), PA_SPEC["COSS_TR"])


def qoss_const(v: np.ndarray | float) -> np.ndarray:
    return PA_SPEC["COSS_TR"] * np.abs(np.asarray(v, float))


def eoss_const(v: np.ndarray | float) -> np.ndarray:
    return 0.5 * PA_SPEC["COSS_TR"] * np.asarray(v, float) ** 2


def coss_tab(v: np.ndarray | float) -> np.ndarray:
    return _COSS_TAB(np.clip(np.abs(v), 0.0, 150.0))


def qoss_tab(v: np.ndarray | float) -> np.ndarray:
    return np.sign(v) * _QOSS_TAB(np.abs(v))


def eoss_tab(v: np.ndarray | float) -> np.ndarray:
    return _EOSS_TAB(np.abs(v))


def analytical_class_e(vdd: float, pout: float, f0: float, coss_tr: float) -> dict:
    w0 = 2.0 * np.pi * f0
    a = 2.0 / np.sqrt(np.pi**2 + 4.0)
    phi = np.pi - np.arctan(2.0 / np.pi)
    kr = 2.0 * a**2
    kc = 2.0 * a**2 / np.pi
    ropt = kr * vdd**2 / pout
    csh = kc / (w0 * ropt)

    ir_norm = np.sqrt(2.0 / kr)
    idc_norm = a * ir_norm
    c_norm = kc / kr
    th = np.linspace(0.0, 2.0 * np.pi, 200001)
    vd_norm = np.zeros_like(th)
    off = th >= np.pi
    vd_norm[off] = (
        idc_norm * (th[off] - np.pi)
        + ir_norm * (np.cos(th[off] + phi) - np.cos(np.pi + phi))
    ) / c_norm
    v1 = np.trapezoid(vd_norm * np.exp(-1j * th), th) / np.pi
    i1 = ir_norm * np.exp(1j * (phi - np.pi / 2.0))
    kl = (v1 / i1).imag / (v1 / i1).real

    l1 = kl * ropt / w0
    l2 = 3.0 * ropt / w0
    c2 = 1.0 / (w0**2 * l2)
    return {
        "a": a,
        "phi": phi,
        "KR": kr,
        "KC": kc,
        "KL": kl,
        "Ropt": ropt,
        "Csh": csh,
        "L1": l1,
        "L2": l2,
        "C2": c2,
        "LT": l1 + l2,
        "Cext_const": csh - coss_tr,
        "IDC_sk": pout / vdd,
        "IR_sk": np.sqrt(2.0 * pout / ropt),
    }


def make_device_model_spline(kind: str, c_ext: float, r_on: float) -> dict:
    match kind:
        case "const":
            coss_f, qoss_f, eoss_f = coss_const, qoss_const, eoss_const
        case "nonlin":
            coss_f, qoss_f, eoss_f = coss_tab, qoss_tab, eoss_tab
        case _:
            raise ValueError(f"Unknown device model kind: {kind}")

    q_knots = float(c_ext) * V_KNOTS + Q_OSS_KNOTS
    return {
        "kind": kind,
        "Cext": float(c_ext),
        "Ron": float(r_on),
        "coss_f": coss_f,
        "qoss_f": qoss_f,
        "eoss_f": eoss_f,
        "v_from_q_fn": PchipInterpolator(q_knots, V_KNOTS, extrapolate=True),
    }


def dev_v_from_q_spline(dev: dict, q: float) -> float:
    qf = float(q)
    if abs(qf) < 1e-30:
        return 0.0
    return float(np.sign(qf) * dev["v_from_q_fn"](abs(qf)))


def make_device_model_lin(kind: str, c_ext: float, r_on: float) -> dict:
    match kind:
        case "const":
            coss_f, qoss_f, eoss_f = coss_const, qoss_const, eoss_const
        case "nonlin":
            coss_f, qoss_f, eoss_f = coss_tab, qoss_tab, eoss_tab
        case _:
            raise ValueError(f"Unknown device model kind: {kind}")

    # Exact inverse cubic PCHIP spline evaluated once on a compact 2,001-point L1-cache grid
    q_knots = float(c_ext) * V_KNOTS + Q_OSS_KNOTS
    v_inv_spline = PchipInterpolator(q_knots, V_KNOTS, extrapolate=True)
    qg = np.linspace(0.0, float(q_knots[-1]), 2001)
    vg = v_inv_spline(qg)

    return {
        "kind": kind,
        "Cext": float(c_ext),
        "Ron": float(r_on),
        "coss_f": coss_f,
        "qoss_f": qoss_f,
        "eoss_f": eoss_f,
        "qg": qg,
        "vg": vg,
    }


def dev_v_from_q_lin(dev: dict, q: float) -> float:
    qf = float(q)
    if abs(qf) < 1e-30:
        return 0.0
    return float(np.sign(qf) * np.interp(abs(qf), dev["qg"], dev["vg"]))


def make_device_model(kind: str, c_ext: float, r_on: float) -> dict:
    match kind:
        case "const":
            coss_f, qoss_f, eoss_f = coss_const, qoss_const, eoss_const
        case "nonlin":
            coss_f, qoss_f, eoss_f = coss_tab, qoss_tab, eoss_tab
        case _:
            raise ValueError(f"Unknown device model kind: {kind}")

    q_knots = float(c_ext) * _V_EXT + _QOSS_DATAPOINTS
    pp = PchipInterpolator(q_knots, _V_EXT, extrapolate=True)
    return {
        "kind": kind,
        "Cext": float(c_ext),
        "Ron": float(r_on),
        "coss_f": coss_f,
        "qoss_f": qoss_f,
        "eoss_f": eoss_f,
        "q_knots": q_knots,
        "pp_c": pp.c,
    }


def dev_v_from_q(dev: dict, q: float) -> float:
    qf = float(q)
    aq = abs(qf)
    if aq < 1e-30:
        return 0.0
    qk = dev["q_knots"]
    c = dev["pp_c"]
    n_int = len(qk) - 1
    idx = int(np.searchsorted(qk, aq, side="right")) - 1
    if idx < 0:
        idx = 0
    elif idx >= n_int:
        idx = n_int - 1
    dq = aq - float(qk[idx])
    # Horner's method for C1-smooth cubic polynomial on interval idx
    v = (
        (float(c[0, idx]) * dq + float(c[1, idx])) * dq + float(c[2, idx])
    ) * dq + float(c[3, idx])
    return v if qf >= 0.0 else -v


def dev_e_total(dev: dict, v: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(v, float)
    return 0.5 * dev["Cext"] * arr**2 + np.asarray(dev["eoss_f"](arr), float)


def simulate_cycle(
    dev: dict,
    i_dc: float,
    lt_val: float,
    c2_val: float,
    ropt_val: float,
    f0: float,
    warmup: int = 100,
    samples: int = 400,
) -> dict:
    t_per = 1.0 / f0
    th = 0.5 * t_per
    match dev:
        case {"kind": "const", "Cext": c_ext, "Ron": r_on}:
            csh = c_ext + PA_SPEC["COSS_TR"]

            def f_on(_, y):
                i_l, v_c2 = y
                return [
                    (r_on * (i_dc - i_l) - v_c2 - ropt_val * i_l) / lt_val,
                    i_l / c2_val,
                ]

            def f_off(_, y):
                v_d, i_l, v_c2 = y
                return [
                    (i_dc - i_l) / csh,
                    (v_d - v_c2 - ropt_val * i_l) / lt_val,
                    i_l / c2_val,
                ]

            i_l, v_c2 = 0.0, 0.0
            for _ in range(warmup):
                s1 = solve_ivp(
                    f_on,
                    [0.0, th],
                    [i_l, v_c2],
                    method="LSODA",
                    rtol=1e-10,
                    atol=1e-13,
                    max_step=th / 20.0,
                )
                i_l, v_c2 = float(s1.y[0, -1]), float(s1.y[1, -1])
                s2 = solve_ivp(
                    f_off,
                    [0.0, th],
                    [r_on * (i_dc - i_l), i_l, v_c2],
                    method="LSODA",
                    rtol=1e-10,
                    atol=1e-13,
                    max_step=th / 20.0,
                )
                i_l, v_c2 = float(s2.y[1, -1]), float(s2.y[2, -1])

            s1 = solve_ivp(
                f_on,
                [0.0, th],
                [i_l, v_c2],
                method="LSODA",
                rtol=1e-10,
                atol=1e-13,
                dense_output=True,
                max_step=th / 20.0,
            )
            i_l_on, v_c2_on = float(s1.y[0, -1]), float(s1.y[1, -1])
            s2 = solve_ivp(
                f_off,
                [0.0, th],
                [r_on * (i_dc - i_l_on), i_l_on, v_c2_on],
                method="LSODA",
                rtol=1e-10,
                atol=1e-13,
                dense_output=True,
                max_step=th / 20.0,
            )

            t_on = np.linspace(0.0, th, samples + 1)
            t_off = np.linspace(0.0, th, samples + 1)
            y_on, y_off = s1.sol(t_on), s2.sol(t_off)
            t = np.concatenate([t_on, th + t_off[1:]])
            v_d = np.concatenate(
                [
                    r_on * (i_dc - y_on[0]) if r_on > 0 else np.zeros_like(t_on),
                    y_off[0][1:],
                ]
            )
            i_l_f = np.concatenate([y_on[0], y_off[1][1:]])
            i_sw = np.concatenate([i_dc - y_on[0], np.zeros_like(y_off[1][1:])])
            return {
                "t": t,
                "v_D": v_d,
                "i_L": i_l_f,
                "i_sw": i_sw,
                "q": csh * v_d,
                "Coss": np.full_like(v_d, PA_SPEC["COSS_TR"]),
                "v_avg": float(np.trapezoid(v_d, t) / t_per),
                "v_peak": float(np.max(v_d)),
                "v_min": float(np.min(v_d)),
                "v_turnon": float(s2.sol(th)[0]),
                "i_turnon": float(s2.sol(th)[1]),
            }

        case {"kind": "nonlin", "Ron": r_on}:

            def f_on(_, y):
                q, i_l, v_2 = y
                v_d = dev_v_from_q(dev, q)
                return [
                    i_dc - i_l - v_d / r_on,
                    (v_d - v_2 - ropt_val * i_l) / lt_val,
                    i_l / c2_val,
                ]

            def f_off(_, y):
                q, i_l, v_2 = y
                v_d = dev_v_from_q(dev, q)
                return [i_dc - i_l, (v_d - v_2 - ropt_val * i_l) / lt_val, i_l / c2_val]

            y = np.zeros(3)
            for _ in range(warmup):
                s1 = solve_ivp(
                    f_on,
                    (0.0, th),
                    y,
                    method="LSODA",
                    rtol=2e-8,
                    atol=1e-12,
                    max_step=th / 20.0,
                )
                s2 = solve_ivp(
                    f_off,
                    (0.0, th),
                    s1.y[:, -1],
                    method="LSODA",
                    rtol=2e-8,
                    atol=1e-12,
                    max_step=th / 20.0,
                )
                y = s2.y[:, -1]

            s1 = solve_ivp(
                f_on,
                (0.0, th),
                y,
                method="LSODA",
                rtol=2e-8,
                atol=1e-12,
                dense_output=True,
                max_step=th / 20.0,
            )
            s2 = solve_ivp(
                f_off,
                (0.0, th),
                s1.y[:, -1],
                method="LSODA",
                rtol=2e-8,
                atol=1e-12,
                dense_output=True,
                max_step=th / 20.0,
            )
            t1 = np.linspace(0.0, th, samples, endpoint=False)
            t2 = np.linspace(0.0, th, samples, endpoint=False)
            y1, y2 = s1.sol(t1), s2.sol(t2)
            q = np.r_[y1[0], y2[0]]
            i_l = np.r_[y1[1], y2[1]]
            t = np.r_[t1, th + t2]
            v_d = np.array([dev_v_from_q(dev, z) for z in q])
            i_sw = np.r_[v_d[:samples] / r_on, np.zeros(samples)]
            coss_wave = np.array([dev["coss_f"](v) for v in v_d])
            return {
                "t": t,
                "q": q,
                "v_D": v_d,
                "i_L": i_l,
                "i_sw": i_sw,
                "Coss": coss_wave,
                "v_avg": float(np.trapezoid(v_d, t) / t_per),
                "v_peak": float(np.max(v_d)),
                "v_min": float(np.min(v_d)),
                "v_turnon": float(dev_v_from_q(dev, s2.y[0, -1])),
                "i_turnon": float(s2.y[1, -1]),
            }

        case _:
            raise ValueError(f"Unsupported device model: {dev}")


def solve_dc_current(
    dev: dict,
    lt_val: float,
    c2_val: float,
    ropt_val: float,
    vdd: float,
    f0: float,
    idc_est: float,
    warmup: int = 50,
    samples: int = 200,
) -> tuple[dict, float]:
    match dev["kind"]:
        case "const":
            lo, hi = 0.7 * idc_est, 1.3 * idc_est
            w_solve, s_solve = 200, 500
            w_final, s_final = 500, 1000
        case "nonlin":
            lo, hi = 0.05 * idc_est, 3.0 * idc_est
            w_solve, s_solve = warmup, samples
            w_final, s_final = warmup * 2, samples * 2
        case _:
            raise ValueError(f"Unknown device kind: {dev['kind']}")

    def resid(i_dc: float) -> float:
        return (
            simulate_cycle(
                dev, i_dc, lt_val, c2_val, ropt_val, f0, warmup=w_solve, samples=s_solve
            )["v_avg"]
            - vdd
        )

    i_sol = float(brentq(resid, lo, hi, xtol=1e-8))
    return (
        simulate_cycle(
            dev, i_sol, lt_val, c2_val, ropt_val, f0, warmup=w_final, samples=s_final
        ),
        i_sol,
    )


def run_pa_point(
    kind: str,
    r_on: float,
    l1_val: float,
    l2_val: float,
    c2_val: float,
    ropt_val: float,
    c_ext: float,
    vdd: float,
    f0: float,
    pgate: float,
    label: str = "",
) -> dict:
    t_per = 1.0 / f0
    idc_est = PA_SPEC["POUT_TARGET"] / vdd
    dev = make_device_model(kind, c_ext, r_on)
    h, i_dc = solve_dc_current(dev, l1_val + l2_val, c2_val, ropt_val, vdd, f0, idc_est)

    p_load = float(np.trapezoid(ropt_val * h["i_L"] ** 2, h["t"]) / t_per)
    p_ron = float(r_on * np.trapezoid(h["i_sw"] ** 2, h["t"]) / t_per)
    e_cap = dev_e_total(dev, h["v_D"])
    p_zvs = (
        float(0.5 * (c_ext + PA_SPEC["COSS_TR"]) * h["v_turnon"] ** 2 * f0)
        if kind == "const"
        else 0.0
    )

    return {
        "h": h,
        "I": i_dc,
        "P_load": p_load,
        "P_Ron": p_ron,
        "P_ZVS": p_zvs,
        "E_turn": float(dev_e_total(dev, h["v_turnon"])),
        "E_cap_avg": float(np.trapezoid(e_cap, h["t"]) / t_per),
        "E_cap_delta": float(e_cap[-1] - e_cap[0]),
        "eta_system": float(p_load / (vdd * i_dc + pgate)),
        "dev": dev,
        "L1_used": l1_val,
        "L2_used": l2_val,
        "C2_used": c2_val,
        "Ropt_used": ropt_val,
        "Cext_used": c_ext,
        "label": label,
    }


def extract_pa_harmonics_dbc(pa_res: dict, n_harmonics: int = 5) -> tuple[float, ...]:
    """Extracts 2f0..Nf0 harmonic power in dBc relative to fundamental f0 via FFT of load current."""
    i_l = pa_res["h"]["i_L"]
    n_pts = len(i_l)
    # Periodic waveform over exact 1 cycle (drop duplicate endpoint if present)
    sig = i_l[:-1] if np.isclose(i_l[0], i_l[-1], atol=1e-9) else i_l
    spec = np.abs(np.fft.rfft(sig)) / len(sig)
    p_fund = spec[1] ** 2
    return tuple(
        float(10.0 * np.log10((spec[k] ** 2) / (p_fund + 1e-30) + 1e-18))
        for k in range(2, n_harmonics + 1)
    )


def build_pa_cap_candidates(
    center: float, max_par: int = 1, window: float = 0.25
) -> list[tuple[float, str]]:
    lo, hi = center * (1.0 - window), center * (1.0 + window)
    cands: list[tuple[float, str]] = [
        (float(c), fmt_eng(c, "F")) for c in PA_CAP_POOL if lo <= c <= hi
    ]
    if max_par >= 2:
        for i, c1 in enumerate(PA_CAP_POOL):
            for c2 in PA_CAP_POOL[i:]:
                csum = float(c1 + c2)
                if lo <= csum <= hi:
                    cands.append((csum, f"{fmt_eng(c1, 'F')}+{fmt_eng(c2, 'F')}"))
    return cands


def optimize_class_e_joint_discrete(
    mode: str,
    ana: dict,
    vdd: float,
    pout_target: float,
    f20: float = F20,
    f40: float = F40,
    r_on: float = PA_SPEC["RON_25"],
    pgate_20: float = 0.0,
    pgate_40: float = 0.0,
    max_cap_par: int = 1,
    share_ind_reel: bool = False,
) -> dict:
    """
    1) Finds continuous joint 20m (1x) + 40m (2x) ZVS + ZDS (Von=0, dV/dt=0) optimum via SLSQP
       starting directly from the analytical Sokal design.
    2) Linearizes Von20, Von40, dVdt20, dVdt40, P20, P40 around x* via 8 finite-diff LSODA evals.
    3) Solves an exact MILP (scipy.optimize.milp) over E24 caps & E12 inductors in <50ms.
    """
    idc_est = pout_target / vdd
    cache: dict[tuple[float, ...], dict] = {}

    def eval_both(x: np.ndarray, warmup: int = 15, samples: int = 80) -> dict:
        key = tuple(np.round(x, 5)) + (warmup, samples)
        if key in cache:
            return cache[key]
        match mode:
            case "T1":
                c_ext, l1_v, l2_v, c2_v, ropt_v = (
                    float(x.item(0)) * 1e-12,
                    float(x.item(1)) * 1e-9,
                    float(x.item(2)) * 1e-9,
                    float(x.item(3)) * 1e-12,
                    float(x.item(4)),
                )
            case "T2":
                c_ext, l1_v, l2_v, c2_v = (
                    float(x.item(0)) * 1e-12,
                    float(x.item(1)) * 1e-9,
                    float(x.item(2)) * 1e-9,
                    float(x.item(3)) * 1e-12,
                )
                ropt_v = float(ana["Ropt"])
            case _:
                raise ValueError(f"Unknown Class-E mode: {mode}")

        try:
            dev20 = make_device_model("nonlin", c_ext, r_on)
            h20, idc20 = solve_dc_current(
                dev20,
                l1_v + l2_v,
                c2_v,
                ropt_v,
                vdd,
                f20,
                idc_est,
                warmup=warmup,
                samples=samples,
            )
            p20 = float(np.trapezoid(ropt_v * h20["i_L"] ** 2, h20["t"]) * f20)
            eta20 = float(p20 / (vdd * idc20 + pgate_20))
            dvdt20 = float(
                (idc20 - h20["i_turnon"])
                / (c_ext + float(dev20["coss_f"](h20["v_turnon"])))
            )

            dev40 = make_device_model("nonlin", 2.0 * c_ext, r_on)
            h40, idc40 = solve_dc_current(
                dev40,
                2.0 * (l1_v + l2_v),
                2.0 * c2_v,
                ropt_v,
                vdd,
                f40,
                idc_est,
                warmup=warmup,
                samples=samples,
            )
            p40 = float(np.trapezoid(ropt_v * h40["i_L"] ** 2, h40["t"]) * f40)
            eta40 = float(p40 / (vdd * idc40 + pgate_40))
            dvdt40 = float(
                (idc40 - h40["i_turnon"])
                / (2.0 * c_ext + float(dev40["coss_f"](h40["v_turnon"])))
            )

            out = {
                "von20": float(h20["v_turnon"]),
                "dvdt20": dvdt20,
                "p20": p20,
                "eta20": eta20,
                "von40": float(h40["v_turnon"]),
                "dvdt40": dvdt40,
                "p40": p40,
                "eta40": eta40,
            }
        except Exception:
            out = {
                "von20": 10.0,
                "dvdt20": 1e9,
                "p20": 0.0,
                "eta20": 0.0,
                "von40": 10.0,
                "dvdt40": 1e9,
                "p40": 0.0,
                "eta40": 0.0,
            }
        cache[key] = out
        return out

    # Start directly from analytical Sokal design (Csh_ideal - Coss_50, L1_ana, L2_ana, C2_ana)
    cext_init = (ana["Csh"] - PA_SPEC["COSS_50"]) * 1e12
    l1_init = ana["L1"] * 1e9
    l2_init = ana["L2"] * 1e9
    c2_init = ana["C2"] * 1e12

    match mode:
        case "T1":
            x0 = np.array([cext_init, l1_init, l2_init, c2_init, ana["Ropt"]])
            bnds = [
                (100.0, 600.0),
                (50.0, 250.0),
                (150.0, 450.0),
                (200.0, 900.0),
                (3.0, 20.0),
            ]
            cons = [
                {"type": "eq", "fun": lambda x: eval_both(x)["von20"]},
                {"type": "eq", "fun": lambda x: eval_both(x)["von40"]},
                {"type": "eq", "fun": lambda x: eval_both(x)["dvdt20"] * 1e-8},
                {"type": "eq", "fun": lambda x: eval_both(x)["dvdt40"] * 1e-8},
                {
                    "type": "eq",
                    "fun": lambda x: 0.5 * (eval_both(x)["p20"] + eval_both(x)["p40"])
                    - pout_target,
                },
            ]
        case "T2":
            x0 = np.array([cext_init, l1_init, l2_init, c2_init])
            bnds = [(100.0, 600.0), (50.0, 250.0), (150.0, 450.0), (200.0, 900.0)]
            cons = [
                {"type": "eq", "fun": lambda x: eval_both(x)["von20"]},
                {"type": "eq", "fun": lambda x: eval_both(x)["von40"]},
                {"type": "eq", "fun": lambda x: eval_both(x)["dvdt20"] * 1e-8},
                {"type": "eq", "fun": lambda x: eval_both(x)["dvdt40"] * 1e-8},
            ]
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    t0 = time.time()
    sol_nlp = minimize(
        lambda x: -0.5 * (eval_both(x)["eta20"] + eval_both(x)["eta40"]),
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 60, "ftol": 1e-4, "eps": 1e-3},
    )

    x_star_raw = sol_nlp.x
    ropt_v = float(x_star_raw.item(4)) if mode == "T1" else float(ana["Ropt"])
    x4_star = np.array(
        [
            float(x_star_raw.item(0)),
            float(x_star_raw.item(1)),
            float(x_star_raw.item(2)),
            float(x_star_raw.item(3)),
        ]
    )
    u_star = np.log(x4_star)

    # Evaluate baseline & 8 finite-difference points around continuous ZVS+ZDS solution
    base = eval_both(x_star_raw, warmup=40, samples=200)
    eps = 1e-2
    j_von20, j_von40 = np.zeros(4), np.zeros(4)
    j_dv20, j_dv40 = np.zeros(4), np.zeros(4)
    j_p20, j_p40 = np.zeros(4), np.zeros(4)
    for i in range(4):
        du = np.zeros(4)
        du[i] = eps
        xp4, xn4 = np.exp(u_star + du), np.exp(u_star - du)
        xp = np.append(xp4, ropt_v) if mode == "T1" else xp4
        xn = np.append(xn4, ropt_v) if mode == "T1" else xn4
        rp = eval_both(xp, warmup=40, samples=200)
        rn = eval_both(xn, warmup=40, samples=200)
        j_von20[i] = (rp["von20"] - rn["von20"]) / (2.0 * eps)
        j_von40[i] = (rp["von40"] - rn["von40"]) / (2.0 * eps)
        j_dv20[i] = (rp["dvdt20"] - rn["dvdt20"]) * 1e-8 / (2.0 * eps)
        j_dv40[i] = (rp["dvdt40"] - rn["dvdt40"]) * 1e-8 / (2.0 * eps)
        j_p20[i] = (rp["p20"] - rn["p20"]) / (2.0 * eps)
        j_p40[i] = (rp["p40"] - rn["p40"]) / (2.0 * eps)

    # Build discrete E24/E12 pools: [Cext_total (pF), L1 (nH), L2 (nH), C2 (pF)]
    # Cext_total = C_base (E24) + TRIM_PA_CEXT_MID so trimmer sits at exact mid-travel
    cext_base_target = max(47e-12, float(x4_star.item(0)) * 1e-12 - TRIM_PA_CEXT_MID)
    cext_raw = build_pa_cap_candidates(
        cext_base_target, max_par=max_cap_par, window=0.35
    )
    cext_pool = [
        (
            v + TRIM_PA_CEXT_MID,
            (
                f"{lbl}+trim({fmt_eng(TRIM_PA_CEXT_MID, 'F')})"
                if TRIM_PA_CEXT_MID > 0
                else lbl
            ),
        )
        for v, lbl in cext_raw
    ]
    c2_pool = build_pa_cap_candidates(
        float(x4_star.item(3)) * 1e-12, max_par=max_cap_par, window=0.35
    )
    l1_pool = [
        (float(l), fmt_eng(l, "H"))
        for l in PA_IND_POOL
        if 0.60 * x4_star.item(1) * 1e-9 <= l <= 1.45 * x4_star.item(1) * 1e-9
    ]
    l2_pool = [
        (float(l), fmt_eng(l, "H"))
        for l in PA_IND_POOL
        if 0.60 * x4_star.item(2) * 1e-9 <= l <= 1.45 * x4_star.item(2) * 1e-9
    ]

    pools = [
        [(v * 1e12, lbl) for v, lbl in cext_pool],
        [(v * 1e9, lbl) for v, lbl in l1_pool],
        [(v * 1e9, lbl) for v, lbl in l2_pool],
        [(v * 1e12, lbl) for v, lbl in c2_pool],
    ]

    offsets, cur = [], 0
    for p in pools:
        offsets.append(cur)
        cur += len(p)

    # Slacks: |Von20|, |Von40|, |dVdt20*1e-8|, |dVdt40*1e-8|, |P_avg - P_target|
    idx_sv20, idx_sv40, idx_sdv20, idx_sdv40, idx_sp = (
        cur,
        cur + 1,
        cur + 2,
        cur + 3,
        cur + 4,
    )
    n_vars = cur + 5

    c_obj = np.zeros(n_vars)
    for i, p in enumerate(pools):
        vals_i = np.array([val for val, _ in p], dtype=float)
        par_pen = np.array([0.05 if "+" in lbl else 0.0 for _, lbl in p], dtype=float)
        c_obj[offsets[i] : offsets[i] + len(p)] = (
            np.abs(np.log(vals_i / x4_star.item(i))) + par_pen
        )
    c_obj[idx_sv20] = 30.0
    c_obj[idx_sv40] = 30.0
    c_obj[idx_sdv20] = 12.0
    c_obj[idx_sdv40] = 12.0
    c_obj[idx_sp] = 3.0 if mode == "T1" else 0.0

    rows, lb, ub = [], [], []
    for i, p in enumerate(pools):
        r = np.zeros(n_vars)
        r[offsets[i] : offsets[i] + len(p)] = 1.0
        rows.append(r)
        lb.append(1.0)
        ub.append(1.0)

    # Optional 1-reel constraint (L1 == L2)
    if share_ind_reel:
        for k1, (v1, _) in enumerate(
            pools.item(1) if isinstance(pools, np.ndarray) else pools[1]
        ):
            for k2, (v2, _) in enumerate(
                pools.item(2) if isinstance(pools, np.ndarray) else pools[2]
            ):
                if not np.isclose(v1, v2, rtol=1e-9):
                    r = np.zeros(n_vars)
                    r[offsets[1] + k1] = 1.0
                    r[offsets[2] + k2] = 1.0
                    rows.append(r)
                    lb.append(-np.inf)
                    ub.append(1.0)

    # Linearized ZVS (Von = 0) & ZDS (dV/dt = 0) constraints for 20m and 40m
    zvs_zds_terms = [
        (j_von20, base["von20"], idx_sv20),
        (j_von40, base["von40"], idx_sv40),
        (j_dv20, base["dvdt20"] * 1e-8, idx_sdv20),
        (j_dv40, base["dvdt40"] * 1e-8, idx_sdv40),
    ]
    for j_vec, ref_val, idx_s in zvs_zds_terms:
        r_pos, r_neg = np.zeros(n_vars), np.zeros(n_vars)
        for i, p in enumerate(pools):
            du_vec = np.log(np.array([val for val, _ in p], dtype=float)) - u_star.item(
                i
            )
            r_pos[offsets[i] : offsets[i] + len(p)] = j_vec[i] * du_vec
            r_neg[offsets[i] : offsets[i] + len(p)] = -j_vec[i] * du_vec
        r_pos[idx_s] = -1.0
        r_neg[idx_s] = -1.0
        rows.append(r_pos)
        lb.append(-np.inf)
        ub.append(-ref_val)
        rows.append(r_neg)
        lb.append(-np.inf)
        ub.append(ref_val)

    if mode == "T1":
        p_base = 0.5 * (base["p20"] + base["p40"])
        j_p_avg = 0.5 * (j_p20 + j_p40)
        r_p1, r_p2 = np.zeros(n_vars), np.zeros(n_vars)
        for i, p in enumerate(pools):
            du_vec = np.log(np.array([val for val, _ in p], dtype=float)) - u_star.item(
                i
            )
            r_p1[offsets[i] : offsets[i] + len(p)] = j_p_avg[i] * du_vec
            r_p2[offsets[i] : offsets[i] + len(p)] = -j_p_avg[i] * du_vec
        r_p1[idx_sp] = -1.0
        r_p2[idx_sp] = -1.0
        rows.append(r_p1)
        lb.append(-np.inf)
        ub.append(pout_target - p_base)
        rows.append(r_p2)
        lb.append(-np.inf)
        ub.append(p_base - pout_target)

    integrality = np.ones(n_vars)
    integrality[idx_sv20:] = 0
    ub_vars = np.ones(n_vars)
    ub_vars[idx_sv20:] = np.inf

    sol_mip = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), ub_vars),
        constraints=LinearConstraint(np.array(rows), lb, ub),
    )
    assert sol_mip.success, f"PA MILP failed for {mode}: {sol_mip.message}"

    picks = [
        pools[i][int(np.argmax(sol_mip.x[offsets[i] : offsets[i] + len(pools[i])]))]
        for i in range(4)
    ]
    c_v, c_lbl = picks[0][0] * 1e-12, picks[0][1]
    l1_v, l1_lbl = picks[1][0] * 1e-9, picks[1][1]
    l2_v, l2_lbl = picks[2][0] * 1e-9, picks[2][1]
    c2_v, c2_lbl = picks[3][0] * 1e-12, picks[3][1]

    print(
        f"  {mode} MILP Joint ZVS+ZDS (20m->40m) in {time.time() - t0:.1f}s | "
        f"20m BOM: Cext={c_lbl}, L1={l1_lbl}, L2={l2_lbl}, C2={c2_lbl}, Ropt={ropt_v:.2f}Ω"
    )

    bom_20 = {"Cext": c_lbl, "L1": l1_lbl, "L2": l2_lbl, "C2": c2_lbl}
    res_20 = run_pa_point(
        "nonlin",
        r_on,
        l1_v,
        l2_v,
        c2_v,
        ropt_v,
        c_v,
        vdd,
        f20,
        pgate_20,
        f"{mode} ZVS+ZDS [20m 1x]",
    )
    res_40 = run_pa_point(
        "nonlin",
        r_on,
        2.0 * l1_v,
        2.0 * l2_v,
        2.0 * c2_v,
        ropt_v,
        2.0 * c_v,
        vdd,
        f40,
        pgate_40,
        f"{mode} ZVS+ZDS [40m 2x]",
    )
    return {"20m": res_20, "40m": res_40, "bom_20": bom_20, "ropt": ropt_v}


def optimize_class_e_zvs_joint(
    mode: str,
    ana: dict,
    vdd: float,
    pout_target: float,
    f20: float = F20,
    f40: float = F40,
    r_on: float = PA_SPEC["RON_25"],
    pgate_20: float = 0.0,
    pgate_40: float = 0.0,
) -> dict:
    """
    Co-optimizes 20m base values (Cext, L1, L2, C2, [Ropt]) such that:
      - 20m (14.10 MHz) operates with 1x (Cext, L1, L2, C2)
      - 40m ( 7.05 MHz) operates with 2x (Cext, L1, L2, C2) + fixed nonlinear Coss(Vds)
    """
    idc_est = pout_target / vdd
    cache: dict[tuple[float, ...], dict] = {}

    def eval_both_bands(x: np.ndarray) -> dict:
        key = tuple(np.round(x, 5))
        if key in cache:
            return cache[key]

        match mode:
            case "T1":
                c_ext, l1_v, l2_v, c2_v, ropt_v = (
                    float(x.item(0)) * 1e-12,
                    float(x.item(1)) * 1e-9,
                    float(x.item(2)) * 1e-9,
                    float(x.item(3)) * 1e-12,
                    float(x.item(4)),
                )
            case "T2":
                c_ext, l1_v, l2_v, c2_v = (
                    float(x.item(0)) * 1e-12,
                    float(x.item(1)) * 1e-9,
                    float(x.item(2)) * 1e-9,
                    float(x.item(3)) * 1e-12,
                )
                ropt_v = float(ana["Ropt"])
            case _:
                raise ValueError(f"Unknown Class-E mode: {mode}")

        try:
            # 20m base build (1x passives)
            dev20 = make_device_model("nonlin", c_ext, r_on)
            h20, idc20 = solve_dc_current(
                dev20,
                l1_v + l2_v,
                c2_v,
                ropt_v,
                vdd,
                f20,
                idc_est,
                warmup=20,
                samples=100,
            )
            p20 = float(np.trapezoid(ropt_v * h20["i_L"] ** 2, h20["t"]) * f20)
            eta20 = float(p20 / (vdd * idc20 + pgate_20))
            csh20 = c_ext + float(dev20["coss_f"](h20["v_turnon"]))
            dvdt20 = float((idc20 - h20["i_turnon"]) / csh20)

            # 40m 2x add-on build (2x passives, same nonlinear EPC2037 Coss)
            dev40 = make_device_model("nonlin", 2.0 * c_ext, r_on)
            h40, idc40 = solve_dc_current(
                dev40,
                2.0 * (l1_v + l2_v),
                2.0 * c2_v,
                ropt_v,
                vdd,
                f40,
                idc_est,
                warmup=20,
                samples=100,
            )
            p40 = float(np.trapezoid(ropt_v * h40["i_L"] ** 2, h40["t"]) * f40)
            eta40 = float(p40 / (vdd * idc40 + pgate_40))
            csh40 = 2.0 * c_ext + float(dev40["coss_f"](h40["v_turnon"]))
            dvdt40 = float((idc40 - h40["i_turnon"]) / csh40)

            out = {
                "ok": True,
                "von20": float(h20["v_turnon"]),
                "dvdt20": dvdt20,
                "p20": p20,
                "eta20": eta20,
                "von40": float(h40["v_turnon"]),
                "dvdt40": dvdt40,
                "p40": p40,
                "eta40": eta40,
            }
        except Exception:
            out = {
                "ok": False,
                "von20": 10.0,
                "dvdt20": 1e9,
                "p20": 0.0,
                "eta20": 0.0,
                "von40": 10.0,
                "dvdt40": 1e9,
                "p40": 0.0,
                "eta40": 0.0,
            }

        cache[key] = out
        return out

    match mode:
        case "T1":
            x0 = np.array([345.0, 106.0, 279.0, 446.0, 6.29])
            bnds = [
                (100.0, 600.0),
                (50.0, 250.0),
                (150.0, 450.0),
                (200.0, 900.0),
                (3.0, 20.0),
            ]
            cons = [
                {"type": "eq", "fun": lambda x: eval_both_bands(x)["von20"]},
                {"type": "eq", "fun": lambda x: eval_both_bands(x)["von40"]},
                {
                    "type": "eq",
                    "fun": lambda x: 0.5
                    * (eval_both_bands(x)["p20"] + eval_both_bands(x)["p40"])
                    - pout_target,
                },
            ]
            label = f"T1 Joint 20m/40m: Ropt free, P_avg={pout_target:.2f}W"
        case "T2":
            x0 = np.array([270.0, 115.0, 288.0, 469.0])
            bnds = [(100.0, 600.0), (50.0, 250.0), (150.0, 450.0), (200.0, 900.0)]
            cons = [
                {"type": "eq", "fun": lambda x: eval_both_bands(x)["von20"]},
                {"type": "eq", "fun": lambda x: eval_both_bands(x)["von40"]},
            ]
            label = "T2 Joint 20m/40m: Ropt locked (8.22Ω)"
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    def objective(x: np.ndarray) -> float:
        r = eval_both_bands(x)
        # Maximize average efficiency while softly penalizing residual dV/dt at turn-on
        dvdt_pen = 1e-18 * (r["dvdt20"] ** 2 + r["dvdt40"] ** 2)
        return -0.5 * (r["eta20"] + r["eta40"]) + dvdt_pen

    t_start = time.time()
    sol = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 120, "eps": 1e-4},
    )
    print(
        f"  {mode} (Joint 20m+40m) finished in {time.time() - t_start:.1f}s ({len(cache)} evals) -> {sol.message}"
    )

    c_ext = float(sol.x.item(0)) * 1e-12
    l1_v = float(sol.x.item(1)) * 1e-9
    l2_v = float(sol.x.item(2)) * 1e-9
    c2_v = float(sol.x.item(3)) * 1e-12
    ropt_v = float(sol.x.item(4)) if mode == "T1" else float(ana["Ropt"])

    res_20 = run_pa_point(
        "nonlin",
        r_on,
        l1_v,
        l2_v,
        c2_v,
        ropt_v,
        c_ext,
        vdd,
        f20,
        pgate_20,
        f"{label} [20m 1x]",
    )
    res_40 = run_pa_point(
        "nonlin",
        r_on,
        2.0 * l1_v,
        2.0 * l2_v,
        2.0 * c2_v,
        ropt_v,
        2.0 * c_ext,
        vdd,
        f40,
        pgate_40,
        f"{label} [40m 2x]",
    )
    return {"20m": res_20, "40m": res_40}


def optimize_class_e_zvs(
    mode: str,
    ana: dict,
    vdd: float,
    pout_target: float,
    f0: float,
    r_on: float,
    pgate: float,
) -> dict:
    """Pure functional SLSQP ZVS optimizer for EPC2037 Class-E PA (no global state)."""
    t_per = 1.0 / f0
    idc_est = pout_target / vdd
    cache: dict[tuple[float, ...], tuple[float, float, float, float, float]] = {}

    def eval_x(x: np.ndarray) -> tuple[float, float, float, float, float]:
        key = tuple(np.round(x, 5))
        if key in cache:
            return cache[key]
        match mode:
            case "T1":
                c_ext, l1_v, l2_v, c2_v, ropt_v = (
                    x[0] * 1e-12,
                    x[1] * 1e-9,
                    x[2] * 1e-9,
                    x[3] * 1e-12,
                    x[4],
                )
            case "T2":
                c_ext, l1_v, l2_v, c2_v = (
                    x[0] * 1e-12,
                    x[1] * 1e-9,
                    x[2] * 1e-9,
                    x[3] * 1e-12,
                )
                ropt_v = ana["Ropt"]
            case _:
                raise ValueError(f"Unknown Class-E optimization mode: {mode}")

        dev = make_device_model("nonlin", c_ext, r_on)
        try:
            h, i_dc = solve_dc_current(
                dev, l1_v + l2_v, c2_v, ropt_v, vdd, f0, idc_est, warmup=20, samples=100
            )
            p_load = float(np.trapezoid(ropt_v * h["i_L"] ** 2, h["t"]) / t_per)
            eta = float(p_load / (vdd * i_dc + pgate))
            csh_tot = c_ext + float(dev["coss_f"](h["v_turnon"]))
            dv_dt = float((i_dc - h["i_turnon"]) / csh_tot)
            out = (float(h["v_turnon"]), dv_dt, p_load, eta, float(np.max(h["i_sw"])))
        except Exception:
            out = (10.0, 10.0, 0.0, 0.0, 10.0)
        cache[key] = out
        return out

    match mode:
        case "T1":
            x0 = np.array([247.33, 107.73, 280.44, 460.83, ana["Ropt"]])
            bnds = [
                (100.0, 500.0),
                (50.0, 250.0),
                (150.0, 450.0),
                (200.0, 900.0),
                (3.0, 20.0),
            ]
            cons = [
                {"type": "eq", "fun": lambda x: eval_x(x)[0]},
                {"type": "eq", "fun": lambda x: eval_x(x)[1] * 1e-8},
                {"type": "eq", "fun": lambda x: eval_x(x)[2] - pout_target},
            ]
            label = f"T1: Ropt free, P={pout_target:.2f}W"
        case "T2":
            x0 = np.array([247.33, 107.73, 280.44, 460.83])
            bnds = [(100.0, 500.0), (50.0, 250.0), (150.0, 450.0), (200.0, 900.0)]
            cons = [
                {"type": "eq", "fun": lambda x: eval_x(x)[0]},
                {"type": "eq", "fun": lambda x: eval_x(x)[1] * 1e-8},
            ]
            label = "T2: Ropt locked, P free"
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    t_start = time.time()
    sol = minimize(
        lambda x: -eval_x(x)[3],
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 100, "eps": 1e-4},
    )
    print(
        f"  {mode} finished in {time.time() - t_start:.1f}s ({len(cache)} unique evals) -> {sol.message}"
    )

    c_ext = float(sol.x[0]) * 1e-12
    l1_v = float(sol.x[1]) * 1e-9
    l2_v = float(sol.x[2]) * 1e-9
    c2_v = float(sol.x[3]) * 1e-12
    ropt_v = float(sol.x[4]) if mode == "T1" else ana["Ropt"]
    return run_pa_point(
        "nonlin", r_on, l1_v, l2_v, c2_v, ropt_v, c_ext, vdd, f0, pgate, label
    )


def plot_pa_waveforms(pa_res: dict, f0: float) -> None:
    # th = 0.5 / f0
    sel = [
        ("A", "A: Const-C (20m)", "k--"),
        ("X", "X: Const-C design, real Coss (20m)", "C3-"),
        ("E", "E: Naive nonlinear (20m)", "C1-"),
        ("T1_20m", "T1 MILP [20m 1x]", "C0-"),
        ("T1_40m", "T1 MILP [40m 2x]", "C0--"),
        ("T2_20m", "T2 MILP [20m 1x]", "C4-"),
        ("T2_40m", "T2 MILP [40m 2x]", "C4--"),
        # ("T1", "T1: Ropt free", "C0-"),
        # ("T2", "T2: Ropt locked", "C4-"),
    ]
    fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    for key, lab, col in sel:
        h = pa_res[key]["h"]
        # ph = h["t"] / th
        ph = 2.0 * h["t"] / h["t"][-1]
        lw = 2.5 if key in ("X", "T1_20m", "T1_40m", "T2_20m", "T2_40m") else 1.2
        ax[0].plot(ph, h["v_D"], col, lw=lw, label=lab)
        ax[1].plot(ph, h["i_L"], col, lw=lw, label=lab)
        ax[2].plot(ph, h["i_sw"], col, lw=lw, label=lab)
        ax[3].plot(ph, h["q"] * 1e12, col, lw=lw, label=lab)
        ax[4].plot(ph, h["Coss"] * 1e12, col, lw=lw, label=lab)

    ax[0].axhline(PA_SPEC["VDD"], color="gray", ls=":")
    ax[0].set_ylabel("VDS (V)")
    ax[0].set_title(
        f"V_turnon: X={pa_res['X']['h']['v_turnon']*1e3:.0f} mV, "
        f"T2_20m={pa_res['T2_20m']['h']['v_turnon']*1e3:.0f} mV, "
        f"T2_40m={pa_res['T2_40m']['h']['v_turnon']*1e3:.0f} mV"
    )
    ax[1].set_ylabel("Load current (A)")
    ax[2].axhline(PA_SPEC["ID_MAX"], color="red", ls=":", alpha=0.6)
    ax[2].set_ylabel("Switch current (A)")
    ax[2].set_title(
        f"Peak i_sw: X={np.max(pa_res['X']['h']['i_sw']):.2f}A, "
        f"T1_20m={np.max(pa_res['T1_20m']['h']['i_sw']):.2f}A, "
        f"T2_20m={np.max(pa_res['T2_20m']['h']['i_sw']):.2f}A, "
        f"T2_40m={np.max(pa_res['T2_40m']['h']['i_sw']):.2f}A"
    )
    ax[3].set_ylabel("Drain charge (pC)")
    ax[4].set_ylabel("Coss (pF)")
    ax[4].set_xlabel("Phase (half-cycle units)")
    for a_i in ax:
        a_i.legend(fontsize=7)
        a_i.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pa_power(pa_res: dict, f0: float) -> None:
    # th = 0.5 / f0
    sel = [
        ("A", "A: Const-C (20m)", "k--"),
        ("X", "X: Const-C design, real Coss (20m)", "C3-"),
        ("T1_20m", "T1 MILP [20m 1x]", "C0-"),
        ("T1_40m", "T1 MILP [40m 2x]", "C0--"),
        ("T2_20m", "T2 MILP [20m 1x]", "C4-"),
        ("T2_40m", "T2 MILP [40m 2x]", "C4--"),
        # ("E", "E: Naive nonlinear", "C1-"),
        # ("T1", "T1: Ropt free", "C0-"),
        # ("T2", "T2: Ropt locked", "C4-"),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for key, lab, col in sel:
        h = pa_res[key]["h"]
        # ph = h["t"] / th
        ph = 2.0 * h["t"] / h["t"][-1]
        dev = pa_res[key]["dev"]
        i_dc = pa_res[key]["I"]
        ropt_v = pa_res[key]["Ropt_used"]
        p_src = i_dc * h["v_D"]
        p_sw = dev["Ron"] * h["i_sw"] ** 2
        p_load = ropt_v * h["i_L"] ** 2
        e_cap = dev_e_total(dev, h["v_D"])
        lw = 2.5 if key in ("X", "T1_20m", "T2_20m", "T2_40m") else 1.2
        ax[0].plot(ph, p_src, col, lw=lw, label=f"{key} P_src")
        ax[0].plot(ph, p_sw, col, lw=lw, ls="--", label=f"{key} P_sw")
        ax[1].plot(ph, e_cap * 1e9, col, lw=lw, label=lab)
        ax[2].plot(ph, p_src - p_sw - p_load, col, lw=lw, label=lab)

    ax[0].set_ylabel("Power (W)")
    ax[0].set_title("Instantaneous Power Flow")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("Stored Energy (nJ)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    ax[2].set_ylabel("Power Residual (W)")
    ax[2].set_xlabel("Phase (half-cycle units)")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pa_datasheet(pa_res: dict, f0: float) -> None:
    t_per = 1.0 / f0
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    vp = np.linspace(0.1, 100.0, 800)

    ax[0, 0].plot(DS_FIG5_VDS, DS_FIG5_COSS * 1e12, "ko", label="Datasheet")
    ax[0, 0].plot(vp, coss_const(vp) * 1e12, "k--", label="Const-C (used for design)")
    ax[0, 0].plot(vp, coss_tab(vp) * 1e12, "C3-", label="Table (real device)")
    ax[0, 0].set(xlabel="VDS (V)", ylabel="Coss (pF)", title="Fig 5: Coss vs VDS")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    tj = np.linspace(0.0, 150.0, 100)
    ax[0, 1].plot(DS_FIG8_TJ, DS_FIG8_RDS_NORM, "ko-", label="Datasheet points")
    ax[0, 1].plot(tj, _RDS_CURVE(tj), "C0--", label="PCHIP Interp")
    ax[0, 1].axvline(75.0, color="red", ls=":", label="Op point (75C)")
    ax[0, 1].set(
        xlabel="Tj (°C)", ylabel="Normalized RDS(on)", title="Fig 8: RDS(on) vs Temp"
    )
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    ds_fig10_vds = np.array([1.0, 10.0, 100.0])
    ds_fig10_id_1ms = np.array([2.5, 2.5, 0.5])
    ax[0, 2].loglog(ds_fig10_vds, [1.7, 1.7, 1.7], "k-", label="DC limit")
    ax[0, 2].loglog(ds_fig10_vds, ds_fig10_id_1ms, "C1--", label="1 ms pulse")
    # for k in ["A", "X", "E", "T1", "T2"]:
    for k in pa_res:
        ax[0, 2].plot(
            pa_res[k]["h"]["v_peak"],
            np.max(pa_res[k]["h"]["i_sw"]),
            "o",
            markersize=8,
            label=k,
        )
    ax[0, 2].set(xlabel="VDS (V)", ylabel="ID (A)", title="Fig 10: SOA")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    ds_fig11_t = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ds_fig11_zjc = np.array([0.001, 0.01, 0.1, 0.4, 1.0, 2.0])
    ax[1, 0].loglog(ds_fig11_t, ds_fig11_zjc, "ko-", label="ZJC")
    ax[1, 0].axvline(t_per / 2.0, color="red", ls=":", label="14MHz Half-Period")
    ax[1, 0].set(
        xlabel="Pulse Width (s)", ylabel="Zth (°C/W)", title="Fig 11: Thermal Response"
    )
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    ds_fig1_vds = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax[1, 1].plot(
        ds_fig1_vds, [0.0, 1.25, 2.2, 3.0, 3.8, 4.5, 5.0], "C0-", label="VGS=5V"
    )
    ax[1, 1].plot(
        ds_fig1_vds, [0.0, 1.0, 1.8, 2.4, 3.0, 3.5, 4.0], "C3-", label="VGS=4V"
    )
    ax[1, 1].plot(
        ds_fig1_vds, [0.0, 0.75, 1.2, 1.6, 1.9, 2.1, 2.3], "C2-", label="VGS=3V"
    )
    ax[1, 1].set(
        xlabel="VDS (V)", ylabel="ID (A)", title="Fig 1: Output Characteristics"
    )
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    ds_fig6_qg = np.array([0.0, 0.032, 0.055, 0.115])
    ds_fig6_vgs = np.array([0.0, 2.2, 2.2, 5.0])
    ax[1, 2].plot(ds_fig6_qg * 1e3, ds_fig6_vgs, "C4-")
    ax[1, 2].set(xlabel="QG (nC)", ylabel="VGS (V)", title="Fig 6: Gate Charge")
    ax[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# LAYER 2A: JOINT 20m -> 40m TX HARMONIC LPF — VECTORIZED ABCD ENGINE
# ==============================================================================
LPF_CAP_POOL = expand_series("E24", (1e-12, 10e-12, 100e-12, 1e-9), 4.7e-12, 4700e-12)
LPF_IND_POOL = expand_series("E12", (10e-9, 100e-9, 1e-6), 22e-9, 2200e-9)

TOPO_SPECS = {
    "LMatch_WB2CBA": {
        "names": ["Cm", "Lm", "C1", "L2", "C3", "L4", "C5"],
        "is_ind": [False, True, False, True, False, True, False],
        "seed_20m": [450e-12, 220e-9, 550e-12, 330e-9, 450e-12, 330e-9, 180e-12],
        "color": "#2563eb",
    },
    # "LMatch_WB2CBA": {
    #     "names": ["Cm", "Lm", "C1", "L2", "C3", "L4", "C5"],
    #     "is_ind": [False, True, False, True, False, True, False],
    #     "seed_20m": [120e-12, 100e-9, 180e-12, 560e-9, 390e-12, 560e-9, 180e-12],
    #     "color": "#2563eb",
    # },
    "JA2GQP": {
        "names": ["C1", "L1", "C2", "C3", "L2", "C4", "C5"],
        "is_ind": [False, True, False, False, True, False, False],
        "seed_20m": [200e-12, 390e-9, 82e-12, 360e-12, 390e-9, 33e-12, 150e-12],
        "color": "#059669",
    },
    "4Element": {
        "names": ["C1", "L2", "C2", "C3"],
        "is_ind": [False, True, False, False],
        "seed_20m": [270e-12, 560e-9, 56e-12, 270e-12],
        "color": "#d97706",
    },
}


def eval_abcd_vec(
    topo: str,
    x_mat: np.ndarray,
    freqs: np.ndarray,
    zs: float = 50.0,
    zl: float = 50.0,
    q_l: float = 60.0,
    q_c: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    w = (2.0 * np.pi * freqs)[None, :]
    n_runs, n_freq = x_mat.shape[0], freqs.shape[0]
    a_mat = np.ones((n_runs, n_freq), dtype=complex)
    b_mat = np.zeros((n_runs, n_freq), dtype=complex)
    c_mat = np.zeros((n_runs, n_freq), dtype=complex)
    d_mat = np.ones((n_runs, n_freq), dtype=complex)

    def sh_c(c_val: np.ndarray) -> None:
        nonlocal a_mat, b_mat, c_mat, d_mat
        y = 1j * w * c_val[:, None] + (w * c_val[:, None] / q_c)
        a_mat = a_mat + b_mat * y
        c_mat = c_mat + d_mat * y

    def ser_l(l_val: np.ndarray) -> None:
        nonlocal a_mat, b_mat, c_mat, d_mat
        z = (w * l_val[:, None] / q_l) + 1j * w * l_val[:, None]
        b_mat = a_mat * z + b_mat
        d_mat = c_mat * z + d_mat

    def notch_lc(l_val: np.ndarray, c_val: np.ndarray) -> None:
        nonlocal a_mat, b_mat, c_mat, d_mat
        z_l = (w * l_val[:, None] / q_l) + 1j * w * l_val[:, None]
        z = 1.0 / (1.0 / z_l + 1j * w * c_val[:, None])
        b_mat = a_mat * z + b_mat
        d_mat = c_mat * z + d_mat

    match topo:
        case "LMatch_WB2CBA":
            sh_c(x_mat[:, 0])
            ser_l(x_mat[:, 1])
            sh_c(x_mat[:, 2])
            ser_l(x_mat[:, 3])
            sh_c(x_mat[:, 4])
            ser_l(x_mat[:, 5])
            sh_c(x_mat[:, 6])
        case "JA2GQP":
            sh_c(x_mat[:, 0])
            notch_lc(x_mat[:, 1], x_mat[:, 2])
            sh_c(x_mat[:, 3])
            notch_lc(x_mat[:, 4], x_mat[:, 5])
            sh_c(x_mat[:, 6])
        case "4Element":
            sh_c(x_mat[:, 0])
            notch_lc(x_mat[:, 1], x_mat[:, 2])
            sh_c(x_mat[:, 3])
        case _:
            raise ValueError(f"Unknown LPF topology: {topo}")

    denom = a_mat * zl + b_mat + c_mat * zs * zl + d_mat * zs
    s21_db = 10.0 * np.log10(4.0 * zs * zl / (np.abs(denom) ** 2) + 1e-18)
    z_in = (a_mat * zl + b_mat) / (c_mat * zl + d_mat)
    s11_db = 20.0 * np.log10(np.abs((z_in - zs) / (z_in + zs)) + 1e-18)
    return s21_db, s11_db


def eval_lpf_band(
    topo: str,
    x_vec: np.ndarray | list[float],
    f0: float,
    zs: float = 50.0,
    zl: float = 50.0,
    harm_dbc: tuple[float, ...] = (-17.6, -31.1, -43.0, -46.0),
    q_l: float = 60.0,
    fcc_limit: float = -43.0,
) -> tuple[float, float, float, float, float]:
    freqs = np.array(
        [f0, 0.94 * f0, 1.06 * f0] + [k * f0 for k in range(2, 2 + len(harm_dbc))]
    )
    s21, s11 = eval_abcd_vec(
        topo, np.asarray(x_vec, float)[None, :], freqs, zs=zs, zl=zl, q_l=q_l
    )
    margins = [
        fcc_limit - (hl + float(s21[0, 3 + idx])) for idx, hl in enumerate(harm_dbc)
    ]
    return (
        float(s21[0, 0]),
        float(s11[0, 0]),
        float(s11[0, 1]),
        float(s11[0, 2]),
        float(min(margins)),
    )


def run_lpf_mc(
    topo: str,
    x_nom: np.ndarray | list[float],
    f0: float,
    zs: float = 50.0,
    zl: float = 50.0,
    harm_dbc: tuple[float, ...] = (-17.6, -31.1, -43.0, -46.0),
    q_l: float = 60.0,
    fcc_limit: float = -43.0,
    tol_val: float = 0.05,
    n_runs: int = 5000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    x_arr = np.asarray(x_nom, dtype=float)
    x_mc = x_arr[None, :] * rng.uniform(
        1.0 - tol_val, 1.0 + tol_val, (n_runs, len(x_arr))
    )
    freqs = np.array([f0] + [k * f0 for k in range(2, 2 + len(harm_dbc))])
    s21_db, s11_db = eval_abcd_vec(topo, x_mc, freqs, zs=zs, zl=zl, q_l=q_l)
    margins = fcc_limit - (np.asarray(harm_dbc)[None, :] + s21_db[:, 1:])
    return (
        float(np.max(s11_db[:, 0])),
        float(np.min(s21_db[:, 0])),
        float(np.min(margins)),
    )


def solve_joint_lpf_mip(
    topo: str,
    f20: float = F20,
    f40: float = F40,
    zs_20: float = 50.0,
    zs_40: float = 50.0,
    zl: float = 50.0,
    harm_dbc: tuple[float, ...] = (-17.6, -31.1, -43.0, -46.0),
    q_l_20: float = 60.0,
    q_l_40: float = 45.0,
    fcc_limit: float = -43.0,
    max_caps_per_node: int = 1,
    max_reels: int | None = 1,
    reel_penalty: float = 0.35,
    conv_mode: str = "DBL_LC",
) -> dict | None:
    """
    Fixed discrete component selection for all 4 20m -> 40m conversion modes:
      - 'DBL_LC' : 20m base E24/E12 + identical E24/E12 add-on on 40m (2x L, 2x C_base)
      - 'DBL_C'  : 20m base E24/E12 + single E24 ~3x add-on cap on 40m (1x L, C_base + E24_addon)
      - 'OPT_LC' : Independent E24/E12 selection on 20m and 40m
      - 'OPT_C'  : Fixed 20m E12 inductors + independent E24 cap selection on 40m
    """
    spec = TOPO_SPECS[topo]
    is_ind = spec["is_ind"]
    n_comp = len(is_ind)
    ind_indices = [i for i, ind in enumerate(is_ind) if ind]

    match conv_mode:
        case "DBL_LC":
            l_scale, c_scale = 2.0, 2.0
        case "DBL_C":
            l_scale, c_scale = 1.0, 4.0
        case "OPT_LC" | "OPT_C":
            l_scale, c_scale = 1.0, 1.0
        case _:
            raise ValueError(f"Unknown conv_mode: {conv_mode}")

    scale_vec = np.array([l_scale if ind else c_scale for ind in is_ind], dtype=float)
    cap_20_base = LPF_CAP_POOL[LPF_CAP_POOL <= LPF_CAP_POOL.max() / c_scale]
    ind_20_base = LPF_IND_POOL[LPF_IND_POOL <= LPF_IND_POOL.max() / l_scale]

    def snap_to_e24_addon(c20_total: float, target_total: float) -> tuple[float, str]:
        desired_addon = max(0.0, target_total - c20_total)
        if desired_addon <= 0.0:
            return c20_total, "none"
        idx = int(np.argmin(np.abs(LPF_CAP_POOL - desired_addon)))
        c_add = float(LPF_CAP_POOL.item(idx))
        return c20_total + c_add, fmt_eng(c_add, "F")

    def build_pools_and_bounds(
        cap_base: np.ndarray,
        ind_base: np.ndarray,
        fixed_inds: list[float] | None = None,
    ):
        lo_arr = np.array(
            [
                (
                    (fixed_inds[i] if fixed_inds is not None else ind_base.min())
                    if ind
                    else (cap_base.min() + (TRIM_LPF_C1_MID if i == 0 else 0.0))
                )
                for i, ind in enumerate(is_ind)
            ]
        )
        hi_arr = np.array(
            [
                (
                    (fixed_inds[i] if fixed_inds is not None else ind_base.max())
                    if ind
                    else (
                        max_caps_per_node * cap_base.max()
                        + (TRIM_LPF_C1_MID if i == 0 else 0.0)
                    )
                )
                for i, ind in enumerate(is_ind)
            ]
        )

        c_cands = list(cap_base)
        c_lbls = [fmt_eng(c, "F") for c in cap_base]
        if max_caps_per_node >= 2:
            for i1 in range(len(cap_base)):
                for i2 in range(i1, len(cap_base)):
                    csum = float(cap_base.item(i1) + cap_base.item(i2))
                    if csum <= max_caps_per_node * cap_base.max():
                        c_cands.append(csum)
                        c_lbls.append(
                            f"{fmt_eng(cap_base.item(i1), 'F')}+{fmt_eng(cap_base.item(i2), 'F')}"
                        )
        c_pool = np.array(c_cands)
        c1_pool = c_pool + TRIM_LPF_C1_MID
        c1_lbls = [
            (
                f"{lbl}+trim({fmt_eng(TRIM_LPF_C1_MID, 'F')})"
                if TRIM_LPF_C1_MID > 0
                else lbl
            )
            for lbl in c_lbls
        ]

        def get_pool(idx: int) -> tuple[np.ndarray, list[str]]:
            if is_ind[idx]:
                if fixed_inds is not None:
                    v_fix = fixed_inds[idx]
                    return np.array([v_fix]), [fmt_eng(v_fix, "H")]
                return ind_base, [fmt_eng(l, "H") for l in ind_base]
            if idx == 0:
                return c1_pool, c1_lbls
            return c_pool, c_lbls

        return lo_arr, hi_arr, ind_base, get_pool

    target_margin = 3.0

    def band_penalty(
        x_band: np.ndarray, f_band: float, zs_band: float, q_l_band: float
    ) -> float:
        s21, s11, s11_lo, s11_hi, min_m = eval_lpf_band(
            topo, x_band, f_band, zs_band, zl, harm_dbc, q_l_band, fcc_limit
        )
        pen = (
            max(0.0, target_margin - min_m) ** 2 * 30.0
            + max(0.0, -0.55 - s21) ** 2 * 400.0
            + max(0.0, s11 + 26.0) ** 2 * 120.0
            + max(0.0, s11_lo + 22.0) ** 2 * 50.0
            + max(0.0, s11_hi + 22.0) ** 2 * 50.0
        )
        match topo:
            case "JA2GQP":
                fn1 = 1.0 / (2.0 * np.pi * np.sqrt(x_band.item(1) * x_band.item(2)))
                fn2 = 1.0 / (2.0 * np.pi * np.sqrt(x_band.item(4) * x_band.item(5)))
                pen += ((fn1 - 2.0 * f_band) / f_band) ** 2 * 4000.0 + (
                    (fn2 - 3.0 * f_band) / f_band
                ) ** 2 * 4000.0
            case "4Element":
                fn1 = 1.0 / (2.0 * np.pi * np.sqrt(x_band.item(1) * x_band.item(2)))
                pen += ((fn1 - 2.0 * f_band) / f_band) ** 2 * 4000.0
            case _:
                pass
        return float(2.0 * max(s11, -30.0) + pen)

    def solve_single_or_coupled_milp(
        f_a: float,
        zs_a: float,
        ql_a: float,
        f_b: float | None,
        zs_b: float | None,
        ql_b: float | None,
        sc_b: np.ndarray | None,
        cap_base: np.ndarray,
        ind_base: np.ndarray,
        fixed_inds: list[float] | None = None,
    ) -> tuple[list[float], list[str], int] | None:
        lo_b, hi_b, ind_pool_eff, get_pool = build_pools_and_bounds(
            cap_base, ind_base, fixed_inds
        )
        # Tapered impedance seed from Zs (left) to Zl (right) instead of symmetric scaling
        f_rat = F20 / f_a
        seed_sc = np.empty(n_comp, dtype=float)
        for i, (val, ind) in enumerate(zip(spec["seed_20m"], is_ind)):
            alpha = 1.0 - (i / max(n_comp - 1, 1))
            z_local = (max(zs_a, 1.0) ** alpha) * (zl ** (1.0 - alpha))
            z_rat_i = np.sqrt(z_local / 50.0)
            seed_sc[i] = (val * f_rat * z_rat_i) if ind else (val * f_rat / z_rat_i)
        if fixed_inds is not None:
            for i, ind in enumerate(is_ind):
                if ind:
                    seed_sc[i] = fixed_inds[i]
        x_init = np.clip(seed_sc, lo_b * 1.01, hi_b * 0.99)

        free_inds = [i for i in ind_indices if fixed_inds is None]
        use_1reel = max_reels == 1 and len(free_inds) > 1

        def unpack(u: np.ndarray) -> np.ndarray:
            x = np.empty(n_comp)
            if use_1reel:
                l_sh = float(np.exp(u.item(0)))
                ui = 1
                for i, ind in enumerate(is_ind):
                    if ind:
                        x[i] = l_sh
                    else:
                        x[i] = float(np.exp(u.item(ui)))
                        ui += 1
            else:
                ui = 0
                for i, ind in enumerate(is_ind):
                    if ind and fixed_inds is not None:
                        x[i] = fixed_inds[i]
                    else:
                        x[i] = float(np.exp(u.item(ui)))
                        ui += 1
            return x

        if use_1reel:
            l_g = float(
                np.prod([x_init.item(i) for i in free_inds]) ** (1.0 / len(free_inds))
            )
            u0 = np.log(
                [l_g] + [x_init.item(i) for i, ind in enumerate(is_ind) if not ind]
            )
        else:
            u0 = np.log(
                [
                    x_init.item(i)
                    for i, ind in enumerate(is_ind)
                    if not (ind and fixed_inds is not None)
                ]
            )

        def obj_nlp(u: np.ndarray) -> float:
            xa = unpack(u)
            bpen = (
                np.sum(
                    np.maximum(0.0, lo_b - xa) / lo_b
                    + np.maximum(0.0, xa - hi_b) / hi_b
                )
                ** 2
                * 5000.0
            )
            xa_c = np.clip(xa, lo_b, hi_b)
            _, s11_a0, _, _, _ = eval_lpf_band(
                topo, xa_c, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
            )
            tot = (
                bpen
                + band_penalty(xa_c, f_a, zs_a, ql_a)
                + 3000.0 * (10.0 ** (s11_a0 / 10.0))
            )
            if (
                f_b is not None
                and sc_b is not None
                and zs_b is not None
                and ql_b is not None
            ):
                _, s11_b0, _, _, _ = eval_lpf_band(
                    topo, sc_b * xa_c, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                )
                tot += band_penalty(sc_b * xa_c, f_b, zs_b, ql_b) + 3000.0 * (
                    10.0 ** (s11_b0 / 10.0)
                )
            return tot

        res_nlp = minimize(
            obj_nlp,
            u0,
            method="Nelder-Mead",
            options={"maxiter": 8000, "xatol": 1e-5, "fatol": 1e-4},
        )
        x_star = np.clip(unpack(res_nlp.x), lo_b, hi_b)
        u_star = np.log(x_star)

        # 1. Diagonal Hessian H_ii = d^2(|Gamma|^2)/du_i^2 at x_star
        _, s11_star_a, _, _, _ = eval_lpf_band(
            topo, x_star, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
        )
        g0_a = 10.0 ** (s11_star_a / 10.0)
        g0_b = 0.0
        if (
            f_b is not None
            and sc_b is not None
            and zs_b is not None
            and ql_b is not None
        ):
            _, s11_star_b, _, _, _ = eval_lpf_band(
                topo, sc_b * x_star, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
            )
            g0_b = 10.0 ** (s11_star_b / 10.0)

        h_gamma = np.zeros(n_comp)
        eps_h = 2e-2
        for i in range(n_comp):
            if is_ind[i] and fixed_inds is not None:
                continue
            du = np.zeros(n_comp)
            du[i] = eps_h
            xp, xn = np.exp(u_star + du), np.exp(u_star - du)
            _, sp, _, _, _ = eval_lpf_band(
                topo, xp, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
            )
            _, sn, _, _, _ = eval_lpf_band(
                topo, xn, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
            )
            d2_a = max(
                0.0,
                (10.0 ** (sp / 10.0) - 2.0 * g0_a + 10.0 ** (sn / 10.0)) / (eps_h**2),
            )
            d2_b = 0.0
            if (
                f_b is not None
                and sc_b is not None
                and zs_b is not None
                and ql_b is not None
            ):
                _, spb, _, _, _ = eval_lpf_band(
                    topo, sc_b * xp, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                )
                _, snb, _, _, _ = eval_lpf_band(
                    topo, sc_b * xn, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                )
                d2_b = max(
                    0.0,
                    (10.0 ** (spb / 10.0) - 2.0 * g0_b + 10.0 ** (snb / 10.0))
                    / (eps_h**2),
                )
            h_gamma[i] = d2_a + d2_b

        offsets, cur = [], 0
        for i in range(n_comp):
            pi, _ = get_pool(i)
            offsets.append(cur)
            cur += len(pi)

        nl_eff = 1 if fixed_inds is not None else len(ind_pool_eff)
        idx_u = cur
        idx_s11_a = idx_u + nl_eff
        idx_sm_a = idx_s11_a + 1
        idx_s11_b = idx_sm_a + 1
        idx_sm_b = idx_s11_b + 1

        match topo:
            case "JA2GQP":
                notch_pairs = [(1, 2, 2.0 * f_a), (4, 5, 3.0 * f_a)]
            case "4Element":
                notch_pairs = [(1, 2, 2.0 * f_a)]
            case _:
                notch_pairs = []

        idx_ns = idx_sm_b + 1
        n_vars = idx_ns + len(notch_pairs)
        integrality = np.ones(n_vars)
        integrality[idx_s11_a:] = 0
        ub_vars = np.ones(n_vars)
        ub_vars[idx_s11_a:idx_ns] = np.inf
        ub_vars[idx_ns:] = 0.16

        def exact_discrete_score(x_cand: np.ndarray) -> float:
            s21_a, s11_a, s11_lo_a, s11_hi_a, m_a = eval_lpf_band(
                topo, x_cand, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
            )
            sc = (
                max(s11_a, -35.0)
                + max(0.0, s11_a + 20.0) * 15.0
                + max(0.0, s11_lo_a + 18.0) * 5.0
                + max(0.0, s11_hi_a + 18.0) * 5.0
                + max(0.0, target_margin - m_a) * 25.0
                + max(0.0, -0.60 - s21_a) * 50.0
            )
            if (
                f_b is not None
                and sc_b is not None
                and zs_b is not None
                and ql_b is not None
            ):
                s21_b, s11_b, s11_lo_b, s11_hi_b, m_b = eval_lpf_band(
                    topo, sc_b * x_cand, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                )
                sc += (
                    max(s11_b, -35.0)
                    + max(0.0, s11_b + 20.0) * 15.0
                    + max(0.0, s11_lo_b + 18.0) * 5.0
                    + max(0.0, s11_hi_b + 18.0) * 5.0
                    + max(0.0, target_margin - m_b) * 25.0
                    + max(0.0, -0.60 - s21_b) * 50.0
                )
            for idx_l, idx_c, f_n in notch_pairs:
                fn_d = 1.0 / (2.0 * np.pi * np.sqrt(x_cand[idx_l] * x_cand[idx_c]))
                sc += abs(fn_d - f_n) / f_n * 200.0
            return float(sc)

        # 2. Sequential Trust-Region MILP Loop (4 iterations)
        x_lin = x_star.copy()
        trust_radii = [np.inf, 0.45, 0.28, 0.18]
        best_res: tuple[list[float], list[str], int] | None = None
        best_score = np.inf

        for tr_rad in trust_radii:
            u_lin = np.log(x_lin)
            _, s11_a, _, _, m_a = eval_lpf_band(
                topo, x_lin, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
            )
            j_s11_a, j_m_a = np.zeros(n_comp), np.zeros(n_comp)
            j_s11_b, j_m_b = np.zeros(n_comp), np.zeros(n_comp)
            s11_b, m_b = -30.0, 10.0
            if (
                f_b is not None
                and sc_b is not None
                and zs_b is not None
                and ql_b is not None
            ):
                _, s11_b, _, _, m_b = eval_lpf_band(
                    topo, sc_b * x_lin, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                )

            eps = 1e-3
            for i in range(n_comp):
                if is_ind[i] and fixed_inds is not None:
                    continue
                du = np.zeros(n_comp)
                du[i] = eps
                xp, xn = np.exp(u_lin + du), np.exp(u_lin - du)
                _, sp, _, _, mp = eval_lpf_band(
                    topo, xp, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
                )
                _, sn, _, _, mn = eval_lpf_band(
                    topo, xn, f_a, zs_a, zl, harm_dbc, ql_a, fcc_limit
                )
                j_s11_a[i] = (sp - sn) / (2.0 * eps)
                j_m_a[i] = (mp - mn) / (2.0 * eps)
                if (
                    f_b is not None
                    and sc_b is not None
                    and zs_b is not None
                    and ql_b is not None
                ):
                    _, spb, _, _, mpb = eval_lpf_band(
                        topo, sc_b * xp, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                    )
                    _, snb, _, _, mnb = eval_lpf_band(
                        topo, sc_b * xn, f_b, zs_b, zl, harm_dbc, ql_b, fcc_limit
                    )
                    j_s11_b[i] = (spb - snb) / (2.0 * eps)
                    j_m_b[i] = (mpb - mnb) / (2.0 * eps)

            c_obj = np.zeros(n_vars)
            for i, (ind, tgt) in enumerate(zip(is_ind, x_star)):
                pool, lbls = get_pool(i)
                du_star = np.log(pool / tgt)
                par_pen = np.array(
                    [
                        0.0 if ind else (0.04 if "+" in lbls[k] else 0.0)
                        for k in range(len(pool))
                    ]
                )
                quad_s11_pen = 1500.0 * h_gamma[i] * (du_star**2)
                c_obj[offsets[i] : offsets[i] + len(pool)] = (
                    (0.5 if ind else 0.3) * np.abs(du_star) + quad_s11_pen + par_pen
                )
            c_obj[idx_u : idx_u + nl_eff] = reel_penalty
            c_obj[idx_s11_a] = 120.0
            c_obj[idx_sm_a] = 15.0
            c_obj[idx_s11_b] = 120.0 if f_b is not None else 0.0
            c_obj[idx_sm_b] = 15.0 if f_b is not None else 0.0
            for j in range(len(notch_pairs)):
                c_obj[idx_ns + j] = 1500.0

            ub_iter = ub_vars.copy()
            if np.isfinite(tr_rad):
                for i in range(n_comp):
                    if is_ind[i] and fixed_inds is not None:
                        continue
                    pool, _ = get_pool(i)
                    in_tr = np.abs(np.log(pool) - u_lin[i]) <= tr_rad
                    if np.any(in_tr):
                        ub_iter[offsets[i] : offsets[i] + len(pool)] = in_tr.astype(
                            float
                        )

            rows, lb, ub = [], [], []
            for i in range(n_comp):
                pool, _ = get_pool(i)
                r = np.zeros(n_vars)
                r[offsets[i] : offsets[i] + len(pool)] = 1.0
                rows.append(r)
                lb.append(1.0)
                ub.append(1.0)

            if fixed_inds is None:
                for i, ind in enumerate(is_ind):
                    if ind:
                        for k in range(nl_eff):
                            r = np.zeros(n_vars)
                            r[offsets[i] + k] = 1.0
                            r[idx_u + k] = -1.0
                            rows.append(r)
                            lb.append(-np.inf)
                            ub.append(0.0)
                if max_reels is not None:
                    r = np.zeros(n_vars)
                    r[idx_u : idx_u + nl_eff] = 1.0
                    rows.append(r)
                    lb.append(1.0)
                    ub.append(float(max_reels))

            for j, (idx_l, idx_c, f_n) in enumerate(notch_pairs):
                log_lc = -2.0 * np.log(2.0 * np.pi * f_n)
                pl, _ = get_pool(idx_l)
                pc, _ = get_pool(idx_c)
                r1, r2 = np.zeros(n_vars), np.zeros(n_vars)
                r1[offsets[idx_l] : offsets[idx_l] + len(pl)] = np.log(pl)
                r1[offsets[idx_c] : offsets[idx_c] + len(pc)] = np.log(pc)
                r1[idx_ns + j] = -1.0
                rows.append(r1)
                lb.append(-np.inf)
                ub.append(log_lc)

                r2[offsets[idx_l] : offsets[idx_l] + len(pl)] = -np.log(pl)
                r2[offsets[idx_c] : offsets[idx_c] + len(pc)] = -np.log(pc)
                r2[idx_ns + j] = -1.0
                rows.append(r2)
                lb.append(-np.inf)
                ub.append(-log_lc)

            bands_lin = [(j_s11_a, s11_a, idx_s11_a, j_m_a, m_a, idx_sm_a)]
            if f_b is not None:
                bands_lin.append((j_s11_b, s11_b, idx_s11_b, j_m_b, m_b, idx_sm_b))

            for js, sref, is_sl, jm, mref, im_sl in bands_lin:
                rs, rm = np.zeros(n_vars), np.zeros(n_vars)
                for i, u0_i in enumerate(u_lin):
                    pool, _ = get_pool(i)
                    rs[offsets[i] : offsets[i] + len(pool)] = js[i] * (
                        np.log(pool) - u0_i
                    )
                    rm[offsets[i] : offsets[i] + len(pool)] = jm[i] * (
                        np.log(pool) - u0_i
                    )
                rs[is_sl] = -1.0
                rm[im_sl] = 1.0
                rows.append(rs)
                lb.append(-np.inf)
                ub.append(-26.0 - sref)
                rows.append(rm)
                lb.append(target_margin - mref)
                ub.append(np.inf)

            sol = milp(
                c=c_obj,
                integrality=integrality,
                bounds=Bounds(np.zeros(n_vars), ub_iter),
                constraints=LinearConstraint(np.array(rows), lb, ub),
            )
            if not sol.success:
                break

            v = np.round(sol.x[:idx_s11_a]).astype(int)
            x_d, lbl_d = [], []
            for i in range(n_comp):
                pool, lbls = get_pool(i)
                ks = int(np.argmax(v[offsets[i] : offsets[i] + len(pool)]))
                x_d.append(float(pool.item(ks)))
                lbl_d.append(lbls[ks])
            reels = len(set(x_d[i] for i in ind_indices))

            sc_cur = exact_discrete_score(np.array(x_d))
            if sc_cur < best_score:
                best_score = sc_cur
                best_res = (x_d, lbl_d, reels)

            if np.allclose(x_d, x_lin, rtol=1e-9):
                break
            x_lin = np.array(x_d)

        return best_res

    if conv_mode in ("DBL_LC", "DBL_C"):
        res20 = solve_single_or_coupled_milp(
            f20, zs_20, q_l_20, f40, zs_40, q_l_40, scale_vec, cap_20_base, ind_20_base
        )
        if res20 is None:
            return None
        x20_disc, labels_20, n_reels = res20
        x40_disc, labels_40 = [], []
        for i in range(n_comp):
            if is_ind[i]:
                if conv_mode == "DBL_LC":
                    x40_disc.append(float(2.0 * x20_disc[i]))
                    labels_40.append(f"{labels_20[i]}+{labels_20[i]}")
                else:
                    x40_disc.append(float(x20_disc[i]))
                    labels_40.append(labels_20[i])
            else:
                trim_off = TRIM_LPF_C1_MID if i == 0 else 0.0
                c20_fixed = x20_disc[i] - trim_off
                if conv_mode == "DBL_LC":
                    x40_disc.append(float(x20_disc[i] + c20_fixed))
                    base_lbl = labels_20[i].split("+trim")[0]
                    labels_40.append(f"{labels_20[i]} || +{base_lbl}")
                else:
                    target_tot = 4.0 * c20_fixed + trim_off
                    c40_tot, add_lbl = snap_to_e24_addon(x20_disc[i], target_tot)
                    x40_disc.append(c40_tot)
                    labels_40.append(f"{labels_20[i]} || +{add_lbl}")
    else:
        res20 = solve_single_or_coupled_milp(
            f20, zs_20, q_l_20, None, None, None, None, LPF_CAP_POOL, LPF_IND_POOL
        )
        if res20 is None:
            return None
        x20_disc, labels_20, _ = res20
        fixed_l = x20_disc if conv_mode == "OPT_C" else None
        res40 = solve_single_or_coupled_milp(
            f40,
            zs_40,
            q_l_40,
            None,
            None,
            None,
            None,
            LPF_CAP_POOL,
            LPF_IND_POOL,
            fixed_inds=fixed_l,
        )
        if res40 is None:
            return None
        x40_disc, labels_40, _ = res40
        n_reels = len(
            set(x20_disc[i] for i in ind_indices)
            | set(x40_disc[i] for i in ind_indices)
        )

    return {
        "topo": topo,
        "conv_mode": conv_mode,
        "cap_only": conv_mode in ("DBL_C", "OPT_C"),
        "x20": x20_disc,
        "x40": x40_disc,
        "labels_20": labels_20,
        "labels_40": labels_40,
        "n_reels": n_reels,
        "metrics_20": eval_lpf_band(
            topo, x20_disc, f20, zs_20, zl, harm_dbc, q_l_20, fcc_limit
        ),
        "mc_20": run_lpf_mc(
            topo, x20_disc, f20, zs_20, zl, harm_dbc, q_l_20, fcc_limit
        ),
        "metrics_40": eval_lpf_band(
            topo, x40_disc, f40, zs_40, zl, harm_dbc, q_l_40, fcc_limit
        ),
        "mc_40": run_lpf_mc(
            topo, x40_disc, f40, zs_40, zl, harm_dbc, q_l_40, fcc_limit
        ),
    }


def plot_joint_lpf_uq(
    results_by_key: dict[str, dict],
    zs_20: float = 50.0,
    zs_40: float = 50.0,
    zl: float = 50.0,
    harm_dbc: tuple[float, ...] = (-17.6, -31.1, -43.0, -46.0),
    fcc_limit: float = -43.0,
    q_l_20: float = 60.0,
    q_l_40: float = 45.0,
    tol_val: float = 0.05,
    n_runs: int = 2000,
) -> None:
    rng = np.random.default_rng(42)
    mode_styles = {"1R_1C": "-", "1R_2C": "--", "MR_1C": "-.", "MR_2C": ":"}
    first_item = next(iter(results_by_key.values()))
    conv_tag = "L fest, 4x C" if first_item.get("cap_only", False) else "2x L, 2x C"

    band_cfg = [
        (
            f"20m Base Build (14.10 MHz, Q=60, {zs_20:.2f}Ω → {zl:.0f}Ω)",
            F20,
            "x20",
            zs_20,
            q_l_20,
        ),
        (
            f"40m Converted [{conv_tag}] (7.05 MHz, Q=45, {zs_40:.2f}Ω → {zl:.0f}Ω)",
            F40,
            "x40",
            zs_40,
            q_l_40,
        ),
    ]

    for band_title, f0, x_key, zs_band, q_l_band in band_cfg:
        freqs = np.linspace(0.7 * f0, 5.5 * f0, 1800)
        f_mhz = freqs / 1e6
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10.5))

        for label, item in results_by_key.items():
            topo = item["topo"]
            col = TOPO_SPECS[topo]["color"]
            ls_med = next(
                (st for tag, st in mode_styles.items() if label.endswith("_" + tag)),
                "-",
            )
            x_arr = np.asarray(item[x_key], dtype=float)
            x_mc = x_arr[None, :] * rng.uniform(
                1.0 - tol_val, 1.0 + tol_val, (n_runs, len(x_arr))
            )
            s21, s11 = eval_abcd_vec(topo, x_mc, freqs, zs=zs_band, zl=zl, q_l=q_l_band)
            s21_nom, s11_nom = eval_abcd_vec(
                topo, x_arr[None, :], freqs, zs=zs_band, zl=zl, q_l=q_l_band
            )

            p5_21, p50_21, p95_21 = np.percentile(s21, [5, 50, 95], axis=0)
            p5_11, p50_11, p95_11 = np.percentile(s11, [5, 50, 95], axis=0)
            lbl_base = f"{label} ({item['n_reels']}R, ${3 * item['n_reels']})"

            ax1.fill_between(f_mhz, p5_21, p95_21, alpha=0.14, color=col)
            ax1.plot(f_mhz, s21_nom[0], "--", color=col, lw=0.9)
            ax1.plot(f_mhz, p50_21, color=col, ls=ls_med, lw=1.9, label=lbl_base)

            ax2.fill_between(f_mhz, p5_11, p95_11, alpha=0.14, color=col)
            ax2.plot(f_mhz, s11_nom[0], "--", color=col, lw=0.9)
            ax2.plot(f_mhz, p50_11, color=col, ls=ls_med, lw=1.9, label=lbl_base)

        for n in range(1, 6):
            ax1.axvline(n * f0 / 1e6, color="grey", ls=":", alpha=0.5)
        harm_freqs = [(k * f0) / 1e6 for k in range(2, 2 + len(harm_dbc))]
        req_atten = [fcc_limit - hl for hl in harm_dbc]
        ax1.scatter(
            harm_freqs,
            req_atten,
            color="#dc2626",
            marker="v",
            s=85,
            zorder=6,
            label="FCC -43 dBc Limit",
        )
        ax1.set(
            xlim=(0.7 * f0 / 1e6, 5.5 * f0 / 1e6),
            ylim=(-85, 3),
            xlabel="Frequency (MHz)",
            ylabel="S21 (dB)",
            title=f"{band_title} — S21 Harmonic Attenuation (±5% MC)",
        )
        ax1.grid(alpha=0.3)
        ax1.legend(loc="lower left", fontsize=8.0, ncol=2)

        ax2.axhline(-20.0, color="red", ls="--", lw=1.3, label="S11 Target (-20 dB)")
        ax2.axhline(-15.0, color="#64748b", ls=":", lw=1.2, label="S11 Limit (-15 dB)")
        ax2.axvline(f0 / 1e6, color="black", ls=":", alpha=0.6)
        ax2.set(
            xlim=(0.95 * f0 / 1e6, 1.05 * f0 / 1e6),
            ylim=(-55, 0),
            xlabel="Frequency (MHz)",
            ylabel="S11 (dB)",
            title=f"{band_title} — Passband Return Loss S11 (±5% MC)",
        )
        ax2.grid(alpha=0.3)
        ax2.legend(loc="lower right", fontsize=8.0, ncol=2)

        plt.tight_layout()
        plt.show()


# ==============================================================================
# LAYER 3: RX AUDIO CHAIN (TAYLOE MIXER + ARCH A + ADC SETTLING + RC2 + CIC/FIR)
# ==============================================================================
FS, FS1, FS2, FSOUT = 192e3, 48e3, 24e3, 8e3
SSB = 2.7e3
FCRIT = FS - FSOUT / 2.0
FCRIT_HI = FS + FSOUT / 2.0

BLOCKER_DBFS = -30.0
ALIAS_BELOW_NOISE_DB = 10.0
ADC_SNR_TYP = 68.0
ADC_VREF, ADC_BITS = 3.10, 12
ADC_LSB = ADC_VREF / (2**ADC_BITS)
ADC_SNR_NOISE_DBFS = -ADC_SNR_TYP - 10.0 * np.log10(FS / SSB)
ALIAS_THRESHOLD_DBFS = ADC_SNR_NOISE_DBFS - ALIAS_BELOW_NOISE_DB
TARGET_ALIAS = -(BLOCKER_DBFS - ALIAS_THRESHOLD_DBFS)
MAX_PASS_LOSS_DB = 0.5
Q_MIN, Q_MAX = 0.60, 0.85

# Tayloe Mixer (20m = 1x 2.2 nF; 40m = 2x 2.2 nF parallel = 4.4 nF via DNP pad)
R_SOURCE_RF, R_ON_TAYLOE = 8.22, 5.0
R_SOURCE_AUDIO = 4.0 * (R_SOURCE_RF + R_ON_TAYLOE)
C_TAYLOE_20M = 2.2e-9
C_TAYLOE_40M = 2.0 * C_TAYLOE_20M

# ADC Kickback & Acquisition
C_ADC, R_ADC_SWITCH, T_ACQ = 6e-12, 0.0, 2600e-9
R_KICK, C_KICK = 100.0, 1e-9
R_ADC_LEAK = 1e12
SETTLE_LIMIT = 0.5 / (2**ADC_BITS)
RC_CMAX = 100e-9

PASSIVE_MODEL = {
    "C0G": (0.05, 0.99),
    "C0G2": (0.02, 0.99),
    "C0G1": (0.01, 0.99),
    "X5R": (0.20, 0.95),
    "R1pct": (0.01, 0.99),
    "L10pct": (0.10, 0.95),
}
CAP_CLASSES = {"C0G": 0.0, "C0G2": 0.15, "C0G1": 0.50}
PAR_COST = 0.04


def tol(cls: str) -> float:
    return PASSIVE_MODEL[cls][0]


def draw_by_class(
    rng: np.random.Generator, cls: str, n: int, dist: str = "truncnorm"
) -> np.ndarray:
    t, y = PASSIVE_MODEL[cls]
    match dist:
        case "uniform":
            return rng.uniform(1.0 - t, 1.0 + t, n)
        case "worstcase":
            return rng.choice([1.0 - t, 1.0 + t], size=n)
        case "truncnorm":
            sigma = t / (np.sqrt(2.0) * erfinv(y))
            out, filled = np.empty(n), 0
            while filled < n:
                s = rng.normal(0.0, sigma, n - filled)
                s = s[np.abs(s) <= t]
                out[filled : filled + len(s)] = 1.0 + s
                filled += len(s)
            return out
        case _:
            raise ValueError(f"Unknown tolerance distribution: {dist}")


def wrap_freq(f: np.ndarray | float, fs: float) -> np.ndarray:
    return (np.asarray(f) + fs / 2.0) % fs - fs / 2.0


def tayloe_tf(
    f: np.ndarray | float,
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> np.ndarray:
    fc_t = 1.0 / (2.0 * np.pi * r_src_audio * c_tayloe)
    return 1.0 / (1.0 + 1j * np.asarray(f) / fc_t)


def arch_a_tf(
    f: np.ndarray | float,
    c_ac: float,
    r_in: float,
    r_fb: float,
    c_fb: float,
    ra: float,
    ca: float,
    cb: float,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> np.ndarray:
    s = 2j * np.pi * np.asarray(f, float)
    r_tot = r_in + r_src_audio
    h_hp = s * r_tot * c_ac / (1.0 + s * r_tot * c_ac)
    h_p1 = 1.0 / (1.0 + s * r_fb * c_fb)
    w0 = 1.0 / (ra * np.sqrt(ca * cb))
    q_val = np.sqrt(ca / cb) / 2.0
    h_sk = w0**2 / (s**2 + s * w0 / q_val + w0**2)
    return h_hp * h_p1 * h_sk


HB_FIR = firwin(63, 12e3, fs=FS1)
POLY_FIR = firwin(161, 3.35e3, fs=FS2)


def cic_tf(f: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(f, float)
    den = 4.0 * np.sin(np.pi * arr / FS)
    num = np.sin(4.0 * np.pi * arr / FS)
    ratio = np.ones_like(arr)
    mask = np.abs(den) > 1e-12
    ratio[mask] = num[mask] / den[mask]
    return np.abs(ratio) ** 4


def fir_tf(f: np.ndarray | float, coeff: np.ndarray, fs: float) -> np.ndarray:
    _, h = freqz(coeff, worN=2.0 * np.pi * np.asarray(f) / fs)
    return np.abs(h)


def digital_chain_tf(f: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(f, float)
    return (
        cic_tf(wrap_freq(arr, FS))
        * fir_tf(wrap_freq(arr, FS1), HB_FIR, FS1)
        * fir_tf(wrap_freq(arr, FS2), POLY_FIR, FS2)
    )


def passive_rc2_tf(
    f: np.ndarray | float,
    R1: float,
    R2: float,
    C1: float,
    C2: float,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> np.ndarray:
    s = 2j * np.pi * np.asarray(f, float)
    g1, g2, gl = 1.0 / (r_src_audio + R1), 1.0 / R2, 1.0 / R_ADC_LEAK
    det = (g1 + g2 + s * C1) * (g2 + gl + s * C2) - g2**2
    return g1 * g2 / det


def adc_settle_error(
    r_rec: np.ndarray | float,
    c_hold: np.ndarray | float,
    t_acq: float = T_ACQ,
    c_adc: float = C_ADC,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    e0 = c_adc / (c_hold + c_adc)
    tau = (r_rec + R_ADC_SWITCH) * (c_hold + c_adc)
    return e0 * np.exp(-t_acq / tau), e0, tau


RX_NAMES = ["c_ac", "r_in", "r_fb", "c_fb", "ra", "ca", "cb"]
RX_OPT = ["c_fb", "ra", "ca", "cb"]
RX_CAPS = {"c_ac", "c_fb", "ca", "cb"}
RX_SIGNS = np.array(list(itertools.product((-1, 1), repeat=len(RX_NAMES))))

RX_ROLE = {
    "c_ac": "AC-coupling capacitor",
    "r_in": "Diff-amp input resistor",
    "r_fb": "Diff-amp feedback resistor",
    "c_fb": "Diff-amp feedback cap (pole 1)",
    "ra": "Sallen-Key R (both resistors)",
    "ca": "Sallen-Key Ca (feedback cap)",
    "cb": "Sallen-Key Cb (cap to ground)",
}

VAL0 = {
    "c_ac": 10e-6,
    "r_in": 1e3,
    "r_fb": 50e3,
    "c_fb": 220e-12,
    "ra": 10e3,
    "ca": 1.5e-9,
    "cb": 750e-12,
}
CLS0 = {
    "c_ac": "X5R",
    "r_in": "R1pct",
    "r_fb": "R1pct",
    "c_fb": "C0G",
    "ra": "R1pct",
    "ca": "C0G",
    "cb": "C0G",
}
LBL0 = {n: fmt_eng(VAL0[n], "F" if n in RX_CAPS else "") for n in RX_NAMES}

F_A = np.linspace(FCRIT, FCRIT_HI, 401)
F_P = np.linspace(300.0, SSB, 120)
F_FC = np.logspace(3.0, 5.0, 1500)

RX_CAP_POOL = expand_series("E24", (1e-12, 10e-12, 100e-12, 1e-9), 47e-12, 10e-9)
RX_R_POOL = expand_series("E96", (1e3, 10e3), 4e3, 30e3)

RC_NAMES = ["R1", "R2", "C1", "C2"]
RC_SIGNS = np.array(list(itertools.product((-1, 1), repeat=4)))
RC_R_POOL = expand_series("E96", (1.0, 10.0, 100.0, 1e3, 1e4, 1e5), 10.0, 100e3)
RC_C_POOL = expand_series(
    "E24", (1e-12, 10e-12, 100e-12, 1e-9, 10e-9, 100e-9), 100e-12, RC_CMAX
)


def _fc_from(h_db: np.ndarray) -> np.ndarray:
    below = h_db <= -3.0103
    has = below.any(axis=1)
    idx = np.clip(below.argmax(axis=1), 1, None)[:, None]
    h0 = np.take_along_axis(h_db, idx - 1, 1)[:, 0]
    h1 = np.take_along_axis(h_db, idx, 1)[:, 0]
    l0, l1 = np.log(F_FC[idx[:, 0] - 1]), np.log(F_FC[idx[:, 0]])
    fc = np.exp(l0 + (-3.0103 - h0) / (h1 - h0 + 1e-30) * (l1 - l0))
    return np.where(has, fc, 0.0)


def rx_metrics_vec(
    x: dict[str, np.ndarray | list[float]],
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
    chunk: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = tayloe_tf(F_A, c_tayloe, r_src_audio) * digital_chain_tf(F_A)
    dp = tayloe_tf(F_P, c_tayloe, r_src_audio) * digital_chain_tf(F_P)
    n = len(next(iter(x.values())))
    out = []
    for i in range(0, n, chunk):
        xc = {k: np.asarray(v, float)[i : i + chunk, None] for k, v in x.items()}
        alias = db(da * arch_a_tf(F_A, **xc, r_src_audio=r_src_audio)).max(axis=1)
        pas = np.abs(db(dp * arch_a_tf(F_P, **xc, r_src_audio=r_src_audio))).max(axis=1)
        h_db = db(arch_a_tf(F_FC, **xc, r_src_audio=r_src_audio)) - db(
            arch_a_tf(np.array([100.0]), **xc, r_src_audio=r_src_audio)
        )
        out.append(np.column_stack([alias, pas, _fc_from(h_db)]))
    stacked = np.vstack(out)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def rx_metrics_nom(
    val: dict[str, float],
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> np.ndarray:
    a, p, fc = rx_metrics_vec(
        {n: [val[n]] for n in RX_NAMES}, c_tayloe=c_tayloe, r_src_audio=r_src_audio
    )
    return np.array([float(a[0]), float(p[0]), float(fc[0]) / 1e3])


def rx_worst_case(
    val: dict[str, float],
    cls: dict[str, str],
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> tuple[float, float, float, float, float]:
    t = np.array([tol(cls[n]) for n in RX_NAMES])
    v = np.array([val[n] for n in RX_NAMES])
    x_mat = v * (1.0 + RX_SIGNS * t)
    x = {n: x_mat[:, i] for i, n in enumerate(RX_NAMES)}
    a, p, fc = rx_metrics_vec(x, c_tayloe=c_tayloe, r_src_audio=r_src_audio)
    q = np.sqrt(x["ca"] / x["cb"]) / 2.0
    return (
        float(a.max()),
        float(p.max()),
        float(fc.min()),
        float(q.min()),
        float(q.max()),
    )


def rx_verify(
    val: dict[str, float],
    cls: dict[str, str],
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> tuple[float, float, float, float]:
    a, p, fc, qlo, qhi = rx_worst_case(
        val, cls, c_tayloe=c_tayloe, r_src_audio=r_src_audio
    )
    viol = (
        max(0.0, a - TARGET_ALIAS)
        + max(0.0, p - MAX_PASS_LOSS_DB)
        + max(0.0, Q_MIN - qlo)
        + max(0.0, qhi - Q_MAX)
    )
    return viol, fc, a, p


def rx_monte_carlo(
    val: dict[str, float],
    cls: dict[str, str],
    c_tayloe: float = C_TAYLOE_20M,
    r_src_audio: float = R_SOURCE_AUDIO,
    runs: int = 5000,
    seed: int = 1,
    dist: str = "truncnorm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = {n: val[n] * draw_by_class(rng, cls[n], runs, dist) for n in RX_NAMES}
    return rx_metrics_vec(x, c_tayloe=c_tayloe, r_src_audio=r_src_audio)


def rx_build_candidates(
    name: str,
    center: float,
    window: float = 0.35,
    max_par: int = 2,
    cap_classes: dict[str, float] | None = None,
) -> list[dict]:
    lo, hi = center * (1.0 - window), center * (1.0 + window)
    match name:
        case "ra":
            combos = [(float(v), fmt_eng(v), 1) for v in RX_R_POOL if lo <= v <= hi]
            classes = {"R1pct": 0.0}
        case _:
            combos = [
                (float(v), fmt_eng(v, "F"), 1) for v in RX_CAP_POOL if lo <= v <= hi
            ]
            if max_par >= 2:
                for i, a in enumerate(RX_CAP_POOL):
                    for b in RX_CAP_POOL[i:]:
                        if lo <= a + b <= hi:
                            combos.append(
                                (
                                    float(a + b),
                                    f"{fmt_eng(a, 'F')}+{fmt_eng(b, 'F')}",
                                    2,
                                )
                            )
            classes = cap_classes or CAP_CLASSES
    return [
        {"v": v, "label": lbl, "cls": c, "tol": tol(c), "cost": k + PAR_COST * (n - 1)}
        for v, lbl, n in combos
        for c, k in classes.items()
    ]


def rx_jacobian(
    val: dict[str, float], r_src_audio: float = R_SOURCE_AUDIO, eps: float = 1e-3
) -> np.ndarray:
    k = len(RX_NAMES)
    x_mat = {n: np.full(2 * k, val[n], float) for n in RX_NAMES}
    for i, n in enumerate(RX_NAMES):
        x_mat[n][i] *= np.exp(eps)
        x_mat[n][k + i] *= np.exp(-eps)
    a, p, fc = rx_metrics_vec(x_mat, c_tayloe=C_TAYLOE_20M, r_src_audio=r_src_audio)
    m = np.column_stack([a, p, fc / 1e3])
    return (m[:k] - m[k:]) / (2.0 * eps)


def rx_solve_step(
    val: dict[str, float],
    cls: dict[str, str],
    cands: dict[str, list[dict]],
    delta: float,
    margin: float,
    r_src_audio: float = R_SOURCE_AUDIO,
    cost_w: float = 0.02,
    slack_w: float = 1e3,
) -> dict[str, dict] | None:
    j_mat = rx_jacobian(val, r_src_audio=r_src_audio)
    m0 = rx_metrics_nom(val, C_TAYLOE_20M, r_src_audio=r_src_audio)
    ix = {n: i for i, n in enumerate(RX_NAMES)}
    sel = {
        n: [c for c in cands[n] if abs(np.log(c["v"] / val[n])) <= delta]
        for n in RX_OPT
    }
    if any(not sel[n] for n in RX_OPT):
        return None

    off, cur = {}, 0
    for n in RX_OPT:
        off[n] = cur
        cur += len(sel[n])
    nv = cur + 2
    i_sa, i_sp = cur, cur + 1

    c = np.zeros(nv)
    c[[i_sa, i_sp]] = slack_w
    for n in RX_OPT:
        for k, cd in enumerate(sel[n]):
            dx = np.log(cd["v"] / val[n])
            c[off[n] + k] = (
                -(j_mat[ix[n], 2] * dx - abs(j_mat[ix[n], 2]) * cd["tol"])
                + cost_w * cd["cost"]
            )

    rows, lo, hi = [], [], []
    for n in RX_OPT:
        r = np.zeros(nv)
        r[off[n] : off[n] + len(sel[n])] = 1.0
        rows.append(r)
        lo.append(1.0)
        hi.append(1.0)

    fixed = [n for n in RX_NAMES if n not in RX_OPT]
    for mi, slack, limit in (
        (0, i_sa, TARGET_ALIAS - margin),
        (1, i_sp, MAX_PASS_LOSS_DB - 0.5 * margin),
    ):
        r = np.zeros(nv)
        for n in RX_OPT:
            for k, cd in enumerate(sel[n]):
                dx = np.log(cd["v"] / val[n])
                r[off[n] + k] = (
                    j_mat[ix[n], mi] * dx + abs(j_mat[ix[n], mi]) * cd["tol"]
                )
        r[slack] = -1.0
        const = sum(abs(j_mat[ix[n], mi]) * tol(cls[n]) for n in fixed)
        rows.append(r)
        lo.append(-np.inf)
        hi.append(limit - m0[mi] - const)

    r_lo, r_hi = np.zeros(nv), np.zeros(nv)
    for k, cd in enumerate(sel["ca"]):
        r_lo[off["ca"] + k] = np.log(cd["v"]) + np.log(1.0 - cd["tol"])
        r_hi[off["ca"] + k] = np.log(cd["v"]) + np.log(1.0 + cd["tol"])
    for k, cd in enumerate(sel["cb"]):
        r_lo[off["cb"] + k] = -(np.log(cd["v"]) + np.log(1.0 + cd["tol"]))
        r_hi[off["cb"] + k] = -(np.log(cd["v"]) + np.log(1.0 - cd["tol"]))
    rows += [r_lo, r_hi]
    lo += [2.0 * np.log(2.0 * Q_MIN), -np.inf]
    hi += [np.inf, 2.0 * np.log(2.0 * Q_MAX)]

    sol = milp(
        c=c,
        integrality=np.r_[np.ones(cur), 0, 0],
        bounds=Bounds(np.zeros(nv), np.r_[np.ones(cur), np.inf, np.inf]),
        constraints=LinearConstraint(np.array(rows), lo, hi),
    )
    if not sol.success:
        return None
    return {
        n: sel[n][int(np.argmax(sol.x[off[n] : off[n] + len(sel[n])]))] for n in RX_OPT
    }


def optimize_rx_audio(
    val: dict[str, float],
    cls: dict[str, str],
    lbl: dict[str, str],
    delta: float = 0.15,
    margin: float = 0.0,
    max_iter: int = 40,
    window: float = 0.35,
    max_par: int = 2,
    cap_classes: dict[str, float] | None = None,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], bool]:
    cands = {
        n: rx_build_candidates(n, val[n], window, max_par, cap_classes) for n in RX_OPT
    }
    lbl_out = dict(lbl)
    viol, fc, _, _ = rx_verify(val, cls, C_TAYLOE_20M, r_src_audio=r_src_audio)

    for _ in range(max_iter):
        pick = rx_solve_step(val, cls, cands, delta, margin, r_src_audio=r_src_audio)
        if pick is None:
            delta *= 0.6
            if delta < 0.01:
                break
            continue

        v2 = {**val, **{n: pick[n]["v"] for n in RX_OPT}}
        c2 = {**cls, **{n: pick[n]["cls"] for n in RX_OPT}}
        if all(
            np.isclose(v2[n], val[n], rtol=1e-9) and c2[n] == cls[n] for n in RX_OPT
        ):
            break

        viol2, fc2, _, _ = rx_verify(v2, c2, C_TAYLOE_20M, r_src_audio=r_src_audio)
        if (viol2 < viol - 1e-9) or (viol2 <= 1e-9 and viol <= 1e-9 and fc2 > fc + 1.0):
            val, cls, viol, fc = v2, c2, viol2, fc2
            for n in RX_OPT:
                lbl_out[n] = pick[n]["label"]
        else:
            delta *= 0.6
            margin = min(margin + 0.1, 0.6)
            if delta < 0.01:
                break

    return val, cls, lbl_out, viol <= 1e-9


def design_passive_rc2(
    cap_cls: str,
    r_src_audio: float = R_SOURCE_AUDIO,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, str], dict[str, float], bool]:
    cls = {"R1": "R1pct", "R2": "R1pct", "C1": cap_cls, "C2": cap_cls}
    da = tayloe_tf(F_A, C_TAYLOE_20M, r_src_audio) * digital_chain_tf(F_A)
    dp = tayloe_tf(F_P, C_TAYLOE_20M, r_src_audio) * digital_chain_tf(F_P)

    def decode(ix: np.ndarray) -> dict[str, float]:
        i = np.round(ix).astype(int)
        return {
            "R1": float(RC_R_POOL[i[0]]),
            "R2": float(RC_R_POOL[i[1]]),
            "C1": float(RC_C_POOL[i[2]]),
            "C2": float(RC_C_POOL[i[3]]),
        }

    def rc2_worst(v_dict: dict[str, float]) -> dict[str, float]:
        t = np.array([tol(cls[n]) for n in RC_NAMES])
        v = np.array([v_dict[n] for n in RC_NAMES])
        x_mat = v * (1.0 + RC_SIGNS * t)
        p = {n: x_mat[:, i][:, None] for i, n in enumerate(RC_NAMES)}
        alias = db(da * passive_rc2_tf(F_A, **p, r_src_audio=r_src_audio)).max(axis=1)
        pas = np.abs(db(dp * passive_rc2_tf(F_P, **p, r_src_audio=r_src_audio))).max(
            axis=1
        )
        h_db = db(passive_rc2_tf(F_FC, **p, r_src_audio=r_src_audio)) - db(
            passive_rc2_tf(np.array([100.0]), **p, r_src_audio=r_src_audio)
        )
        fc = _fc_from(h_db)
        serr = adc_settle_error(x_mat[:, 1], x_mat[:, 3])
        return {
            "alias": float(alias.max()),
            "pas": float(pas.max()),
            "fc_min": float(fc.min()),
            "settle": float(np.max(serr)),
            "c_ratio": float((x_mat[:, 2] / x_mat[:, 3]).min()),
        }

    def rc2_viol(w: dict[str, float]) -> float:
        return (
            max(0.0, w["alias"] - TARGET_ALIAS)
            + max(0.0, w["pas"] - MAX_PASS_LOSS_DB)
            + max(0.0, np.log10(max(w["settle"], 1e-300) / SETTLE_LIMIT))
            + max(0.0, np.log10(5.0 / w["c_ratio"]))
        )

    bounds = [(0, len(RC_R_POOL) - 1)] * 2 + [(0, len(RC_C_POOL) - 1)] * 2
    res = differential_evolution(
        lambda ix: 1e3 * rc2_viol(rc2_worst(decode(ix)))
        - rc2_worst(decode(ix))["fc_min"] / 1e3,
        bounds,
        integrality=[True] * 4,
        seed=seed,
        popsize=40,
        maxiter=300,
        tol=1e-9,
        polish=False,
    )
    val = decode(res.x)
    w = rc2_worst(val)
    return val, cls, w, rc2_viol(w) <= 1e-9


def print_rx_bom(
    title: str, val: dict[str, float], cls: dict[str, str], lbl: dict[str, str]
) -> None:
    print(f"\n{title}")
    print("  " + "-" * 100)
    print(
        f"  {'Ref':<5} {'Role':<33} {'Value':>9} {'Build':<14} {'Class':<7} "
        f"{'Tol':>5} {'Min':>9} {'Max':>9}  {'Exact [SI]':>14}"
    )
    print("  " + "-" * 100)
    for n in RX_NAMES:
        unit = "F" if n in RX_CAPS else ""
        t = tol(cls[n])
        print(
            f"  {n:<5} {RX_ROLE[n]:<33} {fmt_eng(val[n], unit):>9} {lbl[n]:<14} {cls[n]:<7} "
            f"{t * 100:>4.0f}% {fmt_eng(val[n] * (1.0 - t), unit):>9} {fmt_eng(val[n] * (1.0 + t), unit):>9}  "
            f"{val[n]:>14.6g}"
        )
    print("  " + "-" * 100)


def print_rx_derived(
    title: str,
    val: dict[str, float],
    cls: dict[str, str],
    r_src_audio: float = R_SOURCE_AUDIO,
) -> None:
    r_tot = val["r_in"] + r_src_audio
    gain = val["r_fb"] / r_tot
    f_hp = 1.0 / (2.0 * np.pi * r_tot * val["c_ac"])
    f_p1 = 1.0 / (2.0 * np.pi * val["r_fb"] * val["c_fb"])
    f0 = 1.0 / (2.0 * np.pi * val["ra"] * np.sqrt(val["ca"] * val["cb"]))
    q_val = np.sqrt(val["ca"] / val["cb"]) / 2.0
    _, _, fc_min, qlo, qhi = rx_worst_case(
        val, cls, C_TAYLOE_20M, r_src_audio=r_src_audio
    )
    m = rx_metrics_nom(val, C_TAYLOE_20M, r_src_audio=r_src_audio)
    print(f"\n{title}")
    print(
        f"  signal gain (r_fb / (r_in + R_src)) : {gain:.2f} V/V = {20.0 * np.log10(gain):.2f} dB (R_src = {r_src_audio:.2f} ohm)"
    )
    print(f"  AC-coupling HPF corner               : {f_hp:.2f} Hz")
    print(f"  diff-amp feedback pole (pole 1)      : {f_p1 / 1e3:.3f} kHz")
    print(f"  Sallen-Key f0 / Q                    : {f0 / 1e3:.3f} kHz / {q_val:.4f}")
    print(
        f"  cascade -3 dB corner (nominal)       : {m[2]:.3f} kHz | worst-corner fc_min: {fc_min / 1e3:.3f} kHz"
    )
    print(
        f"  Q over tolerance corners             : {qlo:.4f} .. {qhi:.4f} (allowed {Q_MIN}..{Q_MAX})"
    )
    print(
        f"  nominal alias / passband dev         : {m[0]:+.2f} dB (limit {TARGET_ALIAS:+.2f}) / {m[1]:.3f} dB (limit {MAX_PASS_LOSS_DB})"
    )


def plot_rx_uq(
    val: dict[str, float],
    cls: dict[str, str],
    name: str,
    ref_val: dict[str, float] | None = None,
    r_src_audio: float = R_SOURCE_AUDIO,
    runs: int = 5000,
) -> None:
    f = np.union1d(np.linspace(0.0, 1.5 * FS, 3000), F_A)
    mask = (f >= FCRIT) & (f <= FCRIT_HI)
    rng = np.random.default_rng(1)
    fac = {n: draw_by_class(rng, cls[n], runs) for n in RX_NAMES}
    base = tayloe_tf(f, C_TAYLOE_20M, r_src_audio=r_src_audio) * digital_chain_tf(f)
    mc = np.empty((runs, len(f)), np.float32)
    for i in range(0, runs, 500):
        x = {n: (val[n] * fac[n][i : i + 500])[:, None] for n in RX_NAMES}
        mc[i : i + 500] = db(base * arch_a_tf(f, **x, r_src_audio=r_src_audio))

    worst = mc[:, mask].max(axis=1)
    print_rx_quantile_table(name, worst)

    p2_5, p5, p50, p95, p97_5, p99 = np.percentile(
        mc, [2.5, 5, 50, 95, 97.5, 99], axis=0
    )
    nom = db(base * arch_a_tf(f, **val, r_src_audio=r_src_audio))
    ref = (
        None
        if ref_val is None
        else db(base * arch_a_tf(f, **ref_val, r_src_audio=r_src_audio))
    )
    fk = f / 1e3

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    for ax, color in zip(axes, ("steelblue", "darkorange")):
        ax.fill_between(
            fk, p2_5, p97_5, alpha=0.14, color=color, label="95% tolerance interval"
        )
        ax.fill_between(
            fk, p5, p95, alpha=0.28, color=color, label="90% tolerance interval"
        )
        if ref is not None:
            ax.plot(
                fk,
                ref,
                color="gray",
                ls="--",
                lw=1.3,
                label="Original design (nominal)",
            )
        ax.plot(fk, nom, "k-", lw=1.4, label=f"{name} (nominal)")
        ax.plot(fk, p50, color=color, lw=1.5, label="Median (MC)")
        ax.plot(fk, p99, ":", color="navy", lw=1.3, label="p99 (MC)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, which="both", alpha=0.3)

    ax1, ax2 = axes
    ax1.axvspan(
        FCRIT / 1e3, FCRIT_HI / 1e3, color="orange", alpha=0.15, label="1st alias zone"
    )
    ax1.axhline(
        TARGET_ALIAS,
        color="red",
        ls="--",
        lw=1.2,
        label=f"Alias target ({TARGET_ALIAS:.2f} dB)",
    )
    ax1.set(
        xlim=(0.01, 250.0),
        ylim=(-165.0, 5.0),
        xlabel="Frequency (kHz)",
        title=f"{name} — Complete RX Chain UQ",
    )
    ax1.legend(loc="lower left", fontsize=8.5)

    ax2.axvspan(0.3, SSB / 1e3, color="green", alpha=0.12, label="SSB voice passband")
    ax2.axhline(
        -MAX_PASS_LOSS_DB,
        color="red",
        ls="--",
        lw=1.1,
        label=f"Max droop (-{MAX_PASS_LOSS_DB} dB)",
    )
    ax2.set_xscale("log")
    ax2.set(
        xlim=(0.01, 6.0),
        ylim=(-10.0, 1.0),
        xlabel="Frequency (kHz, log)",
        title="Decoder Passband & AC-Coupling HPF UQ",
    )
    ax2.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.show()


def plot_ra_sweep(
    design: dict,
    r_src_audio: float = R_SOURCE_AUDIO,
    lo: float = 6.8e3,
    hi: float = 14e3,
) -> None:
    ras = RX_R_POOL[(RX_R_POOL >= lo) & (RX_R_POOL <= hi)]
    rows = []
    for ra in ras:
        val = {**design["val"], "ra": float(ra)}
        a, p, fc_min, qlo, qhi = rx_worst_case(
            val, design["cls"], C_TAYLOE_20M, r_src_audio=r_src_audio
        )
        if (
            a <= TARGET_ALIAS
            and p <= MAX_PASS_LOSS_DB
            and Q_MIN <= qlo
            and qhi <= Q_MAX
        ):
            rows.append((ra, fc_min, a))

    print(
        f"  {design['name']}: {len(rows)} feasible / {len(ras)} E96 values of ra in {lo / 1e3:.1f}–{hi / 1e3:.1f} kOhm"
    )
    if not rows:
        return
    r = np.array(rows)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(
        r[:, 0] / 1e3,
        r[:, 1] / 1e3,
        "o-",
        color="steelblue",
        label="Worst-corner -3 dB corner",
    )
    ax1.axvline(
        design["val"]["ra"] / 1e3,
        color="gray",
        ls=":",
        label=f"chosen ra = {fmt_eng(design['val']['ra'])}",
    )
    ax1.set(xlabel="Sallen-Key RA (kΩ)", ylabel="Worst-corner -3 dB corner (kHz)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        r[:, 0] / 1e3,
        r[:, 2],
        "s-",
        color="darkorange",
        label="Worst-corner alias response",
    )
    ax2.axhline(
        TARGET_ALIAS, color="red", ls="--", label=f"Alias limit ({TARGET_ALIAS:.2f} dB)"
    )
    ax2.set_ylabel("Worst-corner alias response (dB)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title(
        f"Bandwidth vs Alias Rejection (Feasible E96 RA Values) — {design['name']}"
    )
    plt.tight_layout()
    plt.show()


def plot_rx_analog_compare(
    designs: list[dict],
    rc2_designs: list[dict],
    r_src_audio: float = R_SOURCE_AUDIO,
    fmax: float = 24e3,
) -> None:
    f = np.linspace(1.0, fmax, 6000)
    f_ref = np.array([100.0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    for ax in (ax1, ax2):
        for d in designs:
            h = arch_a_tf(f, **d["val"], r_src_audio=r_src_audio)
            h0 = abs(arch_a_tf(f_ref, **d["val"], r_src_audio=r_src_audio))
            fc_nom = float(rx_metrics_nom(d["val"], r_src_audio=r_src_audio)[2])
            ax.plot(
                f / 1e3,
                db(h / h0),
                lw=2.0,
                color=d["color"],
                label=f"{d['name']} (fc {fc_nom:.2f} kHz)",
            )
        for d in rc2_designs:
            h = passive_rc2_tf(f, **d["val"], r_src_audio=r_src_audio)
            h0 = abs(passive_rc2_tf(f_ref, **d["val"], r_src_audio=r_src_audio))
            hr = db(h / h0)
            fc = f[np.argmax(hr <= -3.0103)] if (hr <= -3.0103).any() else np.nan
            ax.plot(
                f / 1e3,
                hr,
                lw=2.0,
                ls="--",
                color=d["color"],
                label=f"{d['name']} ({'feasible' if d['ok'] else 'infeasible'}; fc {fc / 1e3:.2f} kHz)",
            )
        ax.axvspan(
            0.3, SSB / 1e3, color="green", alpha=0.10, label="SSB voice passband"
        )
        ax.axhline(-3.0103, color="red", ls=":", lw=1.1, label="-3 dB")
        ax.axvline(
            FSOUT / 2e3, color="gray", ls=":", lw=1.1, label="Output Nyquist (4 kHz)"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("Normalized magnitude (dB)")

    ax1.set(
        xlim=(0.0, fmax / 1e3),
        ylim=(-60.0, 3.0),
        xlabel="Frequency (kHz)",
        title=f"Analog Filter Responses, 0–{fmax / 1e3:.0f} kHz",
    )
    ax1.legend(loc="lower left", fontsize=8.0)
    ax2.axhline(
        -MAX_PASS_LOSS_DB,
        color="red",
        ls="--",
        lw=1.1,
        label=f"Max droop (-{MAX_PASS_LOSS_DB} dB)",
    )
    ax2.set(
        xlim=(0.0, 6.0),
        ylim=(-4.0, 1.0),
        xlabel="Frequency (kHz)",
        title="Passband Detail",
    )
    ax2.legend(loc="lower left", fontsize=8.0)
    plt.tight_layout()
    plt.show()


def print_pa_full_report(
    ana: dict, pa_res: dict, r_on_25: float, pgate_20: float, pgate_40: float
) -> None:
    """Restores full EPC2037 Datasheet Card, Case details, and (T - A) Delta table."""
    print("\nEPC2037 DATASHEET CARD & COSS MODEL SUMMARY")
    print("-" * 72)
    print(
        f"  V_DS_max = {PA_SPEC['VDS_MAX']:.1f} V, I_D_max = {PA_SPEC['ID_MAX']:.2f} A"
    )
    print(
        f"  R_on,typ = {PA_SPEC['RON_25']*1e3:.0f} mΩ (@75°C: {r_on_25*temperature_factor(75.0):.4f} Ω), R_on,max = {PA_SPEC['RON_MAX_25']*1e3:.0f} mΩ"
    )
    print(
        f"  Q_G = {PA_SPEC['QG']*1e12:.0f} pC, Q_GS = {PA_SPEC['QGS']*1e12:.0f} pC, Q_GD = {PA_SPEC['QGD']*1e12:.0f} pC, R_G = {PA_SPEC['RG']} Ω"
    )
    print(
        f"  C_ISS = {PA_SPEC['CISS']*1e12:.1f} pF, C_RSS = {PA_SPEC['CRSS']*1e12:.2f} pF"
    )
    print(
        f"  C_OSS(50V) = {PA_SPEC['COSS_50']*1e12:.1f} pF (max {PA_SPEC['COSS_50_MAX']*1e12:.1f} pF), C_OSS(ER) = {PA_SPEC['COSS_ER']*1e12:.1f} pF, C_OSS(TR) = {PA_SPEC['COSS_TR']*1e12:.1f} pF"
    )
    print(
        f"  Integrated Qoss(50V) = {Q_RAW_50*1e12:.2f} pC (Datasheet nom {PA_SPEC['QOSS_50']*1e12:.0f} pC) | Model Eoss(50V) = {eoss_tab(50.0)*1e9:.4f} nJ"
    )
    print(
        f"  Analytical Sokal: KR={ana['KR']:.5f}, KC={ana['KC']:.5f}, KL={ana['KL']:.5f}, phi={np.degrees(ana['phi']):.2f}°"
    )
    print(
        f"  IDC Sokal = {ana['IDC_sk']*1e3:.2f} mA, IR Sokal = {ana['IR_sk']*1e3:.2f} mA | P_gate(20m) = {pgate_20*1e3:.3f} mW, P_gate(40m) = {pgate_40*1e3:.3f} mW"
    )

    for k, p in pa_res.items():
        h = p["h"]
        print(f"\n--- Case {k} ({p['label']}) ---")
        print(
            f"  I_DC = {p['I']*1e3:.3f} mA | Avg drain V = {h['v_avg']:.5f} V | V_pk = {h['v_peak']:.3f} V | V_min = {h['v_min']:.3f} V"
        )
        print(
            f"  V_turnon = {h['v_turnon']*1e3:+.1f} mV | i_turnon = {h['i_turnon']:+.4f} A | i_sw_pk = {np.max(h['i_sw']):.4f} A"
        )
        print(
            f"  P_load = {p['P_load']*1e3:.2f} mW | P_Ron = {p['P_Ron']*1e3:.2f} mW | eta_sys = {p['eta_system']*100:.2f}%"
        )
        print(
            f"  E_cap_avg = {p['E_cap_avg']*1e9:.4f} nJ | E_cap_delta = {p['E_cap_delta']*1e9:+.3e} nJ"
        )

    delta_rows = []
    for k in ["T1_20m", "T2_20m"]:
        if k not in pa_res:
            continue
        delta_rows.append(
            {
                "Tag": k,
                "ΔCext (pF)": round(
                    (pa_res[k]["Cext_used"] - ana["Cext_const"]) * 1e12, 2
                ),
                "ΔL1 (nH)": round((pa_res[k]["L1_used"] - ana["L1"]) * 1e9, 2),
                "ΔL2 (nH)": round((pa_res[k]["L2_used"] - ana["L2"]) * 1e9, 2),
                "ΔC2 (pF)": round((pa_res[k]["C2_used"] - ana["C2"]) * 1e12, 2),
                "ΔRopt (Ω)": round(pa_res[k]["Ropt_used"] - ana["Ropt"], 3),
            }
        )
    if delta_rows:
        print("\n### ABSOLUTE DELTAS FROM ANALYTICAL 20m SOKAL DESIGN (T_20m - A)\n")
        print(pd.DataFrame(delta_rows).to_string(index=False))


def print_rx_quantile_table(name: str, worst_alias: np.ndarray) -> None:
    """Restores the 6-level sigma/quantile alias rejection table from AudioOpt."""
    print(f"\n{name} - alias rejection statistics")
    print("-" * 68)
    print(
        f"{'Level':<10} {'Quantile':>11} {'Response':>13} {'Margin':>12} {'Verdict':>9}"
    )
    levels = [
        ("Median", 50.0),
        ("1 sigma", 84.1345),
        ("2 sigma", 97.725),
        ("3 sigma", 99.865),
        ("4 sigma", 99.99683),
        ("Maximum", 100.0),
    ]
    for label, q in levels:
        v = float(np.percentile(worst_alias, q))
        print(
            f"{label:<10} {q:>10.5f}% {v:>+12.2f} dB {TARGET_ALIAS - v:>+10.2f} dB "
            f"{'PASS' if TARGET_ALIAS - v >= 0 else 'FAIL':>9}"
        )
    print("-" * 68)


def print_rx_full_comparison(
    rx_designs: list[dict],
    rc2_designs: list[dict],
    settle_a_lsb: float,
    r_src_audio: float = R_SOURCE_AUDIO,
) -> None:
    """Restores Sections 9, 9b, and Consolidated 2x2 Comparison Tables from AudioOpt."""
    section("3B. FINAL COMPARISON — ARCH A DESIGNS & PASSIVE RC2")
    w_col = 20
    hdr = f"{'':<36}" + "".join(f"{d['name']:>{w_col}}" for d in rx_designs)
    print(
        f"Alias zone {FCRIT/1e3:.1f}–{FCRIT_HI/1e3:.1f} kHz | Target {TARGET_ALIAS:+.2f} dB\n"
    )
    print("Components (value [class]); a+b = two caps in parallel")
    print(hdr)
    print("-" * len(hdr))
    for n in RX_NAMES:
        print(
            f"{n + ' ' + RX_ROLE[n][:30]:<36}"
            + "".join(
                f"{d['lbl'][n] + ' [' + d['cls'][n] + ']':>{w_col}}" for d in rx_designs
            )
        )

    def row(label: str, vals: list[str]) -> None:
        print(f"{label:<36}" + "".join(f"{v:>{w_col}}" for v in vals))

    def design_cost(d: dict) -> float:
        return sum(
            CAP_CLASSES.get(d["cls"][n], 0.0) + PAR_COST * ("+" in d["lbl"][n])
            for n in RX_OPT
            if n in RX_CAPS
        )

    nom = [
        rx_metrics_nom(d["val"], C_TAYLOE_20M, r_src_audio=r_src_audio)
        for d in rx_designs
    ]
    wc20 = [
        rx_worst_case(d["val"], d["cls"], C_TAYLOE_20M, r_src_audio=r_src_audio)
        for d in rx_designs
    ]
    wc40 = [
        rx_worst_case(d["val"], d["cls"], C_TAYLOE_40M, r_src_audio=r_src_audio)
        for d in rx_designs
    ]
    gain_db = 20.0 * np.log10(VAL0["r_fb"] / (VAL0["r_in"] + r_src_audio))
    settle_ok = settle_a_lsb <= 0.5

    fc0_wc = wc20[0][2]

    print("\nPerformance (20m Base & 40m 2x Tayloe)")
    print(hdr)
    print("-" * len(hdr))
    row("nominal fc [kHz]", [f"{m.item(2):.3f}" for m in nom])
    row("worst-corner fc [kHz]", [f"{fc_m / 1e3:.3f}" for _, _, fc_m, _, _ in wc20])
    row(
        "worst-corner fc vs original",
        [f"{(fc_m / fc0_wc - 1.0) * 100:+.1f}%" for _, _, fc_m, _, _ in wc20],
    )
    row("nominal alias 20m [dB]", [f"{m.item(0):+.2f}" for m in nom])
    row("worst-corner alias 20m [dB]", [f"{a:+.2f}" for a, _, _, _, _ in wc20])
    row("worst-corner alias 40m (2x) [dB]", [f"{a:+.2f}" for a, _, _, _, _ in wc40])
    row(
        f"alias margin 20m vs {TARGET_ALIAS:+.2f} dB",
        [f"{TARGET_ALIAS - a:+.2f}" for a, _, _, _, _ in wc20],
    )
    row("worst-corner passband dev [dB]", [f"{pas:.3f}" for _, pas, _, _, _ in wc20])
    row("Q over corners", [f"{qlo:.3f}..{qhi:.3f}" for _, _, _, qlo, qhi in wc20])
    row("ADC residual [LSB] (limit 0.5)", [f"{settle_a_lsb:.3g}"] * len(rx_designs))
    row("signal gain [dB]", [f"{gain_db:.1f}"] * len(rx_designs))
    row(
        "# parallel-built caps",
        [str(sum(1 for n in RX_CAPS if "+" in d["lbl"][n])) for d in rx_designs],
    )
    row(
        "# C0G1 (1%) caps",
        [str(sum(1 for n in RX_OPT if d["cls"][n] == "C0G1")) for d in rx_designs],
    )
    row(
        "relative cap cost (model units)", [f"{design_cost(d):.2f}" for d in rx_designs]
    )
    row(
        "verdict (20m+40m corners+settle)",
        ["PASS" if (d["ok"] and settle_ok) else "FAIL" for d in rx_designs],
    )

    print("\nPASSIVE RC2 COMPARISON (Same Chain, Corners, Pools & Settling)")
    hdr2 = f"{'':<36}" + "".join(f"{d['name']:>{w_col}}" for d in rc2_designs)
    print(hdr2)
    print("-" * len(hdr2))
    for n in RC_NAMES:
        row(
            n,
            [
                f"{fmt_eng(d['val'][n], 'F' if n.startswith('C') else '')} [{d['cls'][n]}]"
                for d in rc2_designs
            ],
        )
    row("worst-corner alias [dB]", [f"{d['w']['alias']:+.2f}" for d in rc2_designs])
    row("worst-corner passband dev [dB]", [f"{d['w']['pas']:.3f}" for d in rc2_designs])
    row("worst-corner fc [kHz]", [f"{d['w']['fc_min']/1e3:.3f}" for d in rc2_designs])
    row(
        "ADC residual [LSB] (limit 0.5)",
        [f"{d['w']['settle']*2**ADC_BITS:.3g}" for d in rc2_designs],
    )
    row("verdict", ["FEASIBLE" if d["ok"] else "INFEASIBLE" for d in rc2_designs])


def print_consolidated_bom_summary(
    pa_joint: dict,
    pa_label: str,
    lpf_combined_results: dict[str, dict],
    rx_designs: list[dict],
) -> None:
    """
    Prints consolidated JLCPCB 20m Base BOM + 40m Conversion BOM for all feasible designs
    across PA, all 4 TX Match+LPF conversion modes, and RX Audio.
    """
    section(
        "4. CONSOLIDATED 20m JLCPCB BASE BOM & 40m CONVERSION CHECKLIST (ALL FEASIBLE DESIGNS)"
    )

    # --- 1. Class-E PA ---
    bom_pa = pa_joint["bom_20"]
    cext_40_val = pa_joint["20m"]["Cext_used"]
    print(f"\n[STAGE 1: CLASS-E PA — {pa_label} (Ropt = {pa_joint['ropt']:.2f} Ω)]")
    print(
        f"  20m JLCPCB Base : Cext = {bom_pa['Cext']} | L1 = {bom_pa['L1']} | L2 = {bom_pa['L2']} | C2 = {bom_pa['C2']}"
    )
    print(
        f"  40m Hand Add-On : +{fmt_eng(cext_40_val, 'F')} parallel on Cext DNP pad | "
        f"+{bom_pa['L1']} in series on L1 (remove 0R) | +{bom_pa['L2']} in series on L2 (remove 0R) | "
        f"+{bom_pa['C2']} parallel on C2 DNP pad"
    )

    # --- 2. TX Match + Harmonic LPF across all 4 conversion modes ---
    print("\n[STAGE 2: TX IMPEDANCE MATCH + HARMONIC LPF — ALL FEASIBLE DESIGNS]")
    feasible_lpf = []
    for full_key, item in lpf_combined_results.items():
        s21_20, s11_20, _, _, m_20 = item["metrics_20"]
        s21_40, s11_40, _, _, m_40 = item["metrics_40"]
        _, _, wcm_20 = item["mc_20"]
        _, _, wcm_40 = item["mc_40"]
        if (
            s11_20 <= -15.0
            and s11_40 <= -15.0
            and m_20 >= 0.0
            and m_40 >= 0.0
            and wcm_20 >= 0.0
            and wcm_40 >= 0.0
        ):
            feasible_lpf.append((full_key, item))

    if not feasible_lpf:
        print(
            "  No TX LPF design met all strict 20m+40m S11 & FCC limits simultaneously."
        )
    else:
        for full_key, item in feasible_lpf:
            names = TOPO_SPECS[item["topo"]]["names"]
            base_20 = ", ".join(
                f"{nm}={lbl}" for nm, lbl in zip(names, item["labels_20"])
            )
            build_40 = ", ".join(
                f"{nm}={lbl}" for nm, lbl in zip(names, item["labels_40"])
            )
            print(
                f"\n  * {full_key:<26} [{item['n_reels']} Reel(s), ${3*item['n_reels']:.2f} setup]"
            )
            print(f"    20m JLCPCB Base : {base_20}")
            print(f"    40m Target Build: {build_40}")

    # --- 3. RX Tayloe + Active Audio Chain ---
    print("\n[STAGE 3: RX TAYLOE MIXER + ACTIVE AUDIO FILTER — ALL FEASIBLE DESIGNS]")
    print("  20m JLCPCB Base (Tayloe) : 1x 2.2nF C0G per I/Q branch (4x total)")
    print(
        "  40m Hand Add-On (Tayloe) : +1x 2.2nF C0G parallel per I/Q branch (-> 4.4nF total)"
    )
    print(
        "  Active Audio Filter      : 100% IDENTICAL on 20m & 40m (zero rework required):"
    )

    for d in rx_designs:
        if d.get("ok", False):
            parts = ", ".join(f"{n}={d['lbl'][n]} [{d['cls'][n]}]" for n in RX_OPT)
            print(f"    * {d['name']:<22}: {parts}")


# ==============================================================================
# LAYER 4: UNIFIED SYSTEM PIPELINE (PA -> LPF -> RX)
# ==============================================================================
def run_unified_transceiver(pa_tag: str = "T1") -> None:
    # --------------------------------------------------------------------------
    # STEP 1: CLASS-E EPC2037 GaN PA (20m Design + Exact Harmonic Extraction)
    # --------------------------------------------------------------------------
    section("1. CLASS-E EPC2037 GaN PA (BASELINE + ZVS OPTIMIZATION T1 & T2)")
    vdd, pout, f_pa = PA_SPEC["VDD"], PA_SPEC["POUT_TARGET"], PA_SPEC["F0"]
    pgate = PA_SPEC["QG"] * PA_SPEC["VG"] * f_pa
    pgate_20 = PA_SPEC["QG"] * PA_SPEC["VG"] * F20
    pgate_40 = PA_SPEC["QG"] * PA_SPEC["VG"] * F40
    r_on_25 = PA_SPEC["RON_25"]
    ana = analytical_class_e(vdd, pout, f_pa, PA_SPEC["COSS_TR"])

    print(
        f"Analytical Class-E: Ropt={ana['Ropt']:.4f} Ω | Csh={ana['Csh']*1e12:.2f} pF | L1={ana['L1']*1e9:.2f} nH | L2={ana['L2']*1e9:.2f} nH | C2={ana['C2']*1e12:.2f} pF"
    )
    print(
        f"Thermal check (Fig 8): RON_typ @ 75°C = {r_on_25 * temperature_factor(75.0):.4f} Ω | Integrated Qoss(50V) = {Q_RAW_50*1e12:.2f} pC"
    )

    t1_joint = optimize_class_e_joint_discrete(
        "T1", ana, vdd, pout, F20, F40, r_on_25, pgate_20, pgate_40, max_cap_par=1
    )
    t2_joint = optimize_class_e_joint_discrete(
        "T2", ana, vdd, pout, F20, F40, r_on_25, pgate_20, pgate_40, max_cap_par=1
    )

    pa_res = {
        "A": run_pa_point(
            "const",
            r_on_25,
            ana["L1"],
            ana["L2"],
            ana["C2"],
            ana["Ropt"],
            ana["Cext_const"],
            vdd,
            f_pa,
            pgate,
            "A: Const-C",
        ),
        "X": run_pa_point(
            "nonlin",
            r_on_25,
            ana["L1"],
            ana["L2"],
            ana["C2"],
            ana["Ropt"],
            ana["Cext_const"],
            vdd,
            f_pa,
            pgate,
            "X: Const-C design, real Coss",
        ),
        "E": run_pa_point(
            "nonlin",
            r_on_25,
            ana["L1"],
            ana["L2"],
            ana["C2"],
            ana["Ropt"],
            ana["Csh"] - PA_SPEC["COSS_50"],
            vdd,
            f_pa,
            pgate,
            "E: Naive nonlinear",
        ),
        "T1_20m": t1_joint["20m"],
        "T1_40m": t1_joint["40m"],
        "T2_20m": t2_joint["20m"],
        "T2_40m": t2_joint["40m"],
        # "T1": optimize_class_e_zvs("T1", ana, vdd, pout, f_pa, r_on_25, pgate),
        # "T2": optimize_class_e_zvs("T2", ana, vdd, pout, f_pa, r_on_25, pgate),
    }

    summary_df = pd.DataFrame(
        [
            {
                "Tag": k,
                "L1 (nH)": pa_res[k]["L1_used"] * 1e9,
                "L2 (nH)": pa_res[k]["L2_used"] * 1e9,
                "C2 (pF)": pa_res[k]["C2_used"] * 1e12,
                "Cext (pF)": pa_res[k]["Cext_used"] * 1e12,
                "Ropt (Ω)": pa_res[k]["Ropt_used"],
                "V_on (V)": pa_res[k]["h"]["v_turnon"],
                "i_sw_pk (A)": float(np.max(pa_res[k]["h"]["i_sw"])),
                "P_load (mW)": pa_res[k]["P_load"] * 1e3,
                "P_Ron (mW)": pa_res[k]["P_Ron"] * 1e3,
                "η_sys (%)": pa_res[k]["eta_system"] * 100.0,
            }
            # for k in ["A", "X", "E", "T1", "T2"]
            for k in pa_res
        ]
    )
    print("\n### CLASS-E PA FULL DIAGNOSTICS TABLE\n")
    print(summary_df.round(3).to_string(index=False))
    print_pa_full_report(ana, pa_res, r_on_25, pgate_20, pgate_40)

    match pa_tag:
        case "T1":
            pa_joint_sel = t1_joint
        case "T2":
            pa_joint_sel = t2_joint
        case _:
            raise ValueError(f"Unknown pa_tag: {pa_tag!r} (expected 'T1' or 'T2')")

    ropt_pa = float(pa_res[f"{pa_tag}_20m"]["Ropt_used"])
    pa_harm_20 = extract_pa_harmonics_dbc(pa_res[f"{pa_tag}_20m"], n_harmonics=5)
    pa_harm_40 = extract_pa_harmonics_dbc(pa_res[f"{pa_tag}_40m"], n_harmonics=5)
    pa_harm_wc = tuple(max(h20, h40) for h20, h40 in zip(pa_harm_20, pa_harm_40))

    print(
        f"\nCoupled Class-E ({pa_tag}_20m) Harmonics (2f0..5f0): {[round(h, 2) for h in pa_harm_20]} dBc"
    )
    print(
        f"Coupled Class-E ({pa_tag}_40m) Harmonics (2f0..5f0): {[round(h, 2) for h in pa_harm_40]} dBc"
    )
    print(
        f"Coupled System Reference Impedance R_opt ({pa_tag}) : {ropt_pa:.4f} Ω (handed over to TX LPF & RX Tayloe)"
    )

    plot_pa_waveforms(pa_res, f_pa)
    plot_pa_power(pa_res, f_pa)
    plot_pa_datasheet(pa_res, f_pa)

    # --------------------------------------------------------------------------
    # STEP 2: JOINT 20m -> 40m TX HARMONIC LPF (2x ADD-ON CONVERSION, 2x2 MODES)
    # --------------------------------------------------------------------------
    section(
        "2. JOINT 20m -> 40m TX HARMONIC LPF (2x DOUBLING, 2x2 MODES, BAND-SPECIFIC Q)"
    )
    conv_modes = [
        ("DBL_LC", "MILP Verdopplung L & C (2x L, 2x C)"),
        ("DBL_C", "MILP nur Vervierfachung C (1x L, 4x C)"),
        ("OPT_LC", "MILP Optimierung C + L frei auf 40m"),
        ("OPT_C", "MILP Optimierung nur C frei auf 40m (L fest)"),
    ]
    lpf_modes = [
        ("1R_1C", 1, 1),
        ("1R_2C", 1, 2),
        ("MR_1C", None, 1),
        ("MR_2C", None, 2),
    ]

    ropt_20 = ropt_pa
    ropt_40 = ropt_pa
    lpf_combined_results: dict[str, dict] = {}

    for c_code, c_desc in conv_modes:
        section(f"2. TX MATCH+LPF [{c_code}: {c_desc}] ({ropt_20:.2f}Ω -> 50Ω)")
        mode_results: dict[str, dict] = {}
        for topo in ["JA2GQP", "4Element", "LMatch_WB2CBA"]:
            for tag, m_reels, m_caps in lpf_modes:
                res = solve_joint_lpf_mip(
                    topo,
                    f20=F20,
                    f40=F40,
                    zs_20=ropt_20,
                    zs_40=ropt_40,
                    zl=50.0,
                    harm_dbc=pa_harm_wc,
                    q_l_20=60.0,
                    q_l_40=45.0,
                    max_caps_per_node=m_caps,
                    max_reels=m_reels,
                    conv_mode=c_code,
                )
                if res is None:
                    continue
                key = f"{topo}_{tag}"
                mode_results[key] = res
                lpf_combined_results[f"{c_code}_{key}"] = res

                s21_20, s11_20, _, _, m_20 = res["metrics_20"]
                wc11_20, _, wcm_20 = res["mc_20"]
                s21_40, s11_40, _, _, m_40 = res["metrics_40"]
                wc11_40, _, wcm_40 = res["mc_40"]
                parts_20 = ", ".join(
                    f"{nm}={lbl}"
                    for nm, lbl in zip(TOPO_SPECS[topo]["names"], res["labels_20"])
                )
                parts_40 = ", ".join(
                    f"{nm}={lbl}"
                    for nm, lbl in zip(TOPO_SPECS[topo]["names"], res["labels_40"])
                )

                print(
                    f"\n  {key:<20} | Reels: {res['n_reels']} (${3 * res['n_reels']:.2f})"
                )
                print(f"    20m BOM : {parts_20}")
                print(f"    40m BOM : {parts_40}")
                print(
                    f"    20m (14.10 MHz, Q=60) Nom: S11={s11_20:6.2f} dB, S21={s21_20:6.3f} dB, FCC=+{m_20:4.1f} dB | ±5% MC WC: S11={wc11_20:6.2f} dB, FCC=+{wcm_20:4.1f} dB"
                )
                print(
                    f"    40m ( 7.05 MHz, Q=45) Nom: S11={s11_40:6.2f} dB, S21={s21_40:6.3f} dB, FCC=+{m_40:4.1f} dB | ±5% MC WC: S11={wc11_40:6.2f} dB, FCC=+{wcm_40:4.1f} dB"
                )

        if mode_results:
            plot_joint_lpf_uq(
                mode_results,
                zs_20=ropt_20,
                zs_40=ropt_40,
                zl=50.0,
                harm_dbc=pa_harm_wc,
                q_l_20=60.0,
                q_l_40=45.0,
            )

    # --------------------------------------------------------------------------
    # STEP 3: RX AUDIO CHAIN (ADC SETTLING + RC2 + ARCH A 2x2 + 20m/40m TAYLOE)
    # --------------------------------------------------------------------------
    r_src_audio = 4.0 * (ropt_pa + R_ON_TAYLOE)
    section(
        f"3. RX AUDIO CHAIN (R_rf={ropt_pa:.2f}Ω, R_audio={r_src_audio:.2f}Ω + TAYLOE 2x)"
    )
    err, e0, tau = adc_settle_error(R_KICK, C_KICK)
    settle_a_lsb = float(err * (2**ADC_BITS))
    print(
        f"ADC Settling (R_kick={R_KICK:.0f} Ω, C_kick={C_KICK * 1e9:.1f} nF): charge-share={e0 * 100:.3f}%, tau={tau * 1e9:.1f} ns, residual={settle_a_lsb:.3g} LSB -> {'PASS' if err <= SETTLE_LIMIT else 'FAIL'}"
    )

    rc2_designs = []
    for name, ccls, color in [
        ("RC2 (C0G caps)", "C0G", "seagreen"),
        ("RC2 (C0G1 caps)", "C0G1", "teal"),
    ]:
        v_rc, c_rc, w_rc, ok_rc = design_passive_rc2(ccls, r_src_audio=r_src_audio)
        print(
            f"  {name:<18}: {'FEASIBLE' if ok_rc else 'INFEASIBLE'} | alias={w_rc['alias']:+.2f} dB, droop={w_rc['pas']:.3f} dB, fc_min={w_rc['fc_min']/1e3:.2f} kHz, settle={w_rc['settle']*2**ADC_BITS:.2f} LSB"
        )
        rc2_designs.append(
            {
                "name": name,
                "val": v_rc,
                "cls": c_rc,
                "w": w_rc,
                "ok": ok_rc,
                "color": color,
            }
        )

    cls0_cog1 = {**CLS0, "c_fb": "C0G1", "ca": "C0G1", "cb": "C0G1"}
    rx_designs = [
        {
            "name": "Original design",
            "val": VAL0,
            "cls": CLS0,
            "lbl": LBL0,
            "color": "steelblue",
            "ok": rx_verify(VAL0, CLS0, C_TAYLOE_20M, r_src_audio=r_src_audio)[0]
            <= 1e-9,
        },
        {
            "name": "Original, C0G1",
            "val": VAL0,
            "cls": cls0_cog1,
            "lbl": LBL0,
            "color": "gray",
            "ok": rx_verify(VAL0, cls0_cog1, C_TAYLOE_20M, r_src_audio=r_src_audio)[0]
            <= 1e-9,
        },
    ]

    for d_orig in rx_designs:
        print_rx_bom(
            f"{d_orig['name']} — Component Table (Identical on 20m & 40m)",
            d_orig["val"],
            d_orig["cls"],
            d_orig["lbl"],
        )
        print_rx_derived(
            f"{d_orig['name']} — Derived Quantities",
            d_orig["val"],
            d_orig["cls"],
            r_src_audio=r_src_audio,
        )
        for band_lbl, c_tay in [
            ("20m (2.2 nF Tayloe)", C_TAYLOE_20M),
            ("40m (4.4 nF Tayloe = 2x)", C_TAYLOE_40M),
        ]:
            a, p, fc_min, qlo, qhi = rx_worst_case(
                d_orig["val"], d_orig["cls"], c_tayloe=c_tay, r_src_audio=r_src_audio
            )
            print(
                f"    [{band_lbl}] 128-Corner WC -> Alias: {a:+.2f} dB | Droop: {p:.3f} dB | fc_min: {fc_min/1e3:.2f} kHz | Q: {qlo:.3f}..{qhi:.3f}"
            )
        plot_rx_uq(
            d_orig["val"], d_orig["cls"], d_orig["name"], r_src_audio=r_src_audio
        )

    variants = [
        ("Opt. single (C0G)", 1, "seagreen", {"C0G": 0.0}),
        ("Opt. parallel (C0G)", 2, "olive", {"C0G": 0.0}),
        ("Opt. single (any)", 1, "darkorange", None),
        ("Opt. parallel (any)", 2, "crimson", None),
    ]

    for name, max_par, color, ccls in variants:
        v_opt, c_opt, l_opt, ok_opt = optimize_rx_audio(
            VAL0, CLS0, LBL0, max_par=max_par, cap_classes=ccls, r_src_audio=r_src_audio
        )
        print_rx_bom(
            f"{name} — Component Table (Identical on 20m & 40m)", v_opt, c_opt, l_opt
        )
        print_rx_derived(
            f"{name} — Derived Quantities", v_opt, c_opt, r_src_audio=r_src_audio
        )
        for band_lbl, c_tay in [
            ("20m (2.2 nF Tayloe)", C_TAYLOE_20M),
            ("40m (4.4 nF Tayloe = 2x)", C_TAYLOE_40M),
        ]:
            a, p, fc_min, qlo, qhi = rx_worst_case(
                v_opt, c_opt, c_tayloe=c_tay, r_src_audio=r_src_audio
            )
            print(
                f"    [{band_lbl}] 128-Corner WC -> Alias: {a:+.2f} dB | Droop: {p:.3f} dB | fc_min: {fc_min/1e3:.2f} kHz | Q: {qlo:.3f}..{qhi:.3f}"
            )
        plot_rx_uq(v_opt, c_opt, name, ref_val=VAL0, r_src_audio=r_src_audio)
        rx_designs.append(
            {
                "name": name,
                "val": v_opt,
                "cls": c_opt,
                "lbl": l_opt,
                "color": color,
                "ok": ok_opt,
            }
        )

    for d in rx_designs[2:]:
        plot_ra_sweep(d, r_src_audio=r_src_audio)
    plot_rx_analog_compare(rx_designs, rc2_designs, r_src_audio=r_src_audio)
    print_rx_full_comparison(
        rx_designs, rc2_designs, settle_a_lsb, r_src_audio=r_src_audio
    )
    print_consolidated_bom_summary(
        pa_joint_sel,
        f"{pa_tag} (Ropt = {ropt_pa:.2f} Ω)",
        lpf_combined_results,
        rx_designs,
    )


if __name__ == "__main__":
    run_unified_transceiver()

```

    
    ======================================================================================
    1. CLASS-E EPC2037 GaN PA (BASELINE + ZVS OPTIMIZATION T1 & T2)
    ======================================================================================
    Analytical Class-E: Ropt=8.2229 Ω | Csh=253.83 pF | L1=107.73 nH | L2=280.44 nH | C2=460.83 pF
    Thermal check (Fig 8): RON_typ @ 75°C = 0.5200 Ω | Integrated Qoss(50V) = 467.72 pC
      T1 MILP Joint ZVS+ZDS (20m->40m) in 84.5s | 20m BOM: Cext=270pF+trim(18pF), L1=180nH, L2=220nH, C2=470pF, Ropt=7.95Ω
      T2 MILP Joint ZVS+ZDS (20m->40m) in 205.9s | 20m BOM: Cext=270pF+trim(18pF), L1=120nH, L2=270nH, C2=470pF, Ropt=8.22Ω
    
    ### CLASS-E PA FULL DIAGNOSTICS TABLE
    
       Tag  L1 (nH)  L2 (nH)  C2 (pF)  Cext (pF)  Ropt (Ω)  V_on (V)  i_sw_pk (A)  P_load (mW)  P_Ron (mW)  η_sys (%)
         A  107.735  280.439  460.835     241.83     8.223     0.901        0.906     1615.938     109.082     93.174
         X  107.735  280.439  460.835     241.83     8.223     0.918        2.294     1611.554     110.133     93.192
         E  107.735  280.439  460.835     247.33     8.223     0.918        2.296     1593.226     108.013     93.232
    T1_20m  180.000  220.000  470.000     288.00     7.950     0.071        0.752     1205.826      66.763     94.082
    T1_40m  360.000  440.000  940.000     576.00     7.950    -0.156        0.758     1221.943      68.508     94.318
    T2_20m  120.000  270.000  470.000     288.00     8.223     0.755        1.887     1322.939      78.067     93.865
    T2_40m  240.000  540.000  940.000     576.00     8.223     0.580        1.450     1338.678      79.168     94.119
    
    EPC2037 DATASHEET CARD & COSS MODEL SUMMARY
    ------------------------------------------------------------------------
      V_DS_max = 100.0 V, I_D_max = 1.70 A
      R_on,typ = 400 mΩ (@75°C: 0.5200 Ω), R_on,max = 550 mΩ
      Q_G = 115 pC, Q_GS = 32 pC, Q_GD = 25 pC, R_G = 0.5 Ω
      C_ISS = 14.0 pF, C_RSS = 0.10 pF
      C_OSS(50V) = 6.5 pF (max 10.0 pF), C_OSS(ER) = 9.5 pF, C_OSS(TR) = 12.0 pF
      Integrated Qoss(50V) = 467.72 pC (Datasheet nom 600 pC) | Model Eoss(50V) = 9.4095 nJ
      Analytical Sokal: KR=0.57680, KC=0.18360, KL=1.15249, phi=147.52°
      IDC Sokal = 340.21 mA, IR Sokal = 633.50 mA | P_gate(20m) = 8.107 mW, P_gate(40m) = 4.054 mW
    
    --- Case A (A: Const-C) ---
      I_DC = 355.932 mA | Avg drain V = 4.85001 V | V_pk = 18.450 V | V_min = 0.022 V
      V_turnon = +900.8 mV | i_turnon = +0.2036 A | i_sw_pk = 0.9063 A
      P_load = 1615.94 mW | P_Ron = 109.08 mW | eta_sys = 93.17%
      E_cap_avg = 8.4345 nJ | E_cap_delta = +1.025e-01 nJ
    
    --- Case X (X: Const-C design, real Coss) ---
      I_DC = 354.893 mA | Avg drain V = 4.85090 V | V_pk = 18.467 V | V_min = 0.073 V
      V_turnon = +917.6 mV | i_turnon = +0.2067 A | i_sw_pk = 2.2940 A
      P_load = 1611.55 mW | P_Ron = 110.13 mW | eta_sys = 93.19%
      E_cap_avg = 8.4226 nJ | E_cap_delta = -1.158e-02 nJ
    
    --- Case E (E: Naive nonlinear) ---
      I_DC = 350.686 mA | Avg drain V = 4.85091 V | V_pk = 18.240 V | V_min = 0.067 V
      V_turnon = +918.2 mV | i_turnon = +0.2158 A | i_sw_pk = 2.2956 A
      P_load = 1593.23 mW | P_Ron = 108.01 mW | eta_sys = 93.23%
      E_cap_avg = 8.4872 nJ | E_cap_delta = -1.056e-02 nJ
    
    --- Case T1_20m (T1 ZVS+ZDS [20m 1x]) ---
      I_DC = 262.591 mA | Avg drain V = 4.85016 V | V_pk = 16.785 V | V_min = -0.009 V
      V_turnon = +71.1 mV | i_turnon = +0.3062 A | i_sw_pk = 0.7520 A
      P_load = 1205.83 mW | P_Ron = 66.76 mW | eta_sys = 94.08%
      E_cap_avg = 9.1620 nJ | E_cap_delta = +3.119e-04 nJ
    
    --- Case T1_40m (T1 ZVS+ZDS [40m 2x]) ---
      I_DC = 266.290 mA | Avg drain V = 4.84992 V | V_pk = 16.982 V | V_min = -0.156 V
      V_turnon = -156.3 mV | i_turnon = +0.2975 A | i_sw_pk = 0.7578 A
      P_load = 1221.94 mW | P_Ron = 68.51 mW | eta_sys = 94.32%
      E_cap_avg = 18.1916 nJ | E_cap_delta = -8.887e-04 nJ
    
    --- Case T2_20m (T2 ZVS+ZDS [20m 1x]) ---
      I_DC = 288.927 mA | Avg drain V = 4.85085 V | V_pk = 16.709 V | V_min = 0.012 V
      V_turnon = +754.8 mV | i_turnon = +0.2919 A | i_sw_pk = 1.8869 A
      P_load = 1322.94 mW | P_Ron = 78.07 mW | eta_sys = 93.87%
      E_cap_avg = 9.0371 nJ | E_cap_delta = +3.311e-04 nJ
    
    --- Case T2_40m (T2 ZVS+ZDS [40m 2x]) ---
      I_DC = 292.428 mA | Avg drain V = 4.85066 V | V_pk = 16.895 V | V_min = 0.017 V
      V_turnon = +580.1 mV | i_turnon = +0.2829 A | i_sw_pk = 1.4502 A
      P_load = 1338.68 mW | P_Ron = 79.17 mW | eta_sys = 94.12%
      E_cap_avg = 17.9192 nJ | E_cap_delta = -7.645e-04 nJ
    
    ### ABSOLUTE DELTAS FROM ANALYTICAL 20m SOKAL DESIGN (T_20m - A)
    
       Tag  ΔCext (pF)  ΔL1 (nH)  ΔL2 (nH)  ΔC2 (pF)  ΔRopt (Ω)
    T1_20m       46.17     72.27    -60.44      9.17     -0.273
    T2_20m       46.17     12.27    -10.44      9.17      0.000
    
    Coupled Class-E (T1_20m) Harmonics (2f0..5f0): [-18.35, -34.79, -42.03, -49.75] dBc
    Coupled Class-E (T1_40m) Harmonics (2f0..5f0): [-18.17, -34.49, -41.84, -49.37] dBc
    Coupled System Reference Impedance R_opt (T1) : 7.9498 Ω (handed over to TX LPF & RX Tayloe)



    
![png](output_0_1.png)
    



    
![png](output_0_2.png)
    



    
![png](output_0_3.png)
    


    
    ======================================================================================
    2. JOINT 20m -> 40m TX HARMONIC LPF (2x DOUBLING, 2x2 MODES, BAND-SPECIFIC Q)
    ======================================================================================
    
    ======================================================================================
    2. TX MATCH+LPF [DBL_LC: MILP Verdopplung L & C (2x L, 2x C)] (7.95Ω -> 50Ω)
    ======================================================================================
    
      JA2GQP_1R_1C         | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=270pF, C3=430pF, L2=120nH, C4=120pF, C5=240pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L1=120nH+120nH, C2=270pF || +270pF, C3=430pF || +430pF, L2=120nH+120nH, C4=120pF || +120pF, C5=240pF || +240pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.32 dB, S21=-0.236 dB, FCC=+ 9.1 dB | ±5% MC WC: S11=-16.59 dB, FCC=+-4.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-25.76 dB, S21=-0.307 dB, FCC=+ 7.9 dB | ±5% MC WC: S11=-16.59 dB, FCC=+-4.7 dB
    
      JA2GQP_1R_2C         | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=24pF+240pF, C3=5.1pF+430pF, L2=120nH, C4=56pF+62pF, C5=33pF+200pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L1=120nH+120nH, C2=24pF+240pF || +24pF+240pF, C3=5.1pF+430pF || +5.1pF+430pF, L2=120nH+120nH, C4=56pF+62pF || +56pF+62pF, C5=33pF+200pF || +33pF+200pF
        20m (14.10 MHz, Q=60) Nom: S11=-29.98 dB, S21=-0.226 dB, FCC=+11.6 dB | ±5% MC WC: S11=-17.72 dB, FCC=+-6.1 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-28.94 dB, S21=-0.296 dB, FCC=+ 9.6 dB | ±5% MC WC: S11=-17.72 dB, FCC=+-5.8 dB
    
      JA2GQP_MR_1C         | Reels: 2 ($6.00)
        20m BOM : C1=240pF+trim(18pF), L1=100nH, C2=330pF, C3=1nF, L2=330nH, C4=43pF, C5=330pF
        40m BOM : C1=240pF+trim(18pF) || +240pF, L1=100nH+100nH, C2=330pF || +330pF, C3=1nF || +1nF, L2=330nH+330nH, C4=43pF || +43pF, C5=330pF || +330pF
        20m (14.10 MHz, Q=60) Nom: S11=-37.97 dB, S21=-0.332 dB, FCC=+30.2 dB | ±5% MC WC: S11=-17.09 dB, FCC=+18.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-38.39 dB, S21=-0.437 dB, FCC=+29.6 dB | ±5% MC WC: S11=-17.08 dB, FCC=+18.8 dB
    
      JA2GQP_MR_2C         | Reels: 2 ($6.00)
        20m BOM : C1=120pF+120pF+trim(18pF), L1=100nH, C2=18pF+300pF, C3=7.5pF+1nF, L2=330nH, C4=6.8pF+36pF, C5=24pF+300pF
        40m BOM : C1=120pF+120pF+trim(18pF) || +120pF+120pF, L1=100nH+100nH, C2=18pF+300pF || +18pF+300pF, C3=7.5pF+1nF || +7.5pF+1nF, L2=330nH+330nH, C4=6.8pF+36pF || +6.8pF+36pF, C5=24pF+300pF || +24pF+300pF
        20m (14.10 MHz, Q=60) Nom: S11=-42.63 dB, S21=-0.329 dB, FCC=+37.5 dB | ±5% MC WC: S11=-16.90 dB, FCC=+20.0 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-41.78 dB, S21=-0.434 dB, FCC=+35.0 dB | ±5% MC WC: S11=-16.90 dB, FCC=+19.9 dB
    
      4Element_1R_1C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=220pF || +220pF, C3=560pF || +560pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-27.33 dB, S21=-0.340 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.88 dB, FCC=+ 3.5 dB
    
      4Element_1R_2C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=200pF+330pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=82pF+130pF || +82pF+130pF, C3=200pF+330pF || +200pF+330pF
        20m (14.10 MHz, Q=60) Nom: S11=-30.94 dB, S21=-0.230 dB, FCC=+ 8.7 dB | ±5% MC WC: S11=-16.72 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-30.76 dB, S21=-0.300 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.81 dB, FCC=+ 4.5 dB
    
      4Element_MR_1C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=220pF || +220pF, C3=560pF || +560pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-27.33 dB, S21=-0.340 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.88 dB, FCC=+ 3.5 dB
    
      4Element_MR_2C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=200pF+330pF
        40m BOM : C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=82pF+130pF || +82pF+130pF, C3=200pF+330pF || +200pF+330pF
        20m (14.10 MHz, Q=60) Nom: S11=-30.94 dB, S21=-0.230 dB, FCC=+ 8.7 dB | ±5% MC WC: S11=-16.72 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-30.76 dB, S21=-0.300 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.81 dB, FCC=+ 4.5 dB
    
      LMatch_WB2CBA_1R_1C  | Reels: 1 ($3.00)
        20m BOM : Cm=1.3nF+trim(18pF), Lm=150nH, C1=1.1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=240pF
        40m BOM : Cm=1.3nF+trim(18pF) || +1.3nF, Lm=150nH+150nH, C1=1.1nF || +1.1nF, L2=150nH+150nH, C3=4.7pF || +4.7pF, L4=150nH+150nH, C5=240pF || +240pF
        20m (14.10 MHz, Q=60) Nom: S11=-23.98 dB, S21=-0.357 dB, FCC=+-2.7 dB | ±5% MC WC: S11=-14.07 dB, FCC=+-4.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-24.39 dB, S21=-0.461 dB, FCC=+-2.7 dB | ±5% MC WC: S11=-14.29 dB, FCC=+-4.9 dB
    
      LMatch_WB2CBA_1R_2C  | Reels: 1 ($3.00)
        20m BOM : Cm=82pF+1.2nF+trim(18pF), Lm=150nH, C1=36pF+1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=18pF+240pF
        40m BOM : Cm=82pF+1.2nF+trim(18pF) || +82pF+1.2nF, Lm=150nH+150nH, C1=36pF+1nF || +36pF+1nF, L2=150nH+150nH, C3=4.7pF || +4.7pF, L4=150nH+150nH, C5=18pF+240pF || +18pF+240pF
        20m (14.10 MHz, Q=60) Nom: S11=-23.30 dB, S21=-0.331 dB, FCC=+-2.9 dB | ±5% MC WC: S11=-15.62 dB, FCC=+-5.2 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-23.25 dB, S21=-0.427 dB, FCC=+-2.9 dB | ±5% MC WC: S11=-15.76 dB, FCC=+-5.2 dB
    
      LMatch_WB2CBA_MR_1C  | Reels: 3 ($9.00)
        20m BOM : Cm=1.3nF+trim(18pF), Lm=100nH, C1=1.8nF, L2=270nH, C3=510pF, L4=180nH, C5=47pF
        40m BOM : Cm=1.3nF+trim(18pF) || +1.3nF, Lm=100nH+100nH, C1=1.8nF || +1.8nF, L2=270nH+270nH, C3=510pF || +510pF, L4=180nH+180nH, C5=47pF || +47pF
        20m (14.10 MHz, Q=60) Nom: S11=-20.68 dB, S21=-0.425 dB, FCC=+ 3.8 dB | ±5% MC WC: S11=-12.91 dB, FCC=+ 1.3 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-20.58 dB, S21=-0.547 dB, FCC=+ 3.8 dB | ±5% MC WC: S11=-13.02 dB, FCC=+ 1.3 dB
    
      LMatch_WB2CBA_MR_2C  | Reels: 3 ($9.00)
        20m BOM : Cm=22pF+1nF+trim(18pF), Lm=120nH, C1=43pF+1.5nF, L2=330nH, C3=200pF+300pF, L4=270nH, C5=27pF
        40m BOM : Cm=22pF+1nF+trim(18pF) || +22pF+1nF, Lm=120nH+120nH, C1=43pF+1.5nF || +43pF+1.5nF, L2=330nH+330nH, C3=200pF+300pF || +200pF+300pF, L4=270nH+270nH, C5=27pF || +27pF
        20m (14.10 MHz, Q=60) Nom: S11=-24.68 dB, S21=-0.427 dB, FCC=+ 6.2 dB | ±5% MC WC: S11=-13.12 dB, FCC=+ 3.7 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-24.93 dB, S21=-0.556 dB, FCC=+ 6.2 dB | ±5% MC WC: S11=-13.33 dB, FCC=+ 3.7 dB



    
![png](output_0_5.png)
    



    
![png](output_0_6.png)
    


    
    ======================================================================================
    2. TX MATCH+LPF [DBL_C: MILP nur Vervierfachung C (1x L, 4x C)] (7.95Ω -> 50Ω)
    ======================================================================================
    
      JA2GQP_1R_1C         | Reels: 1 ($3.00)
        20m BOM : C1=1nF+trim(18pF), L1=220nH, C2=130pF, C3=560pF, L2=220nH, C4=62pF, C5=75pF
        40m BOM : C1=1nF+trim(18pF) || +3nF, L1=220nH, C2=130pF || +390pF, C3=560pF || +1.6nF, L2=220nH, C4=62pF || +180pF, C5=75pF || +220pF
        20m (14.10 MHz, Q=60) Nom: S11= -9.14 dB, S21=-0.896 dB, FCC=+14.8 dB | ±5% MC WC: S11= -6.55 dB, FCC=+ 7.7 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-26.57 dB, S21=-0.723 dB, FCC=+13.9 dB | ±5% MC WC: S11=-11.28 dB, FCC=+ 7.3 dB
    
      JA2GQP_1R_2C         | Reels: 1 ($3.00)
        20m BOM : C1=1nF+1nF+trim(18pF), L1=120nH, C2=24pF+220pF, C3=36pF+910pF, L2=120nH, C4=56pF+62pF, C5=120pF+130pF
        40m BOM : C1=1nF+1nF+trim(18pF) || +4.3nF, L1=120nH, C2=24pF+220pF || +750pF, C3=36pF+910pF || +2.7nF, L2=120nH, C4=56pF+62pF || +360pF, C5=120pF+130pF || +750pF
        20m (14.10 MHz, Q=60) Nom: S11=-12.79 dB, S21=-0.733 dB, FCC=+17.3 dB | ±5% MC WC: S11= -6.97 dB, FCC=+ 8.1 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-11.09 dB, S21=-1.477 dB, FCC=+19.4 dB | ±5% MC WC: S11= -5.96 dB, FCC=+ 8.2 dB
    
      JA2GQP_MR_1C         | Reels: 2 ($6.00)
        20m BOM : C1=510pF+trim(18pF), L1=150nH, C2=220pF, C3=910pF, L2=470nH, C4=30pF, C5=270pF
        40m BOM : C1=510pF+trim(18pF) || +1.5nF, L1=150nH, C2=220pF || +680pF, C3=910pF || +2.7nF, L2=470nH, C4=30pF || +91pF, C5=270pF || +820pF
        20m (14.10 MHz, Q=60) Nom: S11=-25.86 dB, S21=-0.500 dB, FCC=+36.1 dB | ±5% MC WC: S11=-13.09 dB, FCC=+24.7 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-23.85 dB, S21=-0.608 dB, FCC=+34.5 dB | ±5% MC WC: S11=-12.85 dB, FCC=+25.5 dB
    
      JA2GQP_MR_2C         | Reels: 2 ($6.00)
        20m BOM : C1=24pF+510pF+trim(18pF), L1=150nH, C2=82pF+130pF, C3=910pF, L2=470nH, C4=8.2pF+22pF, C5=68pF+220pF
        40m BOM : C1=24pF+510pF+trim(18pF) || +1.6nF, L1=150nH, C2=82pF+130pF || +620pF, C3=910pF || +2.7nF, L2=470nH, C4=8.2pF+22pF || +91pF, C5=68pF+220pF || +820pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.60 dB, S21=-0.498 dB, FCC=+44.3 dB | ±5% MC WC: S11=-12.85 dB, FCC=+26.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-19.32 dB, S21=-0.633 dB, FCC=+41.3 dB | ±5% MC WC: S11=-11.25 dB, FCC=+27.3 dB
    
      4Element_1R_1C       | Reels: 1 ($3.00)
        20m BOM : C1=1nF+trim(18pF), L2=180nH, C2=160pF, C3=560pF
        40m BOM : C1=1nF+trim(18pF) || +3nF, L2=180nH, C2=160pF || +470pF, C3=560pF || +1.6nF
        20m (14.10 MHz, Q=60) Nom: S11=-15.05 dB, S21=-0.419 dB, FCC=+12.1 dB | ±5% MC WC: S11=-10.73 dB, FCC=+ 5.4 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.85 dB, S21=-2.311 dB, FCC=+15.0 dB | ±5% MC WC: S11= -3.44 dB, FCC=+ 8.7 dB
    
      4Element_1R_2C       | Reels: 1 ($3.00)
        20m BOM : C1=910pF+1nF+trim(18pF), L2=150nH, C2=51pF+130pF, C3=470pF+470pF
        40m BOM : C1=910pF+1nF+trim(18pF) || +4.3nF, L2=150nH, C2=51pF+130pF || +560pF, C3=470pF+470pF || +2.7nF
        20m (14.10 MHz, Q=60) Nom: S11=-10.85 dB, S21=-0.934 dB, FCC=+15.5 dB | ±5% MC WC: S11= -5.40 dB, FCC=+10.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-10.75 dB, S21=-1.618 dB, FCC=+20.2 dB | ±5% MC WC: S11= -5.39 dB, FCC=+14.6 dB
    
      4Element_MR_1C       | Reels: 1 ($3.00)
        20m BOM : C1=1nF+trim(18pF), L2=180nH, C2=160pF, C3=560pF
        40m BOM : C1=1nF+trim(18pF) || +3nF, L2=180nH, C2=160pF || +470pF, C3=560pF || +1.6nF
        20m (14.10 MHz, Q=60) Nom: S11=-15.05 dB, S21=-0.419 dB, FCC=+12.1 dB | ±5% MC WC: S11=-10.73 dB, FCC=+ 5.4 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.85 dB, S21=-2.311 dB, FCC=+15.0 dB | ±5% MC WC: S11= -3.44 dB, FCC=+ 8.7 dB
    
      4Element_MR_2C       | Reels: 1 ($3.00)
        20m BOM : C1=910pF+1nF+trim(18pF), L2=150nH, C2=51pF+130pF, C3=470pF+470pF
        40m BOM : C1=910pF+1nF+trim(18pF) || +4.3nF, L2=150nH, C2=51pF+130pF || +560pF, C3=470pF+470pF || +2.7nF
        20m (14.10 MHz, Q=60) Nom: S11=-10.85 dB, S21=-0.934 dB, FCC=+15.5 dB | ±5% MC WC: S11= -5.40 dB, FCC=+10.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-10.75 dB, S21=-1.618 dB, FCC=+20.2 dB | ±5% MC WC: S11= -5.39 dB, FCC=+14.6 dB
    
      LMatch_WB2CBA_1R_1C  | Reels: 1 ($3.00)
        20m BOM : Cm=910pF+trim(18pF), Lm=220nH, C1=750pF, L2=220nH, C3=18pF, L4=220nH, C5=200pF
        40m BOM : Cm=910pF+trim(18pF) || +2.7nF, Lm=220nH, C1=750pF || +2.2nF, L2=220nH, C3=18pF || +56pF, L4=220nH, C5=200pF || +620pF
        20m (14.10 MHz, Q=60) Nom: S11=-13.08 dB, S21=-0.510 dB, FCC=+-1.5 dB | ±5% MC WC: S11=-10.42 dB, FCC=+-3.6 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-17.52 dB, S21=-0.557 dB, FCC=+ 0.6 dB | ±5% MC WC: S11=-12.02 dB, FCC=+-1.9 dB
    
      LMatch_WB2CBA_1R_2C  | Reels: 1 ($3.00)
        20m BOM : Cm=620pF+680pF+trim(18pF), Lm=180nH, C1=510pF+510pF, L2=180nH, C3=4.7pF, L4=180nH, C5=91pF+130pF
        40m BOM : Cm=620pF+680pF+trim(18pF) || +3.9nF, Lm=180nH, C1=510pF+510pF || +3nF, L2=180nH, C3=4.7pF || +15pF, L4=180nH, C5=91pF+130pF || +680pF
        20m (14.10 MHz, Q=60) Nom: S11=-14.25 dB, S21=-0.521 dB, FCC=+ 0.6 dB | ±5% MC WC: S11= -9.61 dB, FCC=+-1.6 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-16.28 dB, S21=-0.730 dB, FCC=+ 3.6 dB | ±5% MC WC: S11= -9.65 dB, FCC=+ 1.1 dB
    
      LMatch_WB2CBA_MR_1C  | Reels: 3 ($9.00)
        20m BOM : Cm=620pF+trim(18pF), Lm=220nH, C1=910pF, L2=470nH, C3=150pF, L4=39nH, C5=130pF
        40m BOM : Cm=620pF+trim(18pF) || +1.8nF, Lm=220nH, C1=910pF || +2.7nF, L2=470nH, C3=150pF || +430pF, L4=39nH, C5=130pF || +390pF
        20m (14.10 MHz, Q=60) Nom: S11=-15.59 dB, S21=-0.500 dB, FCC=+ 2.6 dB | ±5% MC WC: S11= -9.90 dB, FCC=+ 0.6 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-19.77 dB, S21=-0.540 dB, FCC=+ 4.0 dB | ±5% MC WC: S11=-12.38 dB, FCC=+ 1.8 dB
    
      LMatch_WB2CBA_MR_2C  | Reels: 3 ($9.00)
        20m BOM : Cm=75pF+560pF+trim(18pF), Lm=220nH, C1=300pF+470pF, L2=560nH, C3=22pF+270pF, L4=330nH, C5=6.2pF
        40m BOM : Cm=75pF+560pF+trim(18pF) || +2nF, Lm=220nH, C1=300pF+470pF || +2.4nF, L2=560nH, C3=22pF+270pF || +910pF, L4=330nH, C5=6.2pF || +18pF
        20m (14.10 MHz, Q=60) Nom: S11=-18.80 dB, S21=-0.435 dB, FCC=+ 4.3 dB | ±5% MC WC: S11=-12.10 dB, FCC=+ 1.8 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-11.51 dB, S21=-0.841 dB, FCC=+ 6.7 dB | ±5% MC WC: S11= -7.82 dB, FCC=+ 4.2 dB



    
![png](output_0_8.png)
    



    
![png](output_0_9.png)
    


    
    ======================================================================================
    2. TX MATCH+LPF [OPT_LC: MILP Optimierung C + L frei auf 40m] (7.95Ω -> 50Ω)
    ======================================================================================
    
      JA2GQP_1R_1C         | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=270pF, C3=430pF, L2=120nH, C4=120pF, C5=240pF
        40m BOM : C1=4.7pF+trim(18pF), L1=220nH, C2=560pF, C3=910pF, L2=220nH, C4=270pF, C5=470pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.32 dB, S21=-0.236 dB, FCC=+ 9.1 dB | ±5% MC WC: S11=-16.59 dB, FCC=+-4.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-26.46 dB, S21=-0.293 dB, FCC=+ 3.9 dB | ±5% MC WC: S11=-17.68 dB, FCC=+-8.3 dB
    
      JA2GQP_1R_2C         | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=24pF+240pF, C3=43pF+390pF, L2=120nH, C4=56pF+62pF, C5=82pF+150pF
        40m BOM : C1=4.7pF+trim(18pF), L1=220nH, C2=220pF+360pF, C3=62pF+820pF, L2=220nH, C4=18pF+240pF, C5=470pF
        20m (14.10 MHz, Q=60) Nom: S11=-30.56 dB, S21=-0.224 dB, FCC=+11.5 dB | ±5% MC WC: S11=-17.99 dB, FCC=+-6.1 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-27.64 dB, S21=-0.288 dB, FCC=+ 8.3 dB | ±5% MC WC: S11=-17.53 dB, FCC=+-6.0 dB
    
      JA2GQP_MR_1C         | Reels: 4 ($12.00)
        20m BOM : C1=75pF+trim(18pF), L1=100nH, C2=330pF, C3=820pF, L2=330nH, C4=43pF, C5=240pF
        40m BOM : C1=680pF+trim(18pF), L1=270nH, C2=470pF, C3=1.8nF, L2=820nH, C4=68pF, C5=510pF
        20m (14.10 MHz, Q=60) Nom: S11=-28.82 dB, S21=-0.286 dB, FCC=+25.1 dB | ±5% MC WC: S11=-17.22 dB, FCC=+13.8 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-27.26 dB, S21=-0.518 dB, FCC=+37.0 dB | ±5% MC WC: S11=-13.93 dB, FCC=+22.0 dB
    
      JA2GQP_MR_2C         | Reels: 4 ($12.00)
        20m BOM : C1=7.5pF+56pF+trim(18pF), L1=100nH, C2=18pF+300pF, C3=47pF+750pF, L2=330nH, C4=6.8pF+36pF, C5=240pF
        40m BOM : C1=6.8pF+680pF+trim(18pF), L1=270nH, C2=82pF+390pF, C3=470pF+1.3nF, L2=820nH, C4=33pF+36pF, C5=75pF+430pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.02 dB, S21=-0.280 dB, FCC=+32.2 dB | ±5% MC WC: S11=-16.13 dB, FCC=+14.7 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-31.35 dB, S21=-0.505 dB, FCC=+37.0 dB | ±5% MC WC: S11=-14.62 dB, FCC=+22.2 dB
    
      4Element_1R_1C       | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF), L2=330nH, C2=360pF, C3=1nF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-23.68 dB, S21=-0.302 dB, FCC=+ 8.5 dB | ±5% MC WC: S11=-16.77 dB, FCC=+ 0.5 dB
    
      4Element_1R_2C       | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m BOM : C1=4.7pF+trim(18pF), L2=330nH, C2=56pF+330pF, C3=470pF+510pF
        20m (14.10 MHz, Q=60) Nom: S11=-29.88 dB, S21=-0.230 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.54 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-21.47 dB, S21=-0.318 dB, FCC=+ 8.7 dB | ±5% MC WC: S11=-15.79 dB, FCC=+ 4.8 dB
    
      4Element_MR_1C       | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF), L2=330nH, C2=360pF, C3=1nF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-23.68 dB, S21=-0.302 dB, FCC=+ 8.5 dB | ±5% MC WC: S11=-16.77 dB, FCC=+ 0.5 dB
    
      4Element_MR_2C       | Reels: 2 ($6.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m BOM : C1=4.7pF+trim(18pF), L2=330nH, C2=56pF+330pF, C3=470pF+510pF
        20m (14.10 MHz, Q=60) Nom: S11=-29.88 dB, S21=-0.230 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.54 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-21.47 dB, S21=-0.318 dB, FCC=+ 8.7 dB | ±5% MC WC: S11=-15.79 dB, FCC=+ 4.8 dB
    
      LMatch_WB2CBA_1R_1C  | Reels: 2 ($6.00)
        20m BOM : Cm=1.3nF+trim(18pF), Lm=150nH, C1=1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=270pF
        40m BOM : Cm=2.4nF+trim(18pF), Lm=270nH, C1=2.2nF, L2=270nH, C3=4.7pF, L4=270nH, C5=620pF
        20m (14.10 MHz, Q=60) Nom: S11=-18.89 dB, S21=-0.352 dB, FCC=+-2.9 dB | ±5% MC WC: S11=-13.95 dB, FCC=+-5.2 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-28.52 dB, S21=-0.406 dB, FCC=+-3.7 dB | ±5% MC WC: S11=-16.94 dB, FCC=+-6.1 dB
    
      LMatch_WB2CBA_1R_2C  | Reels: 2 ($6.00)
        20m BOM : Cm=7.5pF+1.3nF+trim(18pF), Lm=150nH, C1=27pF+1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=24pF+240pF
        40m BOM : Cm=330pF+2.2nF+trim(18pF), Lm=270nH, C1=24pF+2.2nF, L2=270nH, C3=4.7pF, L4=270nH, C5=620pF
        20m (14.10 MHz, Q=60) Nom: S11=-20.82 dB, S21=-0.342 dB, FCC=+-2.7 dB | ±5% MC WC: S11=-14.89 dB, FCC=+-5.0 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-26.52 dB, S21=-0.414 dB, FCC=+-3.2 dB | ±5% MC WC: S11=-16.74 dB, FCC=+-5.6 dB
    
      LMatch_WB2CBA_MR_1C  | Reels: 5 ($15.00)
        20m BOM : Cm=910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=39pF
        40m BOM : Cm=1.8nF+trim(18pF), Lm=220nH, C1=3nF, L2=560nH, C3=1.1nF, L4=560nH, C5=51pF
        20m (14.10 MHz, Q=60) Nom: S11=-27.15 dB, S21=-0.405 dB, FCC=+ 2.8 dB | ±5% MC WC: S11=-13.55 dB, FCC=+ 0.2 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-26.30 dB, S21=-0.518 dB, FCC=+ 3.3 dB | ±5% MC WC: S11=-14.60 dB, FCC=+ 0.7 dB
    
      LMatch_WB2CBA_MR_2C  | Reels: 6 ($18.00)
        20m BOM : Cm=62pF+910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=8.2pF+33pF
        40m BOM : Cm=33pF+1.5nF+trim(18pF), Lm=220nH, C1=27pF+3nF, L2=560nH, C3=33pF+1.1nF, L4=680nH, C5=36pF+68pF
        20m (14.10 MHz, Q=60) Nom: S11=-33.47 dB, S21=-0.398 dB, FCC=+ 3.1 dB | ±5% MC WC: S11=-14.57 dB, FCC=+ 0.4 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-26.61 dB, S21=-0.531 dB, FCC=+ 3.0 dB | ±5% MC WC: S11=-13.75 dB, FCC=+ 0.4 dB



    
![png](output_0_11.png)
    



    
![png](output_0_12.png)
    


    
    ======================================================================================
    2. TX MATCH+LPF [OPT_C: MILP Optimierung nur C frei auf 40m (L fest)] (7.95Ω -> 50Ω)
    ======================================================================================
    
      JA2GQP_1R_1C         | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=270pF, C3=430pF, L2=120nH, C4=120pF, C5=240pF
        40m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=1.2nF, C3=4.3nF, L2=120nH, C4=510pF, C5=2.2nF
        20m (14.10 MHz, Q=60) Nom: S11=-26.32 dB, S21=-0.236 dB, FCC=+ 9.1 dB | ±5% MC WC: S11=-16.59 dB, FCC=+-4.9 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -1.45 dB, S21=-7.049 dB, FCC=+15.3 dB | ±5% MC WC: S11= -1.29 dB, FCC=+ 8.8 dB
    
      JA2GQP_1R_2C         | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=24pF+240pF, C3=43pF+390pF, L2=120nH, C4=56pF+62pF, C5=82pF+150pF
        40m BOM : C1=4.7pF+trim(18pF), L1=120nH, C2=43pF+1.2nF, C3=4.3nF+4.3nF, L2=120nH, C4=43pF+510pF, C5=91pF+2nF
        20m (14.10 MHz, Q=60) Nom: S11=-30.56 dB, S21=-0.224 dB, FCC=+11.5 dB | ±5% MC WC: S11=-17.99 dB, FCC=+-6.1 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -0.79 dB, S21=-10.518 dB, FCC=+20.2 dB | ±5% MC WC: S11= -0.71 dB, FCC=+14.5 dB
    
      JA2GQP_MR_1C         | Reels: 2 ($6.00)
        20m BOM : C1=75pF+trim(18pF), L1=100nH, C2=330pF, C3=820pF, L2=330nH, C4=43pF, C5=240pF
        40m BOM : C1=4.7pF+trim(18pF), L1=100nH, C2=1.3nF, C3=820pF, L2=330nH, C4=180pF, C5=910pF
        20m (14.10 MHz, Q=60) Nom: S11=-28.82 dB, S21=-0.286 dB, FCC=+25.1 dB | ±5% MC WC: S11=-17.22 dB, FCC=+13.8 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-28.30 dB, S21=-0.299 dB, FCC=+12.1 dB | ±5% MC WC: S11=-17.65 dB, FCC=+-0.8 dB
    
      JA2GQP_MR_2C         | Reels: 2 ($6.00)
        20m BOM : C1=7.5pF+56pF+trim(18pF), L1=100nH, C2=18pF+300pF, C3=47pF+750pF, L2=330nH, C4=6.8pF+36pF, C5=240pF
        40m BOM : C1=4.7pF+trim(18pF), L1=100nH, C2=75pF+1.2nF, C3=36pF+750pF, L2=330nH, C4=62pF+110pF, C5=56pF+820pF
        20m (14.10 MHz, Q=60) Nom: S11=-26.02 dB, S21=-0.280 dB, FCC=+32.2 dB | ±5% MC WC: S11=-16.13 dB, FCC=+14.7 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-37.49 dB, S21=-0.277 dB, FCC=+12.9 dB | ±5% MC WC: S11=-18.80 dB, FCC=+-4.8 dB
    
      4Element_1R_1C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=910pF, C3=2.4nF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.45 dB, S21=-2.600 dB, FCC=+ 9.0 dB | ±5% MC WC: S11= -4.02 dB, FCC=+ 2.0 dB
    
      4Element_1R_2C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+910pF, C3=180pF+2.2nF
        20m (14.10 MHz, Q=60) Nom: S11=-29.88 dB, S21=-0.230 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.54 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.56 dB, S21=-2.572 dB, FCC=+ 2.2 dB | ±5% MC WC: S11= -4.12 dB, FCC=+-1.8 dB
    
      4Element_MR_1C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=910pF, C3=2.4nF
        20m (14.10 MHz, Q=60) Nom: S11=-26.55 dB, S21=-0.264 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-14.77 dB, FCC=+ 3.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.45 dB, S21=-2.600 dB, FCC=+ 9.0 dB | ±5% MC WC: S11= -4.02 dB, FCC=+ 2.0 dB
    
      4Element_MR_2C       | Reels: 1 ($3.00)
        20m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m BOM : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+910pF, C3=180pF+2.2nF
        20m (14.10 MHz, Q=60) Nom: S11=-29.88 dB, S21=-0.230 dB, FCC=+ 8.6 dB | ±5% MC WC: S11=-16.54 dB, FCC=+ 4.5 dB
        40m ( 7.05 MHz, Q=45) Nom: S11= -4.56 dB, S21=-2.572 dB, FCC=+ 2.2 dB | ±5% MC WC: S11= -4.12 dB, FCC=+-1.8 dB
    
      LMatch_WB2CBA_1R_1C  | Reels: 1 ($3.00)
        20m BOM : Cm=1.3nF+trim(18pF), Lm=150nH, C1=1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=270pF
        40m BOM : Cm=3.6nF+trim(18pF), Lm=150nH, C1=4.7pF, L2=150nH, C3=1.6nF, L4=150nH, C5=560pF
        20m (14.10 MHz, Q=60) Nom: S11=-18.89 dB, S21=-0.352 dB, FCC=+-2.9 dB | ±5% MC WC: S11=-13.95 dB, FCC=+-5.2 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-29.98 dB, S21=-0.493 dB, FCC=+-7.9 dB | ±5% MC WC: S11=-14.26 dB, FCC=+-9.3 dB
    
      LMatch_WB2CBA_1R_2C  | Reels: 1 ($3.00)
        20m BOM : Cm=7.5pF+1.3nF+trim(18pF), Lm=150nH, C1=27pF+1nF, L2=150nH, C3=4.7pF, L4=150nH, C5=24pF+240pF
        40m BOM : Cm=43pF+3.6nF+trim(18pF), Lm=150nH, C1=4.7pF, L2=150nH, C3=13pF+1.6nF, L4=150nH, C5=270pF+270pF
        20m (14.10 MHz, Q=60) Nom: S11=-20.82 dB, S21=-0.342 dB, FCC=+-2.7 dB | ±5% MC WC: S11=-14.89 dB, FCC=+-5.0 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-27.21 dB, S21=-0.496 dB, FCC=+-7.7 dB | ±5% MC WC: S11=-13.75 dB, FCC=+-9.1 dB
    
      LMatch_WB2CBA_MR_1C  | Reels: 3 ($9.00)
        20m BOM : Cm=910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=39pF
        40m BOM : Cm=3nF+trim(18pF), Lm=100nH, C1=4.7pF, L2=270nH, C3=1.6nF, L4=330nH, C5=150pF
        20m (14.10 MHz, Q=60) Nom: S11=-27.15 dB, S21=-0.405 dB, FCC=+ 2.8 dB | ±5% MC WC: S11=-13.55 dB, FCC=+ 0.2 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-20.65 dB, S21=-0.467 dB, FCC=+-6.6 dB | ±5% MC WC: S11=-12.44 dB, FCC=+-8.1 dB
    
      LMatch_WB2CBA_MR_2C  | Reels: 3 ($9.00)
        20m BOM : Cm=62pF+910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=8.2pF+33pF
        40m BOM : Cm=1.3nF+1.6nF+trim(18pF), Lm=100nH, C1=4.7pF, L2=270nH, C3=680pF+1nF, L4=330nH, C5=36pF+56pF
        20m (14.10 MHz, Q=60) Nom: S11=-33.47 dB, S21=-0.398 dB, FCC=+ 3.1 dB | ±5% MC WC: S11=-14.57 dB, FCC=+ 0.4 dB
        40m ( 7.05 MHz, Q=45) Nom: S11=-25.92 dB, S21=-0.476 dB, FCC=+-5.5 dB | ±5% MC WC: S11=-13.31 dB, FCC=+-6.9 dB



    
![png](output_0_14.png)
    



    
![png](output_0_15.png)
    


    
    ======================================================================================
    3. RX AUDIO CHAIN (R_rf=7.95Ω, R_audio=51.80Ω + TAYLOE 2x)
    ======================================================================================
    ADC Settling (R_kick=100 Ω, C_kick=1.0 nF): charge-share=0.596%, tau=100.6 ns, residual=1.46e-10 LSB -> PASS
      RC2 (C0G caps)    : INFEASIBLE | alias=-66.51 dB, droop=4.324 dB, fc_min=2.20 kHz, settle=2.15 LSB
      RC2 (C0G1 caps)   : INFEASIBLE | alias=-66.53 dB, droop=3.883 dB, fc_min=2.37 kHz, settle=1.91 LSB
    
    Original design — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        220pF 220pF          C0G        5%     209pF     231pF         2.2e-10
      ra    Sallen-Key R (both resistors)           10k 10k            R1pct      1%      9.9k     10.1k           10000
      ca    Sallen-Key Ca (feedback cap)          1.5nF 1.5nF          C0G        5%   1.425nF   1.575nF         1.5e-09
      cb    Sallen-Key Cb (cap to ground)         750pF 750pF          C0G        5%   712.5pF   787.5pF         7.5e-10
      ----------------------------------------------------------------------------------------------------
    
    Original design — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 14.469 kHz
      Sallen-Key f0 / Q                    : 15.005 kHz / 0.7071
      cascade -3 dB corner (nominal)       : 11.040 kHz | worst-corner fc_min: 10.113 kHz
      Q over tolerance corners             : 0.6726 .. 0.7434 (allowed 0.6..0.85)
      nominal alias / passband dev         : -66.62 dB (limit -66.52) / 0.322 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -65.03 dB | Droop: 0.370 dB | fc_min: 10.11 kHz | Q: 0.673..0.743
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -65.26 dB | Droop: 0.370 dB | fc_min: 10.11 kHz | Q: 0.673..0.743
    
    Original design - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -66.61 dB      +0.09 dB      PASS
    1 sigma      84.13450%       -66.31 dB      -0.20 dB      FAIL
    2 sigma      97.72500%       -66.04 dB      -0.48 dB      FAIL
    3 sigma      99.86500%       -65.76 dB      -0.76 dB      FAIL
    4 sigma      99.99683%       -65.54 dB      -0.97 dB      FAIL
    Maximum     100.00000%       -65.53 dB      -0.99 dB      FAIL
    --------------------------------------------------------------------



    
![png](output_0_17.png)
    


    
    Original, C0G1 — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        220pF 220pF          C0G1       1%   217.8pF   222.2pF         2.2e-10
      ra    Sallen-Key R (both resistors)           10k 10k            R1pct      1%      9.9k     10.1k           10000
      ca    Sallen-Key Ca (feedback cap)          1.5nF 1.5nF          C0G1       1%   1.485nF   1.515nF         1.5e-09
      cb    Sallen-Key Cb (cap to ground)         750pF 750pF          C0G1       1%   742.5pF   757.5pF         7.5e-10
      ----------------------------------------------------------------------------------------------------
    
    Original, C0G1 — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 14.469 kHz
      Sallen-Key f0 / Q                    : 15.005 kHz / 0.7071
      cascade -3 dB corner (nominal)       : 11.040 kHz | worst-corner fc_min: 10.724 kHz
      Q over tolerance corners             : 0.7001 .. 0.7142 (allowed 0.6..0.85)
      nominal alias / passband dev         : -66.62 dB (limit -66.52) / 0.322 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -66.10 dB | Droop: 0.334 dB | fc_min: 10.72 kHz | Q: 0.700..0.714
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -66.33 dB | Droop: 0.334 dB | fc_min: 10.72 kHz | Q: 0.700..0.714
    
    Original, C0G1 - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -66.62 dB      +0.10 dB      PASS
    1 sigma      84.13450%       -66.53 dB      +0.01 dB      PASS
    2 sigma      97.72500%       -66.44 dB      -0.08 dB      FAIL
    3 sigma      99.86500%       -66.36 dB      -0.16 dB      FAIL
    4 sigma      99.99683%       -66.29 dB      -0.23 dB      FAIL
    Maximum     100.00000%       -66.28 dB      -0.23 dB      FAIL
    --------------------------------------------------------------------



    
![png](output_0_19.png)
    


    
    Opt. single (C0G) — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        200pF 200pF          C0G        5%     190pF     210pF           2e-10
      ra    Sallen-Key R (both resistors)         12.4k 12.4k          R1pct      1%    12.28k    12.52k           12400
      ca    Sallen-Key Ca (feedback cap)          1.6nF 1.6nF          C0G        5%    1.52nF    1.68nF         1.6e-09
      cb    Sallen-Key Cb (cap to ground)         620pF 620pF          C0G        5%     589pF     651pF         6.2e-10
      ----------------------------------------------------------------------------------------------------
    
    Opt. single (C0G) — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 15.915 kHz
      Sallen-Key f0 / Q                    : 12.887 kHz / 0.8032
      cascade -3 dB corner (nominal)       : 11.778 kHz | worst-corner fc_min: 10.936 kHz
      Q over tolerance corners             : 0.7640 .. 0.8444 (allowed 0.6..0.85)
      nominal alias / passband dev         : -68.44 dB (limit -66.52) / 0.214 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -66.84 dB | Droop: 0.261 dB | fc_min: 10.94 kHz | Q: 0.764..0.844
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -67.07 dB | Droop: 0.261 dB | fc_min: 10.94 kHz | Q: 0.764..0.844
    
    Opt. single (C0G) - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -68.42 dB      +1.90 dB      PASS
    1 sigma      84.13450%       -68.13 dB      +1.61 dB      PASS
    2 sigma      97.72500%       -67.85 dB      +1.33 dB      PASS
    3 sigma      99.86500%       -67.57 dB      +1.05 dB      PASS
    4 sigma      99.99683%       -67.36 dB      +0.84 dB      PASS
    Maximum     100.00000%       -67.34 dB      +0.83 dB      PASS
    --------------------------------------------------------------------



    
![png](output_0_21.png)
    


    
    Opt. parallel (C0G) — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        192pF 62pF+130pF     C0G        5%   182.4pF   201.6pF        1.92e-10
      ra    Sallen-Key R (both resistors)         12.1k 12.1k          R1pct      1%    11.98k    12.22k           12100
      ca    Sallen-Key Ca (feedback cap)         1.66nF 750pF+910pF    C0G        5%   1.577nF   1.743nF        1.66e-09
      cb    Sallen-Key Cb (cap to ground)         635pF 75pF+560pF     C0G        5%   603.2pF   666.8pF        6.35e-10
      ----------------------------------------------------------------------------------------------------
    
    Opt. parallel (C0G) — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 16.579 kHz
      Sallen-Key f0 / Q                    : 12.811 kHz / 0.8084
      cascade -3 dB corner (nominal)       : 11.943 kHz | worst-corner fc_min: 11.111 kHz
      Q over tolerance corners             : 0.7690 .. 0.8499 (allowed 0.6..0.85)
      nominal alias / passband dev         : -68.19 dB (limit -66.52) / 0.199 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -66.59 dB | Droop: 0.246 dB | fc_min: 11.11 kHz | Q: 0.769..0.850
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -66.82 dB | Droop: 0.246 dB | fc_min: 11.11 kHz | Q: 0.769..0.850
    
    Opt. parallel (C0G) - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -68.17 dB      +1.65 dB      PASS
    1 sigma      84.13450%       -67.88 dB      +1.36 dB      PASS
    2 sigma      97.72500%       -67.60 dB      +1.08 dB      PASS
    3 sigma      99.86500%       -67.32 dB      +0.80 dB      PASS
    4 sigma      99.99683%       -67.11 dB      +0.59 dB      PASS
    Maximum     100.00000%       -67.09 dB      +0.58 dB      PASS
    --------------------------------------------------------------------



    
![png](output_0_23.png)
    


    
    Opt. single (any) — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        200pF 200pF          C0G1       1%     198pF     202pF           2e-10
      ra    Sallen-Key R (both resistors)           11k 11k            R1pct      1%    10.89k    11.11k           11000
      ca    Sallen-Key Ca (feedback cap)          1.6nF 1.6nF          C0G1       1%   1.584nF   1.616nF         1.6e-09
      cb    Sallen-Key Cb (cap to ground)         680pF 680pF          C0G1       1%   673.2pF   686.8pF         6.8e-10
      ----------------------------------------------------------------------------------------------------
    
    Opt. single (any) — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 15.915 kHz
      Sallen-Key f0 / Q                    : 13.871 kHz / 0.7670
      cascade -3 dB corner (nominal)       : 11.861 kHz | worst-corner fc_min: 11.550 kHz
      Q over tolerance corners             : 0.7593 .. 0.7747 (allowed 0.6..0.85)
      nominal alias / passband dev         : -67.16 dB (limit -66.52) / 0.248 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -66.64 dB | Droop: 0.260 dB | fc_min: 11.55 kHz | Q: 0.759..0.775
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -66.87 dB | Droop: 0.260 dB | fc_min: 11.55 kHz | Q: 0.759..0.775
    
    Opt. single (any) - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -67.16 dB      +0.64 dB      PASS
    1 sigma      84.13450%       -67.07 dB      +0.55 dB      PASS
    2 sigma      97.72500%       -66.98 dB      +0.46 dB      PASS
    3 sigma      99.86500%       -66.90 dB      +0.38 dB      PASS
    4 sigma      99.99683%       -66.82 dB      +0.30 dB      PASS
    Maximum     100.00000%       -66.82 dB      +0.30 dB      PASS
    --------------------------------------------------------------------



    
![png](output_0_25.png)
    


    
    Opt. parallel (any) — Component Table (Identical on 20m & 40m)
      ----------------------------------------------------------------------------------------------------
      Ref   Role                                  Value Build          Class     Tol       Min       Max      Exact [SI]
      ----------------------------------------------------------------------------------------------------
      c_ac  AC-coupling capacitor                  10uF 10uF           X5R       20%       8uF      12uF           1e-05
      r_in  Diff-amp input resistor                  1k 1k             R1pct      1%       990     1.01k            1000
      r_fb  Diff-amp feedback resistor              50k 50k            R1pct      1%     49.5k     50.5k           50000
      c_fb  Diff-amp feedback cap (pole 1)        192pF 82pF+110pF     C0G1       1%   190.1pF   193.9pF        1.92e-10
      ra    Sallen-Key R (both resistors)         12.1k 12.1k          R1pct      1%    11.98k    12.22k           12100
      ca    Sallen-Key Ca (feedback cap)         1.62nF 620pF+1nF      C0G1       1%   1.604nF   1.636nF        1.62e-09
      cb    Sallen-Key Cb (cap to ground)         572pF 62pF+510pF     C0G1       1%   566.3pF   577.7pF        5.72e-10
      ----------------------------------------------------------------------------------------------------
    
    Opt. parallel (any) — Derived Quantities
      signal gain (r_fb / (r_in + R_src)) : 47.54 V/V = 33.54 dB (R_src = 51.80 ohm)
      AC-coupling HPF corner               : 15.13 Hz
      diff-amp feedback pole (pole 1)      : 16.579 kHz
      Sallen-Key f0 / Q                    : 13.664 kHz / 0.8415
      cascade -3 dB corner (nominal)       : 12.915 kHz | worst-corner fc_min: 12.606 kHz
      Q over tolerance corners             : 0.8331 .. 0.8499 (allowed 0.6..0.85)
      nominal alias / passband dev         : -67.06 dB (limit -66.52) / 0.188 dB (limit 0.5)
        [20m (2.2 nF Tayloe)] 128-Corner WC -> Alias: -66.54 dB | Droop: 0.199 dB | fc_min: 12.61 kHz | Q: 0.833..0.850
        [40m (4.4 nF Tayloe = 2x)] 128-Corner WC -> Alias: -66.77 dB | Droop: 0.199 dB | fc_min: 12.61 kHz | Q: 0.833..0.850
    
    Opt. parallel (any) - alias rejection statistics
    --------------------------------------------------------------------
    Level         Quantile      Response       Margin   Verdict
    Median       50.00000%       -67.06 dB      +0.54 dB      PASS
    1 sigma      84.13450%       -66.97 dB      +0.45 dB      PASS
    2 sigma      97.72500%       -66.88 dB      +0.36 dB      PASS
    3 sigma      99.86500%       -66.80 dB      +0.28 dB      PASS
    4 sigma      99.99683%       -66.73 dB      +0.21 dB      PASS
    Maximum     100.00000%       -66.72 dB      +0.20 dB      PASS
    --------------------------------------------------------------------



    
![png](output_0_27.png)
    


      Opt. single (C0G): 6 feasible / 31 E96 values of ra in 6.8–14.0 kOhm



    
![png](output_0_29.png)
    


      Opt. parallel (C0G): 7 feasible / 31 E96 values of ra in 6.8–14.0 kOhm



    
![png](output_0_31.png)
    


      Opt. single (any): 11 feasible / 31 E96 values of ra in 6.8–14.0 kOhm



    
![png](output_0_33.png)
    


      Opt. parallel (any): 7 feasible / 31 E96 values of ra in 6.8–14.0 kOhm



    
![png](output_0_35.png)
    



    
![png](output_0_36.png)
    


    
    ======================================================================================
    3B. FINAL COMPARISON — ARCH A DESIGNS & PASSIVE RC2
    ======================================================================================
    Alias zone 188.0–196.0 kHz | Target -66.52 dB
    
    Components (value [class]); a+b = two caps in parallel
                                             Original design      Original, C0G1   Opt. single (C0G) Opt. parallel (C0G)   Opt. single (any) Opt. parallel (any)
    ------------------------------------------------------------------------------------------------------------------------------------------------------------
    c_ac AC-coupling capacitor                    10uF [X5R]          10uF [X5R]          10uF [X5R]          10uF [X5R]          10uF [X5R]          10uF [X5R]
    r_in Diff-amp input resistor                  1k [R1pct]          1k [R1pct]          1k [R1pct]          1k [R1pct]          1k [R1pct]          1k [R1pct]
    r_fb Diff-amp feedback resistor              50k [R1pct]         50k [R1pct]         50k [R1pct]         50k [R1pct]         50k [R1pct]         50k [R1pct]
    c_fb Diff-amp feedback cap (pole 1)          220pF [C0G]        220pF [C0G1]         200pF [C0G]    62pF+130pF [C0G]        200pF [C0G1]   82pF+110pF [C0G1]
    ra Sallen-Key R (both resistors)             10k [R1pct]         10k [R1pct]       12.4k [R1pct]       12.1k [R1pct]         11k [R1pct]       12.1k [R1pct]
    ca Sallen-Key Ca (feedback cap)              1.5nF [C0G]        1.5nF [C0G1]         1.6nF [C0G]   750pF+910pF [C0G]        1.6nF [C0G1]    620pF+1nF [C0G1]
    cb Sallen-Key Cb (cap to ground)             750pF [C0G]        750pF [C0G1]         620pF [C0G]    75pF+560pF [C0G]        680pF [C0G1]   62pF+510pF [C0G1]
    
    Performance (20m Base & 40m 2x Tayloe)
                                             Original design      Original, C0G1   Opt. single (C0G) Opt. parallel (C0G)   Opt. single (any) Opt. parallel (any)
    ------------------------------------------------------------------------------------------------------------------------------------------------------------
    nominal fc [kHz]                                  11.040              11.040              11.778              11.943              11.861              12.915
    worst-corner fc [kHz]                             10.113              10.724              10.936              11.111              11.550              12.606
    worst-corner fc vs original                        +0.0%               +6.0%               +8.1%               +9.9%              +14.2%              +24.6%
    nominal alias 20m [dB]                            -66.62              -66.62              -68.44              -68.19              -67.16              -67.06
    worst-corner alias 20m [dB]                       -65.03              -66.10              -66.84              -66.59              -66.64              -66.54
    worst-corner alias 40m (2x) [dB]                  -65.26              -66.33              -67.07              -66.82              -66.87              -66.77
    alias margin 20m vs -66.52 dB                      -1.49               -0.42               +0.32               +0.07               +0.12               +0.02
    worst-corner passband dev [dB]                     0.370               0.334               0.261               0.246               0.260               0.199
    Q over corners                              0.673..0.743        0.700..0.714        0.764..0.844        0.769..0.850        0.759..0.775        0.833..0.850
    ADC residual [LSB] (limit 0.5)                  1.46e-10            1.46e-10            1.46e-10            1.46e-10            1.46e-10            1.46e-10
    signal gain [dB]                                    33.5                33.5                33.5                33.5                33.5                33.5
    # parallel-built caps                                  0                   0                   0                   3                   0                   3
    # C0G1 (1%) caps                                       0                   3                   0                   0                   3                   3
    relative cap cost (model units)                     0.00                1.50                0.00                0.12                1.50                1.62
    verdict (20m+40m corners+settle)                    FAIL                FAIL                PASS                PASS                PASS                PASS
    
    PASSIVE RC2 COMPARISON (Same Chain, Corners, Pools & Settling)
                                              RC2 (C0G caps)     RC2 (C0G1 caps)
    ----------------------------------------------------------------------------
    R1                                           324 [R1pct]         309 [R1pct]
    R2                                         3.65k [R1pct]       3.24k [R1pct]
    C1                                           100nF [C0G]        100nF [C0G1]
    C2                                            12nF [C0G]         13nF [C0G1]
    worst-corner alias [dB]                           -66.51              -66.53
    worst-corner passband dev [dB]                     4.324               3.883
    worst-corner fc [kHz]                              2.202               2.368
    ADC residual [LSB] (limit 0.5)                      2.15                1.91
    verdict                                       INFEASIBLE          INFEASIBLE
    
    ======================================================================================
    4. CONSOLIDATED 20m JLCPCB BASE BOM & 40m CONVERSION CHECKLIST (ALL FEASIBLE DESIGNS)
    ======================================================================================
    
    [STAGE 1: CLASS-E PA — T1 (Ropt = 7.95 Ω) (Ropt = 7.95 Ω)]
      20m JLCPCB Base : Cext = 270pF+trim(18pF) | L1 = 180nH | L2 = 220nH | C2 = 470pF
      40m Hand Add-On : +288pF parallel on Cext DNP pad | +180nH in series on L1 (remove 0R) | +220nH in series on L2 (remove 0R) | +470pF parallel on C2 DNP pad
    
    [STAGE 2: TX IMPEDANCE MATCH + HARMONIC LPF — ALL FEASIBLE DESIGNS]
    
      * DBL_LC_JA2GQP_MR_1C        [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=240pF+trim(18pF), L1=100nH, C2=330pF, C3=1nF, L2=330nH, C4=43pF, C5=330pF
        40m Target Build: C1=240pF+trim(18pF) || +240pF, L1=100nH+100nH, C2=330pF || +330pF, C3=1nF || +1nF, L2=330nH+330nH, C4=43pF || +43pF, C5=330pF || +330pF
    
      * DBL_LC_JA2GQP_MR_2C        [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=120pF+120pF+trim(18pF), L1=100nH, C2=18pF+300pF, C3=7.5pF+1nF, L2=330nH, C4=6.8pF+36pF, C5=24pF+300pF
        40m Target Build: C1=120pF+120pF+trim(18pF) || +120pF+120pF, L1=100nH+100nH, C2=18pF+300pF || +18pF+300pF, C3=7.5pF+1nF || +7.5pF+1nF, L2=330nH+330nH, C4=6.8pF+36pF || +6.8pF+36pF, C5=24pF+300pF || +24pF+300pF
    
      * DBL_LC_4Element_1R_1C      [1 Reel(s), $3.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m Target Build: C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=220pF || +220pF, C3=560pF || +560pF
    
      * DBL_LC_4Element_1R_2C      [1 Reel(s), $3.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=200pF+330pF
        40m Target Build: C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=82pF+130pF || +82pF+130pF, C3=200pF+330pF || +200pF+330pF
    
      * DBL_LC_4Element_MR_1C      [1 Reel(s), $3.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m Target Build: C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=220pF || +220pF, C3=560pF || +560pF
    
      * DBL_LC_4Element_MR_2C      [1 Reel(s), $3.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=200pF+330pF
        40m Target Build: C1=4.7pF+trim(18pF) || +4.7pF, L2=150nH+150nH, C2=82pF+130pF || +82pF+130pF, C3=200pF+330pF || +200pF+330pF
    
      * DBL_LC_LMatch_WB2CBA_MR_1C [3 Reel(s), $9.00 setup]
        20m JLCPCB Base : Cm=1.3nF+trim(18pF), Lm=100nH, C1=1.8nF, L2=270nH, C3=510pF, L4=180nH, C5=47pF
        40m Target Build: Cm=1.3nF+trim(18pF) || +1.3nF, Lm=100nH+100nH, C1=1.8nF || +1.8nF, L2=270nH+270nH, C3=510pF || +510pF, L4=180nH+180nH, C5=47pF || +47pF
    
      * DBL_LC_LMatch_WB2CBA_MR_2C [3 Reel(s), $9.00 setup]
        20m JLCPCB Base : Cm=22pF+1nF+trim(18pF), Lm=120nH, C1=43pF+1.5nF, L2=330nH, C3=200pF+300pF, L4=270nH, C5=27pF
        40m Target Build: Cm=22pF+1nF+trim(18pF) || +22pF+1nF, Lm=120nH+120nH, C1=43pF+1.5nF || +43pF+1.5nF, L2=330nH+330nH, C3=200pF+300pF || +200pF+300pF, L4=270nH+270nH, C5=27pF || +27pF
    
      * DBL_C_JA2GQP_MR_1C         [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=510pF+trim(18pF), L1=150nH, C2=220pF, C3=910pF, L2=470nH, C4=30pF, C5=270pF
        40m Target Build: C1=510pF+trim(18pF) || +1.5nF, L1=150nH, C2=220pF || +680pF, C3=910pF || +2.7nF, L2=470nH, C4=30pF || +91pF, C5=270pF || +820pF
    
      * DBL_C_JA2GQP_MR_2C         [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=24pF+510pF+trim(18pF), L1=150nH, C2=82pF+130pF, C3=910pF, L2=470nH, C4=8.2pF+22pF, C5=68pF+220pF
        40m Target Build: C1=24pF+510pF+trim(18pF) || +1.6nF, L1=150nH, C2=82pF+130pF || +620pF, C3=910pF || +2.7nF, L2=470nH, C4=8.2pF+22pF || +91pF, C5=68pF+220pF || +820pF
    
      * DBL_C_LMatch_WB2CBA_MR_1C  [3 Reel(s), $9.00 setup]
        20m JLCPCB Base : Cm=620pF+trim(18pF), Lm=220nH, C1=910pF, L2=470nH, C3=150pF, L4=39nH, C5=130pF
        40m Target Build: Cm=620pF+trim(18pF) || +1.8nF, Lm=220nH, C1=910pF || +2.7nF, L2=470nH, C3=150pF || +430pF, L4=39nH, C5=130pF || +390pF
    
      * OPT_LC_JA2GQP_MR_1C        [4 Reel(s), $12.00 setup]
        20m JLCPCB Base : C1=75pF+trim(18pF), L1=100nH, C2=330pF, C3=820pF, L2=330nH, C4=43pF, C5=240pF
        40m Target Build: C1=680pF+trim(18pF), L1=270nH, C2=470pF, C3=1.8nF, L2=820nH, C4=68pF, C5=510pF
    
      * OPT_LC_JA2GQP_MR_2C        [4 Reel(s), $12.00 setup]
        20m JLCPCB Base : C1=7.5pF+56pF+trim(18pF), L1=100nH, C2=18pF+300pF, C3=47pF+750pF, L2=330nH, C4=6.8pF+36pF, C5=240pF
        40m Target Build: C1=6.8pF+680pF+trim(18pF), L1=270nH, C2=82pF+390pF, C3=470pF+1.3nF, L2=820nH, C4=33pF+36pF, C5=75pF+430pF
    
      * OPT_LC_4Element_1R_1C      [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m Target Build: C1=4.7pF+trim(18pF), L2=330nH, C2=360pF, C3=1nF
    
      * OPT_LC_4Element_1R_2C      [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m Target Build: C1=4.7pF+trim(18pF), L2=330nH, C2=56pF+330pF, C3=470pF+510pF
    
      * OPT_LC_4Element_MR_1C      [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=220pF, C3=560pF
        40m Target Build: C1=4.7pF+trim(18pF), L2=330nH, C2=360pF, C3=1nF
    
      * OPT_LC_4Element_MR_2C      [2 Reel(s), $6.00 setup]
        20m JLCPCB Base : C1=4.7pF+trim(18pF), L2=150nH, C2=82pF+130pF, C3=18pF+510pF
        40m Target Build: C1=4.7pF+trim(18pF), L2=330nH, C2=56pF+330pF, C3=470pF+510pF
    
      * OPT_LC_LMatch_WB2CBA_MR_1C [5 Reel(s), $15.00 setup]
        20m JLCPCB Base : Cm=910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=39pF
        40m Target Build: Cm=1.8nF+trim(18pF), Lm=220nH, C1=3nF, L2=560nH, C3=1.1nF, L4=560nH, C5=51pF
    
      * OPT_LC_LMatch_WB2CBA_MR_2C [6 Reel(s), $18.00 setup]
        20m JLCPCB Base : Cm=62pF+910pF+trim(18pF), Lm=100nH, C1=1.6nF, L2=270nH, C3=560pF, L4=330nH, C5=8.2pF+33pF
        40m Target Build: Cm=33pF+1.5nF+trim(18pF), Lm=220nH, C1=27pF+3nF, L2=560nH, C3=33pF+1.1nF, L4=680nH, C5=36pF+68pF
    
    [STAGE 3: RX TAYLOE MIXER + ACTIVE AUDIO FILTER — ALL FEASIBLE DESIGNS]
      20m JLCPCB Base (Tayloe) : 1x 2.2nF C0G per I/Q branch (4x total)
      40m Hand Add-On (Tayloe) : +1x 2.2nF C0G parallel per I/Q branch (-> 4.4nF total)
      Active Audio Filter      : 100% IDENTICAL on 20m & 40m (zero rework required):
        * Opt. single (C0G)     : c_fb=200pF [C0G], ra=12.4k [R1pct], ca=1.6nF [C0G], cb=620pF [C0G]
        * Opt. parallel (C0G)   : c_fb=62pF+130pF [C0G], ra=12.1k [R1pct], ca=750pF+910pF [C0G], cb=75pF+560pF [C0G]
        * Opt. single (any)     : c_fb=200pF [C0G1], ra=11k [R1pct], ca=1.6nF [C0G1], cb=680pF [C0G1]
        * Opt. parallel (any)   : c_fb=82pF+110pF [C0G1], ra=12.1k [R1pct], ca=620pF+1nF [C0G1], cb=62pF+510pF [C0G1]

