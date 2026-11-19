# Unified Coupled QRP Transceiver Optimization (`PA → LPF → RX`)

* **Notebook:** [`Unified.ipynb`](Unified.ipynb)
* **Rendered Output:** [`Unified.md`](Unified.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

End-to-end coupled optimization pipeline connecting the **EPC2037 Class-E GaN PA**, **joint 20 m → 40 m impedance-matching TX harmonic LPF**, and **RX Tayloe + active audio chain** using a shared reference impedance `R_opt` and extracted PA harmonic spectrum.

The notebook is authoritative for all numeric results; this README describes scope, coupling architecture, and headline results.

## System Coupling & Architecture

1. **Stage 1 — Joint 20 m / 40 m Discrete Class-E PA (`EPC2037`):**
   * Finds continuous ZVS/ZDS (`V_on = 0`, `dV/dt = 0`) operating points via `SLSQP` and maps them onto discrete E24 capacitors and E12 inductors via linearized `MILP`, including a `18 pF` mid-travel SMD trimmer (`TRIM_PA_CEXT_MID = 18 pF`) on `C_ext`.
   * Extracts exact `2f0..5f0` harmonic levels via FFT of the steady-state load current waveform and passes the worst-case harmonic envelope (`[-18.17, -34.49, -41.84, -49.37] dBc` for `T1`) and realized `R_opt` (`7.95 Ω` for `T1`, `8.22 Ω` for `T2`) downstream.
2. **Stage 2 — Joint 20 m → 40 m TX Impedance Match + Harmonic LPF (`7.95 Ω → 50 Ω`):**
   * Performs direct impedance transformation from `R_opt = 7.95 Ω` to `50 Ω` with band-specific inductor quality factors (`Q_L = 60` on `20 m`, `Q_L = 45` on `40 m`) and a `18 pF` mid-travel input trimmer (`TRIM_LPF_C1_MID = 18 pF`).
   * Uses a 4-step sequential trust-region `MILP` with diagonal Hessian `|Γ|²` curvature weighting and `±8%` frequency-locked notch constraints (`2f0` and `3f0`) across four conversion strategies (`DBL_LC`, `DBL_C`, `OPT_LC`, `OPT_C`) and the `2×2` reel/capacitor mode grid (`1R_1C`, `1R_2C`, `MR_1C`, `MR_2C`).
3. **Stage 3 — Coupled RX Tayloe & Active Audio Chain:**
   * Updates the four-phase baseband source resistance from `R_opt` (`R_audio = 4 × (7.95 + 5.0) = 51.80 Ω`), verifies ADC acquisition settling (`1.46e-10 LSB` vs. `0.5 LSB` limit), and evaluates `20 m` (`2.2 nF`) and `40 m` (`4.4 nF`) 128-corner compliance.
4. **Stage 4 — Consolidated 20 m JLCPCB Base & 40 m Conversion BOM:**
   * Emits a unified assembly checklist listing every design that simultaneously passes nominal and `±5%` Monte Carlo `S11` and FCC `-43 dBc` limits on both bands.

## Headline Results (`pa_tag = "T1"`, `R_opt = 7.95 Ω`)

* **PA (`T1` MILP Joint):** `C_ext = 270 pF + trim(18 pF)`, `L1 = 180 nH`, `L2 = 220 nH`, `C2 = 470 pF`, `R_opt = 7.95 Ω` (`η_sys = 94.08%` at `1.21 W` on 20 m; `94.32%` at `1.22 W` on 40 m with `2×` passives).
* **TX Match + LPF (`7.95 Ω → 50 Ω`):**
  * **`DBL_LC_JA2GQP_MR_1C` (`2` reels, `$6.00` setup):** `S11 = -37.97 dB` (`20 m`) / `-38.39 dB` (`40 m`) nominal (`-17.09 dB` / `-17.08 dB` `±5%` MC worst-case), with `+18.9 dB` (`20 m`) and `+18.8 dB` (`40 m`) worst-case FCC margin.
  * **`DBL_LC_4Element_1R_2C` (`1` reel, `$3.00` setup):** `S11 = -30.94 dB` (`20 m`) / `-30.76 dB` (`40 m`) nominal (`-16.72 dB` / `-16.81 dB` `±5%` MC worst-case), with `+4.5 dB` worst-case FCC margin on both bands.
  * **`DBL_C_JA2GQP_MR_1C` (`2` reels, fixed L, `4×` C):** `S11 = -25.86 dB` (`20 m`) / `-23.85 dB` (`40 m`) nominal, `+24.7 dB` / `+25.5 dB` MC worst-case FCC margin.
* **RX Audio (`R_audio = 51.80 Ω`):** `Opt. single (C0G)` (`c_fb = 200 pF`, `ra = 12.4 kΩ`, `ca = 1.6 nF`, `cb = 620 pF`) passes all 128 corners on both bands (`-66.84 dB` on 20 m, `-67.07 dB` on 40 m).
