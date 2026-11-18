# Class-E GaN Power Amplifier Optimization (`EPC2037`)

* **Notebook:** [`GaNOpt.ipynb`](GaNOpt.ipynb)
* **Rendered Output:** [`GaNOpt.md`](GaNOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

Transient ODE simulation and constrained SLSQP optimization of the **4.85 V / 1.65 W nominal Class-E HF Power Amplifier** (`14.0 MHz` / `7.0 MHz`) using the **EPC2037** enhancement-mode GaN FET. The notebook is authoritative for all design values and results; this README only describes scope.

## Methodology

* **Datasheet Digitization:** Raw `C_oss(V_DS)` from Fig. 5, R_DS(on)–T_j normalization from Fig. 8 (temperature-scaled value is *reported* at `T_j = 75 °C` for reference; the optimizations run at the `25 °C` R_DS(on) unless overridden in the notebook), SOA from Fig. 10, Z_th from Fig. 11.
* **Self-Consistent Periodic Steady-State:** Coupled on/off-state ODEs (`LSODA`) iterated via `brentq` until the cycle-averaged drain voltage matches `V_DD = 4.85 V`.
* **Constrained Optimization (SLSQP):**
  * **T1 (`R_opt` free):** Maximizes efficiency subject to `V_DS(t_on) = 0`, `dV_DS/dt(t_on) = 0`, and `P_out = 1.65 W`.
  * **T2 (`R_opt` locked at 8.223 Ω):** Same ZVS/ZDS constraints, but `P_out` floats — this case *does not* hit the 1.65 W target and is included only for comparison.

## Baseline Cases

* **A:** Analytical Class-E design with a constant `C_oss = 12 pF` model.
* **X:** Same component values, real digitized `C_oss(V_DS)`.
* **E:** Naive `C_ext = C_shunt,ideal − C_oss(50 V)` substitution.

## Key Results (`14.0 MHz`)

| Case | `C_ext` (pF) | `L_1` (nH) | `L_2` (nH) | `C_2` (pF) | `R_opt` (Ω) | `P_out` (mW) | `η_sys` (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | 241.8 | 107.7 | 280.4 | 460.8 | 8.223 | 1615.9 | 93.17 | Analytical, constant-C |
| **T1** | 348.6 | 106.2 | 278.9 | 446.2 | 6.292 | 1650.9 | 92.41 | `R_opt` free, `P = 1.65 W`, ZVS/ZDS |
| **T2** | 271.3 | 115.3 | 288.0 | 469.1 | 8.223 | 1271.3 | 94.01 | `R_opt` locked, `P` floats, ZVS/ZDS |

**Note:** T2's reported `P_out` is ~1.27 W, not 1.65 W — the locked-`R_opt` constraint shifts the operating point away from the nominal power target. The `η_sys` column is `P_load / (V_DD · I_DC + P_gate)`.
