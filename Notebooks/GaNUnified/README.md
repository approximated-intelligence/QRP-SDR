# Class-E GaN Power Amplifier Optimization (`EPC2037`)

* **Notebook:** [`GaNUnified.ipynb`](GaNUnified.ipynb)
* **Rendered Output:** [`GaNUnified.md`](GaNUnified.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

Transient ODE simulation and constrained SLSQP optimization of the **4.85 V / 1.65 W Class-E HF Power Amplifier** (`14.0 MHz` / `7.0 MHz`) using the **EPC2037** enhancement-mode GaN FET.

## Methodology
* **Datasheet Modeling:** Digitizes raw `C_oss(V_DS)` charge curves (`Fig. 5`), temperature-normalized `R_DS(on)` at `T_j = 75 °C` (`Fig. 8`), SOA limits (`Fig. 10`), and thermal impedance (`Fig. 11`) directly without artificial calibration offsets.
* **Self-Consistent Periodic Steady-State:** Solves the coupled on/off state differential equations (`LSODA`) via `brentq` until `<v_D> = V_DD` (`4.85 V`).
* **ZVS & ZDS Optimization:** Optimizes shunt capacitance `C_ext`, series inductance `L_1`, tank inductor `L_2`, and tank capacitor `C_2` for Zero-Voltage Switching (`V_DS(t_on) = 0 V`) and Zero-Derivative Switching (`dV_DS/dt(t_on) = 0`).

## Key Results (`14.0 MHz`)

| Case | `C_ext` (pF) | `L_1` (nH) | `L_2` (nH) | `C_2` (pF) | `R_opt` (Ω) | `P_out` (mW) | `η_sys` (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A: Analytical Const-C** | 241.8 | 107.7 | 280.4 | 460.8 | 8.223 | 1615.9 | 93.17 | Untuned |
| **T1: Free `R_opt` (`1.65 W`)** | 348.6 | 106.2 | 278.9 | 446.2 | 6.292 | 1650.9 | 92.41 | **ZVS/ZDS PASS** |
| **T2: Locked `R_opt` (`8.223 Ω`)** | 271.3 | 115.3 | 288.0 | 469.1 | 8.223 | 1271.3 | 94.01 | **ZVS/ZDS PASS** |
