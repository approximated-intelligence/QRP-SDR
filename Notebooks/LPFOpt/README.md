# TX Harmonic Filter MINLP Synthesis & UQ

* **Notebook:** [`LPFOpt.ipynb`](LPFOpt.ipynb)
* **Rendered Output:** [`LPFOpt.md`](LPFOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

Mixed-Integer Nonlinear Program (MINLP) and vectorized Monte Carlo Uncertainty Quantification (`±5%`, `2,000` runs) mapping HF TX low-pass filters onto **JLCPCB Basic E24 C0G capacitors (`$0` feeder fee)** and **E12 wirewound inductor reels (`$3` per unique reel)**.

## Topologies Evaluated
* **`LMatch_WB2CBA`:** 5-pole Chebyshev LPF (`0.1 dB` ripple) with input L-match.
* **`JA2GQP`:** 5th-order elliptic LPF with parallel-resonant transmission zeros at `2f0` and `3f0`.
* **`4Element`:** Compact 4-element LPF with a single parallel-resonant transmission zero at `2f0`.

## Selected Single-Reel BOM Solutions (`$3.00` Inductor Loading Fee)

| Band | Topology | Inductor Reel (`$3.00`) | Basic E24 Capacitors (`$0.00` Fee) | Nominal `S11` / `S21` | `±5%` MC Worst-Case (`S11` / FCC Margin) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **20 m (`14.10 MHz`)** | **`JA2GQP` (`1R_2C`)** | `2× 390 nH` (1 reel) | `180p`, `6.8p+75p`, `300p`, `6.2p+30p`, `130p` | `-36.37 dB` / `-0.25 dB` | `-19.91 dB` / `+11.5 dB` |
| **20 m (`14.10 MHz`)** | **`JA2GQP` (`1R_1C`)** | `2× 390 nH` (1 reel) | `180p`, `82p`, `270p`, `39p`, `130p` | `-28.81 dB` / `-0.24 dB` | `-18.16 dB` / `+9.0 dB` |
| **40 m (`7.05 MHz`)** | **`JA2GQP` (`1R_2C`)** | `2× 820 nH` (1 reel) | `430p`, `5.6p+150p`, `680p`, `33p+36p`, `300p` | `-28.85 dB` / `-0.29 dB` | `-17.15 dB` / `+15.8 dB` |
