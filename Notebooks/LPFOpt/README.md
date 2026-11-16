# TX Harmonic Filter MINLP Synthesis & UQ

* **Notebook:** [`LPFOpt.ipynb`](LPFOpt.ipynb)
* **Rendered Output:** [`LPFOpt.md`](LPFOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

Mixed-Integer Nonlinear Program (MINLP) synthesis plus vectorized Monte Carlo UQ mapping HF TX low-pass filters onto **JLCPCB Basic E24 C0G capacitors (`$0` feeder fee)** and **E12 wirewound inductor reels (`$3` per unique reel)**. The notebook is authoritative for all BOM selections, S-parameters, and MC results; this README only describes scope.

## Topologies Evaluated

* **`LMatch_WB2CBA`:** 5-pole Chebyshev LPF with input L-match.
* **`JA2GQP`:** 5th-order elliptic LPF with parallel-resonant transmission zeros at `2f0` and `3f0`.
* **`4Element`:** Compact 4-element LPF with a single parallel-resonant transmission zero at `2f0`.

## Sweep

For each band (`20 m` / `14.10 MHz` and `40 m` / `7.05 MHz`) the notebook evaluates all topologies across a 2×2 mode grid:

| Mode | Max inductor reels | Max caps per node |
| :--- | :---: | :---: |
| `1R_1C` | 1 | 1 |
| `1R_2C` | 1 | 2 (parallel pairs allowed) |
| `MR_1C` | unlimited | 1 |
| `MR_2C` | unlimited | 2 |

The single-reel modes (`1R_*`) force the optimizer to a shared inductance value across all inductor positions, with a small reel-count penalty in the objective.

## UQ

Each candidate BOM is re-evaluated under `±5%` tolerance on every L and C (`2,000` runs per candidate) using a vectorized ABCD cascade. The notebook reports worst-case `S11`, `S21`, and the FCC-`-43 dBc` harmonic margin under tolerance, per topology, per band, per mode.

**See `LPFOpt.md`'s per-band output blocks for the actual winning BOMs and their UQ margins.** The README intentionally does not duplicate them, to keep the notebook authoritative.
