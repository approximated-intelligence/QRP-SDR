# RX Baseband Gain, Anti-Aliasing & DSP Decimation UQ

* **Notebook:** [`AudioOpt.ipynb`](AudioOpt.ipynb)
* **Rendered Output:** [`AudioOpt.md`](AudioOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

End-to-end receive-path simulation from the **4-phase commutating Tayloe detector (`74CBTLV3253`)** through the **OPA2320 active gain/filter stage**, **STM32U375** 12-bit ADC (`192 kSps`, `3.10 V` reference), and digital decimation chain (`192 → 48 → 24 → 8 kSps`). The notebook also selects discrete component values for the analog filter with a sequential MILP, verified by exhaustive tolerance-corner enumeration.

The notebook is authoritative for all numeric results; this README describes scope, method and topology, and summarizes the headline results.

## Signal Chain

* **Tayloe Baseband Model:** 25% per-phase duty-cycle resistance scaling (`R_bb = 4 × (R_s + R_on)` = `52.88 Ω`), with band-selectable baseband capacitor (`2.2 nF` for `20 m`, `4.3 nF` for `40 m`). The analysis uses the 20 m pole (`1.368 MHz`).
* **Analog Front-End, Architecture A (`2× OPA2320AIDR`, 4 op-amps):**
  * `10 µF` AC coupling (X5R, ±20%).
  * Stage 1 (`U14A`): differential amplifier (`R_in = 1.0 kΩ`, `R_fb = 50 kΩ || C_fb`). The notebook reports the realized gain including `R_SOURCE_AUDIO` (`47.49 V/V` = `33.53 dB`), not the ideal `R_fb / R_in`.
  * Stage 2 (`U14B`): component-derived 2-pole Sallen-Key LPF (`R_A` for both resistors, `C_A` feedback, `C_B` to ground; `f0 = 1/(2π·R_A·√(C_A·C_B))`, `Q = ½·√(C_A/C_B)`).
  * The proposed `100 Ω / 1 nF` ADC kickback network is **not** part of the transfer function; `R_kick = 100 Ω` is used only in the settling check.
* **Passive RC2 reference:** a two-stage passive RC filter (4 parameters: `R1`, `R2`, `C1`, `C2`) optimized with differential evolution as a baseline. It is evaluated against the same alias target, the passband limit and the ADC settling budget.
* **Architecture B (`3× OPA2320`):** removed from the notebook. Only Architecture A and the passive RC2 are analyzed.
* **ADC Settling:** switched-capacitor settling budget (`C_sample = 6 pF`, `13·ln2·τ` required) versus `T_acq = 2600 ns`. This covers RC settling only.
* **Digital Decimation:** CIC (`R=4`, `N=4`) → 63-tap half-band FIR (`48 → 24 kSps`) → 161-tap polyphase FIR (`24 → 8 kSps`).
* **Not modeled:** front-end (T/R switch + LPF) insertion loss, ADC kickback network, op-amp finite bandwidth/noise.

## Requirements

| Item | Value | Derivation |
| :--- | :---: | :--- |
| Alias zone | `188–196 kHz` | `fs ± fs_out/2` (first alias of the `8 kSps` output) |
| Required alias response | `≤ -66.52 dB` | `-30 dBFS` blocker must land `10 dB` below the ADC noise floor in `2.7 kHz` (`68 dB` typ. SNR) |
| Passband | `0.3–2.7 kHz` | SSB voice |
| Max passband deviation | `0.5 dB` | |
| Sallen-Key Q window | `0.60–0.85` | Applied at every tolerance corner |

All frequency responses are normalized to the low-frequency signal gain; amplifier gain is handled separately.

## Tolerance Model

Passives are grouped by class, each with its own tolerance and assumed pre-screening bin yield. The class names are labels defined in the notebook, not manufacturer part codes.

| Class | Tolerance | Assumed bin yield | Relative cost (optimizer) | Used for |
| :--- | :---: | :---: | :---: | :--- |
| `C0G` (NP0) | ±5% | 0.99 | 0.00 | Original filter caps |
| `C0G2` | ±2% | 0.99 | 0.15 | Optimizer choice |
| `C0G1` | ±1% | 0.99 | 0.50 | Optimizer choice |
| `X5R` | ±20% | 0.95 | n/a | `C_AC = 10 µF` |
| `R1pct` (thick-film 1%) | ±1% | 0.99 | 0.00 | All resistors |
| `L10pct` | ±10% | 0.95 | n/a | Defined, unused |

Two parallel caps add `0.04` relative cost. **Tolerances, yields and costs are assumptions, not datasheet data.** Before building a `C0G1` design, confirm the chosen values exist as ±1% C0G parts from your supplier; if not, relax the class and rerun.

Monte Carlo draws each part from `N(nominal, σ)` truncated at its tolerance, with `σ = tol / (√2 · erfinv(yield))`. A uniform-distribution mode is also available. `n_runs = 5000` per design, seed `1`.

## Compliance

**The compliance number is the corner**, not the MC percentile. The notebook evaluates all `2^7 = 128` combinations of `nominal × (1 ± tol)` for the seven Arch A parts and takes the worst alias response, passband deviation, `-3 dB` corner and Q. The alias band is sampled at 401 points. MC percentiles are informational: they describe the assumed distribution, not the guarantee.

## Discrete Part Optimization

The optimizer maximizes the **worst-corner `-3 dB` bandwidth** subject to the alias, passband and Q limits holding at every corner.

* **Variables:** `C_FB`, `R_A`, `C_A`, `C_B` (`C_AC`, `R_in`, `R_fb` stay fixed).
* **Candidate pools:** capacitors from the E24 series (`47 pF–10 nF`), optionally built as **two parallel caps**; `R_A` from the E96 series (`4–30 kΩ`). Each capacitor candidate also chooses a tolerance class (`C0G`, `C0G2`, `C0G1`).
* **Method:** sequential MILP. Each step linearizes the metrics around the current design, restricts candidates to a trust region (`|ln(v/x)| ≤ delta`, `delta = 0.15`), and solves with slack variables so it is always feasible. The proposed step is then checked with the **exact 128-corner enumeration** and accepted only if it strictly improves (less violation, or more worst-corner `fc` when feasible). Rejections shrink the trust region and add curvature margin. It stops when the MILP returns the current design.
* **Two variants are run from the original design:** single caps only, and parallel caps allowed.
* **Search is local.** The result is a good local optimum around the original design, not a proven global optimum.

## Results (REV 40N)

| | Original | Optimized, single caps | Optimized, parallel caps |
| :--- | :---: | :---: | :---: |
| `C_FB` | 220 pF `C0G` | 200 pF `C0G1` | 82p+110p `C0G1` (192 pF) |
| `R_A` | 10 kΩ | 11 kΩ | 12.1 kΩ |
| `C_A` | 1.5 nF `C0G` | 1.6 nF `C0G1` | 620p+1n `C0G1` (1.62 nF) |
| `C_B` | 750 pF `C0G` | 680 pF `C0G1` | 62p+510p `C0G1` (572 pF) |
| Nominal `-3 dB` corner | 11.04 kHz | 11.86 kHz | 12.92 kHz |
| Worst-corner `-3 dB` corner | 10.11 kHz | 11.55 kHz | 12.61 kHz |
| Worst-corner alias | -65.03 dB | -66.64 dB | -66.54 dB |
| Alias margin (limit -66.52 dB) | -1.49 dB | +0.12 dB | +0.02 dB |
| Worst-corner passband deviation | 0.370 dB | 0.260 dB | 0.199 dB |
| Q over corners | 0.673–0.743 | 0.759–0.775 | 0.833–0.850 |
| MC alias, p99.9 | -65.69 dB | -66.89 dB | -66.80 dB |
| **Corner verdict** | **FAIL** | **PASS** | **PASS** |

* The original design **fails** the corner analysis. Its MC median (`-66.61 dB`) passes but the tail does not (97.725th percentile `-66.04 dB`).
* Both optimized designs pass at all 128 corners.
* Parallel caps buy about `1 kHz` more worst-corner bandwidth than single caps, at the cost of 3 extra capacitors, a Q pinned at the top of its window (`0.8499` vs `0.85`) and an alias margin of only `+0.02 dB`. That margin is below the expected modeling error (401-point alias grid, idealized digital filters). The single-cap design is the more conservative choice.
* The parallel design is tuned to a narrow operating point: in the `R_A` sweep (other parts fixed) only `7/31` E96 values remain feasible, versus `11/31` for single caps.
* **Passive RC2: infeasible.** Nominal `-3 dB` corner `6.15 kHz`, alias `-52.09 dB` (need `-66.52 dB`), passband deviation `0.835 dB` (limit `0.5 dB`), and ADC settling requirement `≈2.6 µs` against the `2.6 µs` budget. Because of the high source resistance driving the ADC, it does not meet the requirements.

## Notebook Layout

See [AudioOpt.md](AudioOpt.md):

* **§5** original design: component table, derived quantities, corner verdict, Monte Carlo.
* **§6.1 / §6.2** MILP optimization (single caps / parallel caps), each with component table, corners, MC statistics and UQ plots.
* **§7** analog response comparison, `0–24 kHz`, for all Arch A designs and the passive RC2.
* **§8** `R_A` sweeps. Only values that pass all corner, passband and Q constraints are plotted.
* **§9** side-by-side comparison of all designs, plus final component tables.

Fixed parameters, the passive RC2 study and the ADC settling check run before §5. The notebook prints the component values (value, build, class, tolerance, min/max) for every design.