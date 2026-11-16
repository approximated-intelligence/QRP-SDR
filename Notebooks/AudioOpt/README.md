# RX Baseband Gain, Anti-Aliasing & DSP Decimation UQ

* **Notebook:** [`AudioOpt.ipynb`](AudioOpt.ipynb)
* **Rendered Output:** [`AudioOpt.md`](AudioOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

End-to-end receive-path simulation from the **4-phase commutating Tayloe detector (`74CBTLV3253`)** through the **OPA2320 active gain/filter stage**, **STM32U375** 12-bit ADC (`192 kSps`, `3.10 V` reference), and digital decimation chain (`192 → 48 → 24 → 8 kSps`). The notebook is authoritative for all numeric results; this README only describes scope and signal-chain topology.

## Signal Chain

* **Tayloe Baseband Model:** 25% per-phase duty-cycle resistance scaling (`R_bb = 4 × (R_s + R_on)`), with band-selectable baseband capacitor (`2.2 nF` for `20 m`, `4.3 nF` for `40 m`).
* **Analog Front-End, Architecture A (`2× OPA2320AIDR`, 4 op-amps):**
  * `10 µF` AC coupling.
  * Stage 1 (`U14A`): differential amplifier (`R_in = 1.0 kΩ`, `R_fb = 50 kΩ || 220 pF`). The notebook reports the realized gain including `R_SOURCE_AUDIO`, not the ideal `R_fb / R_in`.
  * Stage 2 (`U14B`): component-derived 2-pole Sallen-Key LPF. The proposed `100 Ω / 1 nF` ADC kickback network is not included in the active transfer function.
* **Analog Front-End, Architecture B (`3× OPA2320AIDR`, 6 op-amps):** component definitions and preliminary calculations remain, but its transfer function, corner analysis, and Monte Carlo execution are currently disabled.
* **ADC Settling:** Switched-capacitor settling budget versus `T_acq = 2600 ns`. Section 3 notes that this check covers RC settling only; Arch B's pole-3 into the 6 pF sample cap adds a charge-share error that the RC check does not capture.
* **RX Front-End Loss:** `L_FRONTEND_DB` configures T/R switch + LPF insertion loss ahead of the Tayloe (default `0.0 dB`; set to a realistic 0.5–1.0 dB for noise-floor analysis). It is currently reported but not applied to the transfer function.
* **Digital Decimation:** CIC (`R=4`, `N=4`) → half-band FIR (`48 → 24 kSps`) → polyphase FIR (`24 → 8 kSps`).

## Tolerance Model

Passives are grouped by class, each with its own datasheet tolerance and assumed pre-screening bin yield:

| Class | Datasheet tol | Assumed bin yield |
| :--- | :---: | :---: |
| C0G/NP0 (signal-path caps) | ±5% | 0.99 |
| X5R (`C_AC = 10 µF`) | ±20% | 0.95 |
| Thick-film 1% R | ±1% | 0.99 |
| Inductors | ±10% | 0.95 |

The shipped part is modelled as `N(nominal, σ)` truncated at the datasheet tolerance, with `σ = tol / (√2 · erfinv(yield))`. A uniform-distribution reference run is also supported. `n_runs = 5000` per architecture when enabled; seed is fixed at `42`.

## Compliance

**The compliance number is the corner**, not the MC percentile. The notebook evaluates all `2^N` combinations of `nominal × (1 ± tol)` per component: 128 corners for Arch A. Architecture B corner analysis is currently disabled.

For REV 40N, Arch A's worst alias response is `-65.03 dB` against a required maximum of `-66.52 dB`, giving a compliance margin of `-1.49 dB` and a **FAIL** verdict.

MC percentiles are informational — they describe the assumed distribution, not the guarantee. The MC median is `-66.64 dB`, while the 97.725th percentile is `-66.06 dB`; both the higher-percentile tail and exhaustive corner fail the requirement.

**See [AudioOpt.md](AudioOpt.md) §4 for the current corner verdict, §5 for Monte Carlo statistics, and §6 for the final summary.** The passive RC2 optimization in §2 also reports `Feasible: False`.