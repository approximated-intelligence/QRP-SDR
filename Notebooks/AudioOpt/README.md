# RX Baseband Gain, Anti-Aliasing & DSP Decimation UQ

* **Notebook:** [`AudioOpt.ipynb`](AudioOpt.ipynb)
* **Rendered Output:** [`AudioOpt.md`](AudioOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

End-to-end receive-path simulation from the **4-phase commutating Tayloe detector (`74CBTLV3253`)** through the **`2× OPA2320AIDR`** active gain/filter stage, **`STM32U375`** 12-bit ADC (`192 kSps`, `3.10 V` rail), and the digital decimation chain (`192 → 48 → 24 → 8 kSps`). The notebook is authoritative for all numeric results; this README only describes scope and signal-chain topology.

## Signal Chain

* **Tayloe Baseband Model:** 25% per-phase duty-cycle resistance scaling (`R_bb = 4 × (R_s + R_on)`), with band-selectable baseband capacitor (`2.2 nF` for `20 m`, `4.3 nF` for `40 m`).
* **Analog Front-End, Architecture A (`2× OPA2320AIDR`, 4 op-amps):**
  * `10 µF` + `1.0 kΩ` AC coupling.
  * Stage 1 (`U14A`): differential amplifier (`R_in = 1.0 kΩ`, `R_fb = 50 kΩ || 220 pF`). The notebook reports the realized gain including `R_SOURCE_AUDIO`, not the ideal `R_fb / R_in`.
  * Stage 2 (`U14B`): unity-gain 2-pole Butterworth Sallen-Key LPF + `100 Ω / 1 nF` ADC kickback buffer.
* **Analog Front-End, Architecture B (`3× OPA2320AIDR`, 6 op-amps):** compared side-by-side for cost vs. performance.
* **ADC Settling:** Switched-capacitor settling budget versus `T_acq = 2600 ns`.
* **Digital Decimation:** CIC (`R=4`, `N=4`) → half-band FIR (`48 → 24 kSps`) → polyphase FIR (`24 → 8 kSps`).

## Monte Carlo UQ

Both architectures are evaluated under `±5%` capacitor and `±1%` resistor tolerances (`1,000` runs). The notebook reports nominal, median, `90%`, and `95%` tolerance intervals for the complete chain (Tayloe × analog AAF × digital decimation) and flags the alias-zone worst case against the derived target. **Refer to `AudioOpt.md` for the current pass/fail numbers** — the tail behavior (97.5th percentile) is what matters for compliance and is reported per architecture.
