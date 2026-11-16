# RX Baseband Gain, Anti-Aliasing & DSP Decimation UQ

* **Notebook:** [`AudioOpt.ipynb`](AudioOpt.ipynb)
* **Rendered Output:** [`AudioOpt.md`](AudioOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

End-to-end receive-path simulation from the **4-phase commutating Tayloe detector (`74CBTLV3253`)** through the **`2× OPA2320AIDR` active gain/filter stage** (`+33.5 dB` / `50×`), **`STM32U375` 12-bit ADC (`192 kSps`, `3.10 V` rail)**, and **digital decimation chain (`192 → 48 → 24 → 8 kSps`)**.

## Signal Chain & Key Budgets
* **Tayloe Baseband Impedance:** Models `25%` per-phase duty-cycle resistance scaling (`R_bb = 4 × (8.22 Ω + 5.0 Ω) = 52.88 Ω`), giving `-3 dB` baseband corners at `1.368 MHz` (`20 m`, `2.2 nF`) and `700 kHz` (`40 m`, `4.3 nF`).
* **Analog Front-End (`2× OPA2320AIDR`, Architecture A):**
  * **AC Coupling:** `10 µF` + `1.0 kΩ` (`f_hp = 15.1 Hz`).
  * **Stage 1 (`U14A`):** Symmetric differential amplifier (`R_in = 1.0 kΩ`, `R_fb = 50 kΩ || 220 pF`) providing **`+33.5 dB` (`47.5×`)** gain and Pole 1 (`14.47 kHz`), lifting quiet `20 m` sky noise (`-121.7 dBm`) `11.2 dB` above the STM32 ADC noise floor (`-76.5 dBm`) with `S9+52 dB` (`-20.0 dBm`) clipping headroom.
  * **Stage 2 (`U14B`):** Unity-gain 2-pole Butterworth Sallen-Key LPF (`10 kΩ`, `1.5 nF`, `750 pF`, `f_0 = 15.0 kHz`, `Q = 0.707`) + `100 Ω / 1 nF` ADC kickback buffer (`t_settle = 5.41 ns << 2600 ns` `T_acq`).
* **Anti-Aliasing Performance (`F_crit = 188 kHz`):** Achieves **`-66.69 dB`** nominal rejection (`-65.78 dB` at the `97.5th` percentile across `1,000` Monte Carlo runs with `±5%` C and `±1%` R tolerances).
