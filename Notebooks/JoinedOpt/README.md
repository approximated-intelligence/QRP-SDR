# Joint 20 m Base → 40 m Add-On Frontend & Audio Optimization

* **Notebook:** [`JoinedOpt.ipynb`](JoinedOpt.ipynb)
* **Rendered Output:** [`JoinedOpt.md`](JoinedOpt.md)
* **Parent Directory:** [`Notebooks/`](../README.md)

Joint discrete optimization for a **20 m (`14.10 MHz`) JLCPCB base assembly** convertible to **40 m (`7.05 MHz`)** via parallel capacitor DNP pads and series inductor jumpers (`2×` component doubling), evaluated at `50 Ω → 50 Ω` on the TX filter path.

The notebook is authoritative for all numeric results; this README describes scope, methodology, and headline results.

## Scope & Methodology

* **Part 1 — RX Audio & Tayloe Chain:**
  * Optimizes Architecture A (`OPA2320`) discrete E24 C0G capacitors and E96 1% resistors (`c_fb`, `ra`, `ca`, `cb`) so the active audio filter remains **100% identical** on both bands.
  * Doubles the Tayloe baseband integration capacitor from `2.2 nF` (`20 m`) to `4.4 nF` (`40 m`, via parallel DNP pad) and verifies all `2^7 = 128` tolerance corners on both bands against the `-66.52 dB` alias limit, `0.5 dB` passband droop limit, and `0.60–0.85` Sallen-Key `Q` window.
* **Part 2 — Joint 20 m → 40 m TX Harmonic LPF (`50 Ω → 50 Ω`):**
  * Optimizes 20 m base values (`E24` C0G capacitors, single `E12` inductor reel) such that doubling all capacitors in parallel (`2× C`) and inductors in series (`2× L`) satisfies `S11`, insertion loss, and FCC `-43 dBc` harmonic suppression on both `14.10 MHz` and `7.05 MHz` (`Q_L = 60`).
  * Evaluates `JA2GQP`, `4Element`, and `LMatch_WB2CBA` under `5,000`-run `±5%` Monte Carlo UQ.

## Headline Results

* **RX Audio Filter (1-Reel / Single-Cap C0G, Identical 20 m & 40 m):** `c_fb = 200 pF`, `ra = 12.4 kΩ`, `ca = 1.6 nF`, `cb = 620 pF` passes all 128 corners on both bands (`-66.84 dB` worst-corner alias on 20 m, `-67.08 dB` on 40 m, `fc_min = 10.94 kHz`).
* **TX LPF (`50 Ω → 50 Ω`, 1-Reel, Exact `2×` Doubling):**
  * **`JA2GQP` (`1` reel, `$3.00` setup):** `C1 = 180 pF`, `L1 = L2 = 470 nH`, `C2 = 68 pF`, `C3 = 300 pF`, `C4 = 30 pF`, `C5 = 120 pF` achieves `S11 = -22.93 dB` nominal (`-15.61 dB` `±5%` MC worst-case) and `+32.5 dB` nominal (`+15.3 dB` MC worst-case) FCC margin on both bands.
  * **`4Element` (`1` reel):** `S11 = -19.56 dB` nominal (`-14.38 dB` MC worst-case), `+8.7 dB` nominal (`+4.8 dB` MC worst-case) FCC margin.
  * **`LMatch_WB2CBA` (`1` reel):** Fails 1-reel `50 Ω → 50 Ω` matching (`S11 = -8.93 dB` nominal).
