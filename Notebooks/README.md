# Simulation & Optimization Notebooks (`Notebooks/`)

RF, power-stage, and mixed-signal design notebooks for the **4.85 V / 1.65 W QRP Class-E HF Transceiver** (`20 m` & `40 m` bands).

## Active Directories

| Directory | Notebook & Export | Documentation | Scope & Key Results |
| :--- | :--- | :--- | :--- |
| [`GaNUnified/`](GaNUnified/) | [`GaNUnified.ipynb`](GaNUnified/GaNUnified.ipynb) · [`GaNUnified.md`](GaNUnified/GaNUnified.md) | [`README.md`](GaNUnified/README.md) | **Class-E PA Transient ODE & ZVS/ZDS Optimization (`EPC2037`)**<br>• Digitizes raw `C_oss(V_DS)` charge curves (`Fig. 1–11`) and temperature-scaled `R_DS(on)` (`75 °C`).<br>• Solves self-consistent periodic steady-state ODEs (`T1` free `R_opt` at `1.65 W`, `T2` locked `R_opt = 8.223 Ω` achieving `94.01%` system efficiency with true ZVS/ZDS). |
| [`LPFOpt/`](LPFOpt/) | [`LPFOpt.ipynb`](LPFOpt/LPFOpt.ipynb) · [`LPFOpt.md`](LPFOpt/LPFOpt.md) | [`README.md`](LPFOpt/README.md) | **TX Harmonic Filter MINLP Synthesis & UQ**<br>• Maps `LMatch_WB2CBA`, `JA2GQP` (`2f0` + `3f0` notches), and `4Element` (`2f0` notch) onto **JLCPCB Basic E24 C0G capacitors (`$0` fee)** and **E12 wirewound inductor reels (`$3` per reel)**.<br>• Evaluates `(1-Reel vs. Multi-Reel) × (1-Cap vs. 2-Cap)` with vectorized `±5%` Monte Carlo UQ (`90%` & `95%` tolerance intervals).<br>• **Result:** Single-reel `JA2GQP` (`390 nH` on 20 m, `820 nH` on 40 m) meets FCC `-43 dBc` limits (`+11.5 dB` to `+15.8 dB` worst-case MC margin) using **one `$3.00` inductor reel**. |
| [`AudioOpt/`](AudioOpt/) | [`AudioOpt.ipynb`](AudioOpt/AudioOpt.ipynb) · [`AudioOpt.md`](AudioOpt/AudioOpt.md) | [`README.md`](AudioOpt/README.md) | **RX Baseband Gain, Anti-Aliasing & DSP Decimation UQ**<br>• Models the 4-phase commutating Tayloe detector (`4 × (R_s + R_on)`), `10 µF` AC coupling, active `2× OPA2320AIDR` (`+33.5 dB` / `50×` gain + 3-pole Butterworth AAF), `100 Ω + 1 nF` ADC kickback buffer, `3.10 V` STM32U375 ADC settling (`5.41 ns << 2600 ns`), and digital `CIC → Half-Band → Polyphase FIR` decimation (`192 → 8 kSps`).<br>• Verifies `-66.69 dB` alias rejection at `188 kHz` (`F_crit`) and `S9+52 dB` (`-20.0 dBm`) clipping headroom under `±5% C / ±1% R` Monte Carlo UQ. |

## Archive

| Directory | Documentation | Contents |
| :--- | :--- | :--- |
| [`Attic/`](Attic/) | [`README.md`](Attic/README.md) | Deprecated early prototypes and single-topology exploratory notebooks superseded by the three consolidated notebooks above. |

## Directory Tree

```text
Notebooks/
├── README.md
├── GaNUnified/
│   ├── README.md
│   ├── GaNUnified.ipynb
│   └── GaNUnified.md
├── LPFOpt/
│   ├── README.md
│   ├── LPFOpt.ipynb
│   └── LPFOpt.md
├── AudioOpt/
│   ├── README.md
│   ├── AudioOpt.ipynb
│   └── AudioOpt.md
└── Attic/
    └── README.md
```

## Dependencies
```bash
pip install numpy scipy matplotlib pandas
```
