# Simulation & Optimization Notebooks (`Notebooks/`)

RF, power-stage, and mixed-signal design notebooks for the **4.85 V / 1.65 W QRP Class-E HF Transceiver** (`20 m` & `40 m` bands).

Each subdirectory contains an authoritative notebook (`.ipynb`), its rendered output (`.md`), and a short navigation README. **The notebooks are the single source of truth for design values, tolerances, and results** — READMEs describe scope and point to the notebooks.

## Active Directories

| Directory | Notebook & Export | Documentation | Scope |
| :--- | :--- | :--- | :--- |
| [`GaNOpt/`](GaNOpt/) | [`GaNOpt.ipynb`](GaNOpt/GaNOpt.ipynb) · [`GaNOpt.md`](GaNOpt/GaNOpt.md) | [`README.md`](GaNOpt/README.md) | Class-E PA transient ODE solve + constrained SLSQP optimization (`EPC2037`). Digitizes raw `C_oss(V_DS)`, R_DS(on)–T_j, SOA, and Z_th curves from the datasheet figures. |
| [`LPFOpt/`](LPFOpt/) | [`LPFOpt.ipynb`](LPFOpt/LPFOpt.ipynb) · [`LPFOpt.md`](LPFOpt/LPFOpt.md) | [`README.md`](LPFOpt/README.md) | TX harmonic filter MINLP synthesis (`LMatch_WB2CBA`, `JA2GQP`, `4Element`) onto E24 C0G capacitors + E12 inductor reels, with vectorized `±5%` Monte Carlo UQ. |
| [`AudioOpt/`](AudioOpt/) | [`AudioOpt.ipynb`](AudioOpt/AudioOpt.ipynb) · [`AudioOpt.md`](AudioOpt/AudioOpt.md) | [`README.md`](AudioOpt/README.md) | RX baseband gain, anti-aliasing, and DSP decimation UQ: Tayloe detector → OPA2320 gain/AAF → STM32U375 ADC settling → CIC/HB/Polyphase FIR. |

## Archive

| Directory | Documentation | Contents |
| :--- | :--- | :--- |
| [`Attic/`](Attic/) | [`README.md`](Attic/README.md) | Deprecated early prototypes and single-topology exploratory notebooks superseded by the three consolidated notebooks above. See `Attic/README.md` for the full list of archived `.ipynb` files. |

## Directory Tree

The `Attic/` archive also contains the superseded `.ipynb` files listed in its own README (not enumerated here, to avoid drift):

```text
Notebooks/
├── README.md
├── GaNOpt/
│   ├── README.md
│   ├── GaNOpt.ipynb
│   └── GaNOpt.md
├── LPFOpt/
│   ├── README.md
│   ├── LPFOpt.ipynb
│   └── LPFOpt.md
├── AudioOpt/
│   ├── README.md
│   ├── AudioOpt.ipynb
│   └── AudioOpt.md
└── Attic/
    ├── README.md
    └── (see Attic/README.md for the archived .ipynb files)
```

## Dependencies

```bash
pip install numpy scipy matplotlib pandas
```
