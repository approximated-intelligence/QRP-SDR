# QRP Class-E HF Transceiver

A small HF transceiver design for the **20 m** (`14.10 MHz`) and **40 m** (`7.05 MHz`) amateur bands, built around a **4.85 V / 1.65 W** Class-E power amplifier using the **EPC2037** enhancement-mode GaN FET.

## Design Notebooks

Currently this is a design stage, currently at validation, the calculations are in [`Notebooks/`](Notebooks/), organized into three self-contained analysis notebooks. Each directory contains an authoritative Jupyter notebook (`.ipynb`), its rendered `.md` output, and a short README.

**The notebooks are the single source of truth for all design values, tolerances, and simulation results** — the READMEs only describe scope and point to the notebooks.

| Design Area | Directory | Notebook | Rendered Output |
| :--- | :--- | :--- | :--- |
| TX power stage | [`Notebooks/GaNUnified/`](Notebooks/GaNUnified/) | `GaNUnified.ipynb` | [`GaNUnified.md`](Notebooks/GaNUnified/GaNUnified.md) |
| TX/RX harmonic filter | [`Notebooks/LPFOpt/`](Notebooks/LPFOpt/) | `LPFOpt.ipynb` | [`LPFOpt.md`](Notebooks/LPFOpt/LPFOpt.md) |
| RX baseband | [`Notebooks/AudioOpt/`](Notebooks/AudioOpt/) | `AudioOpt.ipynb` | [`AudioOpt.md`](Notebooks/AudioOpt/AudioOpt.md) |

Deprecated prototypes live in [`Notebooks/Attic/`](Notebooks/Attic/) and are listed there.

## Dependencies

All notebooks run on the same minimal Python stack:

```bash
pip install numpy scipy matplotlib pandas
```
