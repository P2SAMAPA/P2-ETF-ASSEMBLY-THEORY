# Assembly Theory – Market Complexity for ETFs

Applies Assembly Theory (recently published in Nature) to measure the "assembly index" of ETF price patterns. The assembly index counts how many steps are needed to construct a given price trajectory from elementary moves. High assembly index signals more structured, potentially predictable patterns.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Quantisation of returns into primitive moves
- Assembly index via substring complexity
- Macro‑modulated assembly complexity
- Score = assembly index (higher = more structured)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-assembly-theory-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High assembly index → price pattern is complex and structured → potentially predictable.
- Low assembly index → price pattern is simple or random.

## Requirements

See `requirements.txt`.
