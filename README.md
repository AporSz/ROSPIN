# Dr. Equator

Dr. Equator is a reproducible research scaffold for studying satellite detector
“snow” anomalies using GOES-16/18 ABI L1b scans, SWPC space-weather archives,
and EUMETSAT MSG/SEVIRI data. The analysis is intended to quantify data-quality
flags, local outliers, and dark-target noise, then produce figures for papers
and presentations.

## Repository layout

- `config/`: project settings, region of interest, bands, and event windows.
- `data/`: temporary inputs, cropped/processed data, SWPC archives, and exports.
- `notebooks/`: exploratory ABI scan inspection.
- `src/ingestion/`: GOES and EUMETSAT acquisition scaffolds.
- `src/analysis/`: anomaly metrics and statistical-analysis scaffolds.
- `src/visualization/`: shared plot style and figure-generation scaffold.
- `tests/`: unit and synthetic-injection test scaffolds.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `EUMETSAT_CONSUMER_KEY` and
`EUMETSAT_CONSUMER_SECRET` using credentials from the EUMETSAT Data Store.
Keep `.env` private; it is excluded from Git. Configure the ROI and event
windows in `config/config.yaml` before running analyses.

## Data flow

```text
S3 / EUMETSAT -> src/ingestion -> data/processed -> src/analysis
              -> data/exports -> src/visualization
```

The ingestion pipeline is intended to crop GOES scans to the configured ROI
before removing temporary full-disk downloads. Processed data should follow a
stable schema; exported arrays and tables are intended as inputs to figure
generation.
