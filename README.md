# Dr. Equator

Dr. Equator is a reproducible research scaffold for studying satellite detector
“snow” anomalies using GOES-16/18 ABI L1b scans, SWPC space-weather archives,
and EUMETSAT MSG/SEVIRI data. The analysis is intended to quantify data-quality
flags, local outliers, and dark-target noise, then produce figures for papers
and presentations.

## Repository layout

- `config/`: project settings, region of interest, bands, and event windows.
- `docs/`: science framing, off-disk masking diagnostics, and metric methodology.
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

## GOES ABI ingestion

Set the event `start`/`end` timestamps and all four `goes.roi_lat_lon` bounds
in `config/config.yaml`, then run:

```bash
python -m src.ingestion.goes_downloader --event event_1
```

The downloader uses anonymous access to the configured public NOAA GOES S3
bucket, crops each configured band to the geographic ROI, and writes compressed
per-scan NetCDF files under `data/processed/<event>/G16/<band>/` (or `G18`).
Existing cropped scans are skipped on subsequent runs. Full-disk downloads
are removed from `data/raw_tmp/` after each scan. For a small run, pass
`--max-scans 3` to process at most three scans per configured band; `--bucket`
can select another configured satellite bucket.

## Data flow

```text
S3 / EUMETSAT -> src/ingestion -> data/processed -> src/analysis
              -> data/exports -> src/visualization
```

The ingestion pipeline is intended to crop GOES scans to the configured ROI
before removing temporary full-disk downloads. Processed data should follow a
stable schema; exported arrays and tables are intended as inputs to figure
generation.

## Methodology

Because NOAA Level-1b ground processing masks off-disk deep-space pixels in
`ABI-L1b-RadF` (`_FillValue = 4095`, `DQF = -1`), Dr. Equator uses a dual-band
open-ocean dark-target strategy combining nightside `C02` ($0.64\ \mu\text{m}$
VNIR, $0.5\text{ km}$) and continuous `C13` ($10.3\ \mu\text{m}$ IR,
$2.0\text{ km}$) at native spatial resolution. Full details on DQF partitioning,
$>5\hat{\sigma}_{\text{MAD}}$ local-median spike detection, ROI selection, and
2024 SPE/baseline windows are documented in [`docs/methods.md`](docs/methods.md).
