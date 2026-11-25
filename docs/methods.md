# Science Framing & Methodology

## 1. Objective

Dr. Equator investigates transient single-pixel radiometric anomalies ("detector snow" caused by energetic particle single-event upsets, SEUs) in geostationary meteorological imagers during intense Solar Proton Events (SPEs) of Solar Cycle 25. By pairing native-resolution GOES-16 ($75.2^\circ\text{W}$) and GOES-18 ($137.2^\circ\text{W}$) ABI Level-1b full-disk radiance scans (`ABI-L1b-RadF`) with 1-minute GOES Space Weather Prediction Center (SWPC / NCEI L2) X-ray (`XRS`) and energetic proton (`SGPS`) fluxes, the pipeline tests:

1. Whether operational Level-1b Data Quality Flags (`DQF != 0`) capture all proton-induced pixel hits or leave unflagged bright spikes in `DQF == 0` science pixels.
2. How unflagged $>5\hat{\sigma}_{\text{MAD}}$ single-pixel outlier rates scale with $>10\text{ MeV}$, $>50\text{ MeV}$, and penetrating $>100\text{ MeV}$ integral proton fluxes across two co-observing geostationary platforms.

---

## 2. Empirical Diagnostic: Off-Disk Space Masking (Plan B Guardrail)

### Initial Hypothesis vs. Level-1b Ground Processing Reality
An intuitive baseline for isolating pure focal-plane readout and radiation noise is to sample deep-space pixels outside Earth's limb in full-disk (`RadF`) frames. Direct inspection of GOES-16 `ABI-L1b-RadF` scans from May 10, 2024 (`2024/131/03/`, `03:00 UTC`) across Band 13 (`C13`, $10.3\ \mu\text{m}$ IR) and Band 2 (`C02`, $0.64\ \mu\text{m}$ VNIR) demonstrated that off-disk space pixels contain **no usable detector noise** in published L1b products:

- **Decoded (`mask_and_scale=True`):** 100% of off-disk corner pixels (`Rad[0:50, 0:50]`) decode to `NaN`, with `DQF = NaN`.
- **Raw unscaled (`mask_and_scale=False`):** 100% of off-disk corner pixels hold integer sentinel `_FillValue = 4095` (`int16`) and `DQF = -1` (`_FillValue = -1`).

Because NOAA's operational L1b ground segment deterministically masks off-disk space views prior to dissemination, deep-space corners cannot be used to monitor radiation-induced dark-current or SEU rates.

### Dual-Band On-Disk Dark-Target Strategy
To replace off-disk space views without introducing false positives from sharp cloud or coastline gradients, we adopt a complementary **dual-band open-ocean dark-target design**:

| Channel | Wavelength & Type | Native Grid (Full Disk) | Role in Anomaly Detection |
| :--- | :--- | :--- | :--- |
| **`C02`** (Band 2) | $0.64\ \mu\text{m}$ Visible/VNIR | $0.5\text{ km}$ $(21696 \times 21696)$ | **Primary nightside dark target.** Absence of solar illumination yields a zero-centered, scene-free detector noise baseline. Diagnostic verification on a $200 \times 200$ equatorial Pacific box ($40{,}000$ pixels at `03:00 UTC`, May 10, 2024) confirmed 100% finite `DQF = 0` pixels with $\text{median} = 0.0099$, $\sigma = 0.5518$, and range $[-1.893, 2.072]$. Daytime solar contamination is excluded automatically whenever spatial dispersion exceeds $\sigma_{\text{Rad}} \ge 2.5$ (`is_nightside = False`). |
| **`C13`** (Band 13) | $10.3\ \mu\text{m}$ Clean IR Window | $2.0\text{ km}$ $(5424 \times 5424)$ | **Continuous 24-hour thermal reference.** Provides uninterrupted 10-minute cadence across both daytime and nighttime intervals over thermally uniform tropical ocean, where smooth sea-surface temperature (SST) gradients are removed by local spatial high-pass filtering. |

---

## 3. Native-Grid Preservation, DQF Accounting, and Spike Detection

### Multi-Resolution Handling
`C02` ($0.5\text{ km}$) has $4\times$ higher linear spatial resolution—and **$16\times$ more pixels per geographic degree**—than `C13` ($2.0\text{ km}$). Cropped ROI subsets are stored and analyzed strictly on their **native ABI fixed-grid `(y, x)` scan-angle coordinates** without spatial regridding or resampling, preventing $4 \times 4$ averaging from diluting single-detector-element SEU spikes in `C02`.

### Data Quality Flag (`DQF`) Partitioning
For each cropped scan:
1. **Off-disk exclusion:** Pixels with non-finite `DQF` or `DQF < 0` (`-1` space fill) are excluded; remaining pixels define valid on-disk count $N_{\text{valid}}$.
2. **Operational flag tracking:** The overall flagged fraction (`dqf_nonzero_frac`) and individual flag-code fractions (`dqf_1_frac` through `dqf_4_frac`, covering conditionally usable, out-of-range, no-value, and focal-plane temperature threshold exceedance codes) are recorded over $N_{\text{valid}}$.
3. **Pre-filter masking:** Only finite `DQF == 0` pixels ($N_{\text{good}}$) are eligible for unflagged snow detection. Before applying the $3 \times 3$ spatial filter, non-`DQF == 0` pixels are filled with the scan's `DQF == 0` scene median so saturated or flagged defects cannot bleed across $3 \times 3$ neighborhoods.

### Robust $3 \times 3$ Median High-Pass & $>5\hat{\sigma}_{\text{MAD}}$ Threshold
On the median-filled radiance array $\tilde{R}$, we compute the local residual against a $3 \times 3$ neighborhood median filter:

$$r_{i,j} = \tilde{R}_{i,j} - \operatorname{Median}_{3 \times 3}(\tilde{R})_{i,j}$$

Restricting $r$ to `DQF == 0` pixels ($r_{\text{good}}$) with median $\tilde{r} = \operatorname{median}(r_{\text{good}})$, background high-frequency sensor/scene noise is estimated via the normalized Median Absolute Deviation (MAD):

$$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \operatorname{median}\!\left(|r_{\text{good}} - \tilde{r}|\right)$$

Positive radiation-induced spikes are counted where $r_{\text{good}} - \tilde{r} > 5\,\hat{\sigma}_{\text{MAD}}$ (`outlier_count_5sig`). To compare `C02` ($16\times$ denser pixel grid) directly against `C13` and across slightly differing satellite viewing footprints, counts are normalized to parts per million (ppm) of unflagged pixels:

$$\text{Outlier Rate (ppm)} = \frac{N_{>5\sigma}}{N_{\text{good}}} \times 10^6$$

---

## 4. Region of Interest (ROI) Geometry

The geographic ROI is locked in `config/config.yaml` to **`lat: [-5.0°, +5.0°], lon: [-110.0°, -100.0°]`** in the eastern-central equatorial Pacific:

- **Landmass and island avoidance:** Shifting west of $[-95^\circ, -85^\circ]$ excludes the Galápagos Archipelago ($1.4^\circ\text{S}–1.7^\circ\text{N}$, $92.0^\circ\text{W}–89.2^\circ\text{W}$), volcanic orographic cloud wakes, and South American coastal upwelling fronts, keeping every pixel $>1{,}000\text{ km}$ from land.
- **Symmetric dual-satellite viewing zenith angle:** Centered at $105^\circ\text{W}$, the ROI lies nearly midway between GOES-16 East ($75.2^\circ\text{W}$, $|\Delta\lambda| \approx 30^\circ$) and GOES-18 West ($137.2^\circ\text{W}$, $|\Delta\lambda| \approx 32^\circ$), matching fixed-grid scan angles ($|x| \in [0.073, 0.104]\text{ rad}$) and optical path lengths across both instruments.
- **Nightside alignment with SPE maxima:** Local solar midnight at $105^\circ\text{W}$ occurs at `07:00 UTC`, placing the ~`01:45–12:15 UTC` dark-sky window directly across the May 11, 2024 `02:00–10:00 UTC` Ground Level Enhancement (GLE #74) and S2–S3 proton peak.

---

## 5. Event & Baseline Windows

All four primary analysis intervals span **24 hours (144 full-disk 10-minute ABI cadences)** in UTC:

| Key | UTC Start | UTC End | Space-Weather Context |
| :--- | :--- | :--- | :--- |
| **`event_1`** | `2024-05-10T18:00:00Z` | `2024-05-11T18:00:00Z` | **AR3664 May 2024 Superstorm & GLE #74:** Captures the sharp $\ge 100\text{ MeV}$ relativistic proton onset (`02:00–10:00 UTC` on May 11) and S2–S3 $\ge 10\text{ MeV}$ peak centered near `2024-05-11T06:00Z`. |
| **`event_2`** | `2024-05-14T12:00:00Z` | `2024-05-15T12:00:00Z` | **AR3664 X8.7 Flare & Secondary SEP Surge:** Covers the May 14 `16:51 UTC` X8.7 solar flare (western limb, magnetically well-connected) and subsequent high-energy proton enhancement. |
| **`event_3`** | `2024-06-08T00:00:00Z` | `2024-06-09T00:00:00Z` | **AR3697 June 2024 S3 Radiation Storm:** Driven by the returning AR3664 active region (M9.6 flare), reaching ~1,030 pfu ($\ge 10\text{ MeV}$)—the highest integral proton flux of 2024. |
| **`baseline`** | `2024-05-01T18:00:00Z` | `2024-05-02T18:00:00Z` | **Pre-Storm Solar-Quiet Control:** Matched early-May solar declination, seasonal SST structure, and satellite viewing geometry prior to AR3664's eruptive sequence, establishing the nominal cosmic-ray and sensor false-alarm floor. |

---

## 6. Space-Weather Ground Truth & Statistical Controls

- **X-ray and energetic proton drivers (`src/ingestion/fetch_swpc_flux.py`):** 1-minute GOES-16 L2 `xrsf-l2-avg1m_science` (`xrsa_flux`, `xrsb_flux` gated on `xrsb_flag == 0`) and `sgps-l2-avg1m` proton fluxes (`proton_gt10`, `proton_gt50`, `proton_gt100`, reconstructed by integrating differential channels above $10$ and $50\text{ MeV}$ onto the $+X/-X$ telescope-averaged $>100\text{ MeV}$ integral channel).
- **Autocorrelation-aware inference (`src/analysis/stats.py`):** Because magnetospheric particle fluxes and 10-minute cloud/radiance series exhibit strong serial autocorrelation, naive *i.i.d.* *p*-values inflate significance. Associations and event-minus-baseline differences are evaluated with lag cross-correlation, moving block bootstrap confidence intervals, and circular time-shift permutation tests that preserve exact power spectra.
