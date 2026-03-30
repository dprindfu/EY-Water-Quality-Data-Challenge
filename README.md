# Predicting River Water Quality Across South Africa

Geospatial ML pipeline for predicting three water quality indicators — Total Alkalinity (TA), Electrical Conductance (EC), and Dissolved Reactive Phosphorus (DRP) — at ungauged river monitoring stations using only publicly available satellite, climate, and terrain data.

Built for the **2026 EY Open Science AI & Data Challenge**. Placed in the top 10% out of 300+ teams internationally. Leaderboard R² of 0.44 using zero in-situ water chemistry features.

---

## The Problem

South Africa monitors water quality at ~186 river stations, but predictions are needed at locations with no historical chemistry data. Given only a station's coordinates and sample date, predict three targets that are fundamentally driven by geology and land use — information the model has to infer indirectly from remote sensing and terrain.

The core difficulty is **spatial extrapolation**. All validation stations sit in the Eastern Cape, geographically separated from training data. Standard random CV gives misleadingly high scores; the real test is whether the model generalizes to entirely unseen catchments with different geology.

## Approach

The problem splits naturally into *where* a station is and *when* a sample was taken:

**Spatial signal** — Features that don't change over time: elevation, distance to coast, catchment hydrology (Pitman model parameters, mean annual runoff), Human Development Index, and irrigation extent. These set the baseline chemistry at each location.

**Temporal signal** — Date-specific features: Landsat 8 surface reflectance bands and indices, TerraClimate monthly climate variables, and CHIRPS rainfall. These capture seasonal dilution and concentration dynamics.

Each target gets its own model because the three indicators respond to different physical drivers:

- **TA → Gradient Boosting.** Alkalinity is geology-dominated. GB's sequential residual fitting extrapolates more smoothly than bagging methods when the validation geology differs from training.
- **EC → Two-stage architecture.** Stage 1 learns the spatial baseline (typical EC at each station from geology and catchment features, trained on station means). Stage 2 learns the temporal residual (how EC deviates from the baseline on a given date, using climate and spectral features). This separation improved EC by ~0.02 over a single-stage ExtraTrees because it decouples the two fundamentally different signals.
- **DRP → Random Forest (conservative depth).** Phosphorus signal is sparse and noisy. Deeper trees just memorized training stations. Soil chemistry features consistently hurt DRP in both CV and pseudo evaluation, confirming it's driven more by land use and rainfall than bedrock.

A **monthly anomaly correction** adds the training set's average seasonal offset per month to each prediction. Simple, but worth ~0.016 on the leaderboard — alkalinity dips in summer from dilution, conductance peaks in winter from concentration.

## Feature Engineering

Started with ~25 features, systematically screened candidates from 10+ public datasets, and landed on 38 features in the final model. Every addition had to pass three checks: physically motivated relationship to water chemistry, no train→val distribution collapse, and confirmed improvement on held-out-station spatial CV.

| Source | Features | Why |
|--------|----------|-----|
| Landsat 8 SR | NIR, Green, SWIR bands, NDMI, MNDWI, NDVI, SWIR ratio | Turbidity, vegetation cover, soil moisture proxies |
| TerraClimate | PET, precip, temp, runoff, soil moisture, deficit, solar radiation, PDSI, AET | Seasonal climate drivers of dilution and evapotranspiration |
| CHIRPS | Daily rainfall | Short-term dilution signal, finer resolution than TerraClimate |
| SRTM / Copernicus | Elevation, distance to coast | Terrain gradient and marine aerosol influence |
| WR2012 + Pitman | Mean annual runoff, 7 Pitman catchment parameters | Catchment-scale hydrology encodes geology indirectly |
| BasinATLAS | HDI, irrigation extent | Human development proxies for urbanization and water infrastructure |

Datasets tested but excluded: ERA5 soil moisture (hurt leaderboard despite strong CV signal), HWSD soil properties (helped TA in isolation but didn't improve the combined submission), MODIS NDVI (redundant with Landsat), full BasinATLAS lithology/land cover (near-constant in val region — caused overfitting), SoilGrids API properties (intermittent availability), JRC water occurrence, water reflectance indices, city distances, WWTP proximity.

## Results

| Metric | Score |
|--------|-------|
| **Leaderboard R²** | **0.44** |
| Spatial CV (5-fold, stations held out) | 0.30 |

The CV-to-leaderboard gap is expected — spatial CV deliberately holds out entire station clusters, making it more pessimistic than the leaderboard's specific geographic split.

**What moved the needle** (cumulative leaderboard impact):
- Pitman hydrological parameters: +0.04
- BasinATLAS HDI + irrigation: +0.02
- Two-stage EC architecture: +0.02
- Monthly anomaly correction: +0.016

**What didn't work:**
- KNN and spatial mixture-of-experts — too few training stations near the val region
- Target transforms (log, Box-Cox, sqrt) — neutral or slightly negative
- Regressor chains (EC→TA→DRP) — error propagation outweighed the cross-target signal
- Feature removal / lean models — the model was underfitting, not overfitting

## Repo Structure

```
├── pipeline.py                # Full training + prediction pipeline
├── feature_screening.py       # Framework for evaluating candidate features
├── submission_final.csv       # Actual competition submission (0.44 R²)
├── data/                      # Input CSVs (not included — see Data Sources below)
└── README.md
```

## Data Sources

All features derived from publicly available datasets:

- [Landsat 8 Surface Reflectance](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) — USGS via Google Earth Engine
- [TerraClimate](https://www.climatologylab.org/terraclimate.html) — Abatzoglou et al., University of Idaho
- [CHIRPS](https://www.chc.ucsb.edu/data/chirps) — Climate Hazards Group, UC Santa Barbara
- [SRTM / Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) — NASA / ESA
- [WR2012](http://waterresourceswr2012.co.za/) — South Africa Water Resources 2012 study
- [BasinATLAS](https://www.hydrosheds.org/hydroatlas) — WWF HydroSHEDS
- [HWSD v2.0](https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/) — FAO (tested, not used in final model)

Training labels provided by EY as part of the competition.

## Setup

```bash
pip install numpy pandas scikit-learn lightgbm
```

Place input CSVs in the working directory, then:

```bash
python pipeline.py
```

Prints spatial CV scores and saves predictions to `submission_reproduced.csv`. The actual competition submission is `submission_final.csv`.

## Tech Stack

Python · scikit-learn · LightGBM · pandas · NumPy · GeoPandas · rasterio · Google Earth Engine

---

*Daniel Rindfuss — 2026 EY Open Science AI & Data Challenge*