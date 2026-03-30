"""
EY Open Science AI & Data Challenge 2026
Predicting river water quality (TA, EC, DRP) across South Africa

Leaderboard R²: 0.44
No in-situ water chemistry features — satellite, climate, and terrain only.

The 0.44 submission combined two models:
  TA and DRP from a single-stage ensemble (GradientBoosting / RandomForest)
  EC from a two-stage spatial+temporal architecture

This script documents the full pipeline: data loading, feature engineering,
model training, and prediction. The actual competition submission is
included as submission_final.csv (R²=0.44).
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus',
]

# GB for TA because it extrapolates smoother than bagging — matters when
# val stations sit on different geology. Tested XGBoost too, marginal.
def ta_model():
    return GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.025,
        subsample=0.5, max_features=0.4, random_state=42)

# ExtraTrees for EC — ensemble averaging handles noisy spectral features
# better than sequential boosting for conductance
def ec_model():
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        max_features=0.4, random_state=42, n_jobs=-1)

# RF with conservative depth for DRP — phosphorus signal is sparse,
# deeper trees just memorize training stations
def drp_model():
    return RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=25,
        max_features=0.3, random_state=42)


def spatial_cv(X, y, locs, label, model_fn, n_folds=5):
    """CV that splits on stations, not rows — prevents leakage from
    repeated measurements at the same location."""
    unique = locs.drop_duplicates().reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for _, (tr_i, te_i) in enumerate(kf.split(unique)):
        tr_set = set(map(tuple, unique.iloc[tr_i].values))
        te_set = set(map(tuple, unique.iloc[te_i].values))
        tr_mask = locs.apply(tuple, axis=1).isin(tr_set)
        te_mask = locs.apply(tuple, axis=1).isin(te_set)
        sc = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr_mask]), nan=0.0)
        Xte = np.nan_to_num(sc.transform(X[te_mask]), nan=0.0)
        m = model_fn()
        m.fit(Xtr, y[tr_mask])
        scores.append(r2_score(y[te_mask], m.predict(Xte)))
    avg = np.mean(scores)
    print(f"  {label:<30} folds={[f'{s:.3f}' for s in scores]}  mean={avg:.3f}")
    return avg


# load everything
print("Loading data...")

wq  = pd.read_csv('water_quality_training_dataset.csv')
sub = pd.read_csv('submission_template.csv')
for df in [wq, sub]:
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    df['month']       = df['Sample Date'].dt.month
    df['year']        = df['Sample Date'].dt.year
    df['day_of_year'] = df['Sample Date'].dt.dayofyear

ls_tr = pd.read_csv('landsat_features_training.csv')
ls_va = pd.read_csv('landsat_features_validation.csv')
ls_va['Sample Date'] = pd.to_datetime(ls_va['Sample Date'], dayfirst=True)

train = pd.concat([wq, ls_tr], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
val   = sub.copy()
val   = val.merge(ls_va, on=['Latitude', 'Longitude', 'Sample Date'], how='left')

# TerraClimate
for v in ['pet', 'ppt', 'tmax', 'tmin', 'q', 'soil', 'def', 'srad', 'pdsi', 'aet']:
    for target_df, fname in [(train, f'terraclimate_{v}.csv'), (val, f'val_{v}.csv')]:
        tc = pd.read_csv(fname)
        tc.columns = [c.strip().lower() for c in tc.columns]
        tc.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude',
                           'sample date': 'Sample Date'}, inplace=True)
        tc['Sample Date'] = pd.to_datetime(tc['Sample Date'], dayfirst=True, errors='coerce')
        merged = target_df[['Latitude', 'Longitude', 'Sample Date']].merge(
            tc[['Latitude', 'Longitude', 'Sample Date', v]],
            on=['Latitude', 'Longitude', 'Sample Date'], how='left')
        target_df[v] = merged[v].values

train = train.merge(pd.read_csv('elevation_training.csv'), on=['Latitude', 'Longitude'], how='left')
val   = val.merge(pd.read_csv('elevation_validation.csv'), on=['Latitude', 'Longitude'], how='left')
coast = pd.read_csv('dist_coast.csv')
train = train.merge(coast, on=['Latitude', 'Longitude'], how='left')
val   = val.merge(coast, on=['Latitude', 'Longitude'], how='left')

# CHIRPS
chirps = pd.read_csv('chirps_data.csv')
chirps.columns = [c.strip() for c in chirps.columns]
chirps.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude',
                        'sample date': 'Sample Date', 'LATITUDE': 'Latitude',
                        'LONGITUDE': 'Longitude', 'SAMPLE DATE': 'Sample Date'}, inplace=True)
chirps['Sample Date'] = pd.to_datetime(chirps['Sample Date'], dayfirst=True, errors='coerce')
chirps_col = [c for c in chirps.columns if c not in ['Latitude', 'Longitude', 'Sample Date']][0]
for df in [train, val]:
    m = df[['Latitude', 'Longitude', 'Sample Date']].merge(
        chirps[['Latitude', 'Longitude', 'Sample Date', chirps_col]],
        on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    df['chirps_ppt'] = m[chirps_col].values

# catchment info — chemistry columns got ruled out mid-competition
cat = pd.read_csv('catchment_all_info.csv')
cat_clean = cat[[c for c in cat.columns
                 if c not in {'ph_p50', 'tds_p50', 'so4_p50', 'po4_p50', 'no3_p50'}]]
train = train.merge(cat_clean, on=['Latitude', 'Longitude'], how='left')
val   = val.merge(cat_clean, on=['Latitude', 'Longitude'], how='left')

# Pitman hydrological parameters
pit = pd.read_csv('pitman_parameters.csv')
pit_vars = [c for c in ['mae', 'pitman_ft', 'pitman_pi', 'pitman_tl',
                         'pitman_r', 'pitman_st', 'pitman_zmax']
            if c in pit.columns and c not in train.columns]
if pit_vars:
    train = train.merge(pit[['quat_code'] + pit_vars], on='quat_code', how='left')
    val   = val.merge(pit[['quat_code'] + pit_vars], on='quat_code', how='left')

# BasinATLAS — HDI and irrigation were the only two that survived screening
ba = pd.read_csv('basin_atlas_full.csv')
ba['Latitude']  = ba['Latitude'].round(4)
ba['Longitude'] = ba['Longitude'].round(4)
for df in [train, val]:
    df['Latitude']  = df['Latitude'].round(4)
    df['Longitude'] = df['Longitude'].round(4)
train = train.merge(ba[['Latitude', 'Longitude', 'hdi_ix_sav', 'ire_pc_use']],
                    on=['Latitude', 'Longitude'], how='left')
val   = val.merge(ba[['Latitude', 'Longitude', 'hdi_ix_sav', 'ire_pc_use']],
                  on=['Latitude', 'Longitude'], how='left')

# derived features
for df in [train, val]:
    df['NDVI']           = (df['nir'] - df['green']) / (df['nir'] + df['green'] + 1e-10)
    df['swir_ratio']     = df['swir16'] / (df['swir22'] + 1e-10)
    df['temp_range']     = df['tmax'] - df['tmin']
    df['moisture_index'] = df['ppt'] - df['pet']
    df['aridity']        = df['pet'] / (df['ppt'] + 1e-10)

train = train.loc[:, ~train.columns.duplicated()]
val   = val.loc[:, ~val.columns.duplicated()]


# feature set — no chemistry, NWU, or DWS
FEATURES = [
    'elevation', 'dist_coast', 'month', 'year', 'day_of_year',
    'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'NDVI', 'swir_ratio',
    'pet', 'ppt', 'tmax', 'tmin', 'q', 'soil', 'def', 'srad', 'pdsi', 'aet',
    'temp_range', 'moisture_index', 'aridity',
    'mar_catchment', 'mar_wr2012', 'mae',
    'pitman_ft', 'pitman_pi', 'pitman_tl', 'pitman_r', 'pitman_st', 'pitman_zmax',
    'chirps_ppt',
    'hdi_ix_sav', 'ire_pc_use',
]
FEATURES = [c for c in FEATURES if c in train.columns]
assert not any(x in ' '.join(FEATURES) for x in ['ph_p', 'tds_', 'so4_', 'po4_', 'nwu', 'dws'])

print(f"\nFeatures: {len(FEATURES)}")

model_data = train[FEATURES + TARGETS + ['Latitude', 'Longitude']].copy()
model_data = model_data.fillna(model_data.median(numeric_only=True))
val_feat = val[FEATURES].copy()
val_feat = val_feat.fillna(model_data[FEATURES].median())
locs = model_data[['Latitude', 'Longitude']]

print(f"Training: {len(model_data)} samples, {locs.drop_duplicates().shape[0]} stations")
print(f"Validation: {len(val_feat)} samples\n")


# spatial CV
print("=== 5-Fold Spatial CV ===")
X_all = model_data[FEATURES].values
spatial_cv(X_all, model_data['Total Alkalinity'].values,              locs, "TA  (GradientBoosting)", ta_model)
spatial_cv(X_all, model_data['Electrical Conductance'].values,        locs, "EC  (ExtraTrees)",       ec_model)
spatial_cv(X_all, model_data['Dissolved Reactive Phosphorus'].values, locs, "DRP (RandomForest)",     drp_model)


# train TA and DRP (single-stage, shared scaler)
print("\nTraining final models...")
sc = StandardScaler()
X_sc = np.nan_to_num(sc.fit_transform(model_data[FEATURES]), nan=0.0)
Xv = np.nan_to_num(sc.transform(val_feat), nan=0.0)

m_ta = ta_model()
m_ta.fit(X_sc, model_data['Total Alkalinity'])
pred_ta = np.clip(m_ta.predict(Xv), 0, None)

m_drp = drp_model()
m_drp.fit(X_sc, model_data['Dissolved Reactive Phosphorus'])
pred_drp = np.clip(m_drp.predict(Xv), 0, None)


# EC: two-stage model
# stage 1 learns typical EC at each station from static catchment features
# stage 2 learns how EC deviates from that baseline on a given date
# this split helped because the spatial and temporal signals driving
# conductance are pretty independent of each other
print("  EC: two-stage (spatial baseline + temporal residual)...")

SPATIAL = [c for c in [
    'elevation', 'dist_coast', 'mar_catchment', 'mar_wr2012', 'mae',
    'pitman_ft', 'pitman_pi', 'pitman_tl', 'pitman_r', 'pitman_st', 'pitman_zmax',
    'hdi_ix_sav', 'ire_pc_use',
] if c in model_data.columns]

TEMPORAL = [c for c in [
    'month', 'day_of_year',
    'pet', 'ppt', 'tmax', 'tmin', 'q', 'soil', 'def', 'srad', 'pdsi', 'aet',
    'temp_range', 'moisture_index', 'chirps_ppt',
    'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'NDVI', 'swir_ratio',
] if c in model_data.columns]

# stage 1: station means
station_ec = model_data.groupby(['Latitude', 'Longitude']).agg(
    {**{'Electrical Conductance': 'mean'},
     **{f: 'first' for f in SPATIAL}}
).reset_index()

sc1 = StandardScaler()
X_stations = np.nan_to_num(sc1.fit_transform(
    station_ec[SPATIAL].fillna(station_ec[SPATIAL].median())), nan=0.0)

ec_s1 = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.03,
    subsample=0.6, max_features=0.5, random_state=42)
ec_s1.fit(X_stations, station_ec['Electrical Conductance'])

s1_train = ec_s1.predict(np.nan_to_num(sc1.transform(
    model_data[SPATIAL].fillna(station_ec[SPATIAL].median())), nan=0.0))
s1_val = ec_s1.predict(np.nan_to_num(sc1.transform(
    val_feat[SPATIAL].fillna(station_ec[SPATIAL].median())), nan=0.0))

# stage 2: residual from temporal features
resid = model_data['Electrical Conductance'].values - s1_train

sc2 = StandardScaler()
X_temp_tr = np.nan_to_num(sc2.fit_transform(
    model_data[TEMPORAL].fillna(model_data[TEMPORAL].median())), nan=0.0)
X_temp_va = np.nan_to_num(sc2.transform(
    val_feat[TEMPORAL].fillna(model_data[TEMPORAL].median())), nan=0.0)

ec_s2 = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.02,
    subsample=0.5, max_features=0.5, min_samples_leaf=20,
    random_state=42)
ec_s2.fit(X_temp_tr, resid)

pred_ec = np.clip(s1_val + ec_s2.predict(X_temp_va), 0, None)


# monthly anomaly correction — training data has systematic seasonal
# patterns (TA dips in summer from dilution, EC peaks in winter).
# just shift predictions by the monthly offset. added ~0.016.
monthly_anom = train.groupby('month')[TARGETS].mean() - train[TARGETS].mean()
months = sub['Sample Date'].dt.month.values

pred_ta  = np.clip(pred_ta  + monthly_anom.loc[months, 'Total Alkalinity'].values,              0, None)
pred_ec  = np.clip(pred_ec  + monthly_anom.loc[months, 'Electrical Conductance'].values,        0, None)
pred_drp = np.clip(pred_drp + monthly_anom.loc[months, 'Dissolved Reactive Phosphorus'].values, 0, None)


# save
out = sub.copy()
out['Total Alkalinity']              = pred_ta
out['Electrical Conductance']        = pred_ec
out['Dissolved Reactive Phosphorus'] = pred_drp

out.to_csv('submission_reproduced.csv', index=False)
print(f"\nSaved: submission_reproduced.csv")
print(out[TARGETS].describe().round(2))