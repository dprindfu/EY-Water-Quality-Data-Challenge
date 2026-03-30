"""
Feature screening for the EY Water Quality Challenge.

Tests candidate features against the existing model with three checks:
distribution shift, geo-correlation, and per-target spatial CV impact.

Run directly: python feature_screening.py soilgrids_extra.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
)
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus',
]

MODEL_FNS = [
    lambda: GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.025,
        subsample=0.5, max_features=0.4, random_state=42),
    lambda: ExtraTreesRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        max_features=0.4, random_state=42, n_jobs=-1),
    lambda: RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=25,
        max_features=0.3, random_state=42),
]


def spatial_cv_single(X, y, locs, model_fn, n_folds=5):
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
    return np.mean(scores)


def check_distribution(train_vals, val_vals):
    tr = train_vals.dropna()
    va = val_vals.dropna()
    if len(tr) == 0 or len(va) == 0:
        return "NO_DATA"
    if va.std() < 0.01:
        return "CONSTANT_IN_VAL"
    in_range = ((va >= tr.quantile(0.05)) & (va <= tr.quantile(0.95))).mean()
    shift = abs(va.mean() - tr.mean()) / (tr.std() + 1e-6)
    if in_range < 0.5:
        return f"DANGER ({in_range:.0%} in train range, shift={shift:.1f}σ)"
    if in_range < 0.7:
        return f"CAUTION ({in_range:.0%} in range, shift={shift:.1f}σ)"
    return f"OK ({in_range:.0%} in range)"


def check_geo_correlation(df, col):
    lat_r = df[col].corr(df['Latitude'])
    lon_r = df[col].corr(df['Longitude'])
    if abs(lat_r) > 0.6 or abs(lon_r) > 0.6:
        return f"GEO_PROXY (lat={lat_r:.2f}, lon={lon_r:.2f})"
    return f"OK (lat={lat_r:.2f}, lon={lon_r:.2f})"


def test_feature_cv(model_data, base_features, new_values, col_name, locs):
    """Add one column, rerun spatial CV per target, return deltas."""
    md = model_data.copy()
    md[col_name] = new_values
    md[col_name] = md[col_name].fillna(md[col_name].median())

    results = {}
    for t, mfn, short in zip(TARGETS, MODEL_FNS, ['TA', 'EC', 'DRP']):
        X_base = md[base_features].fillna(md[base_features].median()).values
        X_new = md[base_features + [col_name]].fillna(
            md[base_features + [col_name]].median()).values
        cv_base = spatial_cv_single(X_base, md[t].values, locs, mfn)
        cv_new = spatial_cv_single(X_new, md[t].values, locs, mfn)
        results[short] = cv_new - cv_base
    return results


def screen_file(filepath, model_data, base_features, locs):
    """Load a CSV, merge to stations, run all checks on every numeric column."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    for old, new in [('latitude', 'Latitude'), ('longitude', 'Longitude'),
                     ('LATITUDE', 'Latitude'), ('LONGITUDE', 'Longitude')]:
        df.rename(columns={old: new}, inplace=True)

    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print(f"  {filepath}: no Latitude/Longitude columns found")
        return

    df['Latitude'] = df['Latitude'].round(4)
    df['Longitude'] = df['Longitude'].round(4)
    df = df.drop_duplicates(subset=['Latitude', 'Longitude'])

    # merge to training rows
    md = model_data.copy()
    md['Latitude'] = md['Latitude'].round(4)
    md['Longitude'] = md['Longitude'].round(4)
    merged = md[['Latitude', 'Longitude']].merge(df, on=['Latitude', 'Longitude'], how='left')

    # merge to val rows for distribution check
    val_locs = pd.read_csv('submission_template.csv')[['Latitude', 'Longitude']]
    val_locs['Latitude'] = val_locs['Latitude'].round(4)
    val_locs['Longitude'] = val_locs['Longitude'].round(4)
    val_merged = val_locs.merge(df, on=['Latitude', 'Longitude'], how='left')

    # find testable columns
    skip = {'Latitude', 'Longitude', 'Sample Date', 'geometry', 'quat_code',
            'station_id', 'HYBAS_ID'}
    chem = {'ph', 'tds', 'so4', 'po4', 'no3', 'alk', 'nwu', 'dws',
            'conductance', 'alkalinity', 'phosphate', 'phosphorus'}
    candidates = [
        c for c in merged.columns
        if c not in skip
        and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        and not any(kw in c.lower() for kw in chem)
        and merged[c].notna().sum() > 50
    ]

    if not candidates:
        print(f"  {filepath}: no usable numeric columns")
        return

    print(f"\nScreening {filepath} — {len(candidates)} columns")
    print("=" * 70)

    for col in candidates:
        print(f"\n  {col}")

        dist = check_distribution(merged[col], val_merged[col])
        print(f"    distribution: {dist}")
        if 'CONSTANT' in dist:
            print(f"    skipping — constant in val")
            continue

        temp = merged[['Latitude', 'Longitude', col]].dropna()
        if len(temp) > 20:
            geo = check_geo_correlation(temp, col)
            print(f"    geo-correlation: {geo}")

        print(f"    running spatial CV...")
        impact = test_feature_cv(
            model_data, base_features,
            merged[col].values[:len(model_data)],
            col, locs)

        for short, delta in impact.items():
            flag = " <-- worth testing" if delta > 0.003 else ""
            print(f"    {short}: {delta:+.4f}{flag}")


def load_data():
    """Rebuild the training data from pipeline.py's inputs."""
    wq = pd.read_csv('water_quality_training_dataset.csv')
    sub = pd.read_csv('submission_template.csv')
    for df in [wq, sub]:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
        df['month'] = df['Sample Date'].dt.month
        df['year'] = df['Sample Date'].dt.year
        df['day_of_year'] = df['Sample Date'].dt.dayofyear

    ls_tr = pd.read_csv('landsat_features_training.csv')
    ls_va = pd.read_csv('landsat_features_validation.csv')
    ls_va['Sample Date'] = pd.to_datetime(ls_va['Sample Date'], dayfirst=True)
    train = pd.concat([wq, ls_tr], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    val = sub.copy()
    val = val.merge(ls_va, on=['Latitude', 'Longitude', 'Sample Date'], how='left')

    for v in ['pet', 'ppt', 'tmax', 'tmin', 'q', 'soil', 'def', 'srad', 'pdsi', 'aet']:
        for target_df, fname in [(train, f'terraclimate_{v}.csv'), (val, f'val_{v}.csv')]:
            tc = pd.read_csv(fname)
            tc.columns = [c.strip().lower() for c in tc.columns]
            tc.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude',
                               'sample date': 'Sample Date'}, inplace=True)
            tc['Sample Date'] = pd.to_datetime(tc['Sample Date'], dayfirst=True, errors='coerce')
            m = target_df[['Latitude', 'Longitude', 'Sample Date']].merge(
                tc[['Latitude', 'Longitude', 'Sample Date', v]],
                on=['Latitude', 'Longitude', 'Sample Date'], how='left')
            target_df[v] = m[v].values

    train = train.merge(pd.read_csv('elevation_training.csv'), on=['Latitude', 'Longitude'], how='left')
    val = val.merge(pd.read_csv('elevation_validation.csv'), on=['Latitude', 'Longitude'], how='left')
    coast = pd.read_csv('dist_coast.csv')
    train = train.merge(coast, on=['Latitude', 'Longitude'], how='left')
    val = val.merge(coast, on=['Latitude', 'Longitude'], how='left')

    chirps = pd.read_csv('chirps_data.csv')
    chirps.columns = [c.strip() for c in chirps.columns]
    chirps.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude',
                            'sample date': 'Sample Date', 'LATITUDE': 'Latitude',
                            'LONGITUDE': 'Longitude', 'SAMPLE DATE': 'Sample Date'}, inplace=True)
    chirps['Sample Date'] = pd.to_datetime(chirps['Sample Date'], dayfirst=True, errors='coerce')
    cc = [c for c in chirps.columns if c not in ['Latitude', 'Longitude', 'Sample Date']][0]
    for df in [train, val]:
        m = df[['Latitude', 'Longitude', 'Sample Date']].merge(
            chirps[['Latitude', 'Longitude', 'Sample Date', cc]],
            on=['Latitude', 'Longitude', 'Sample Date'], how='left')
        df['chirps_ppt'] = m[cc].values

    cat = pd.read_csv('catchment_all_info.csv')
    cat_clean = cat[[c for c in cat.columns
                     if c not in {'ph_p50', 'tds_p50', 'so4_p50', 'po4_p50', 'no3_p50'}]]
    train = train.merge(cat_clean, on=['Latitude', 'Longitude'], how='left')
    val = val.merge(cat_clean, on=['Latitude', 'Longitude'], how='left')

    pit = pd.read_csv('pitman_parameters.csv')
    pit_vars = [c for c in ['mae', 'pitman_ft', 'pitman_pi', 'pitman_tl',
                             'pitman_r', 'pitman_st', 'pitman_zmax']
                if c in pit.columns and c not in train.columns]
    if pit_vars:
        train = train.merge(pit[['quat_code'] + pit_vars], on='quat_code', how='left')
        val = val.merge(pit[['quat_code'] + pit_vars], on='quat_code', how='left')

    ba = pd.read_csv('basin_atlas_full.csv')
    ba['Latitude'] = ba['Latitude'].round(4)
    ba['Longitude'] = ba['Longitude'].round(4)
    for df in [train, val]:
        df['Latitude'] = df['Latitude'].round(4)
        df['Longitude'] = df['Longitude'].round(4)
    train = train.merge(ba[['Latitude', 'Longitude', 'hdi_ix_sav', 'ire_pc_use']],
                        on=['Latitude', 'Longitude'], how='left')
    val = val.merge(ba[['Latitude', 'Longitude', 'hdi_ix_sav', 'ire_pc_use']],
                    on=['Latitude', 'Longitude'], how='left')

    for df in [train, val]:
        df['NDVI'] = (df['nir'] - df['green']) / (df['nir'] + df['green'] + 1e-10)
        df['swir_ratio'] = df['swir16'] / (df['swir22'] + 1e-10)
        df['temp_range'] = df['tmax'] - df['tmin']
        df['moisture_index'] = df['ppt'] - df['pet']
        df['aridity'] = df['pet'] / (df['ppt'] + 1e-10)

    train = train.loc[:, ~train.columns.duplicated()]
    val = val.loc[:, ~val.columns.duplicated()]

    features = (
        ['elevation', 'dist_coast', 'month', 'year', 'day_of_year',
         'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'NDVI', 'swir_ratio',
         'pet', 'ppt', 'tmax', 'tmin', 'q', 'soil', 'def', 'srad', 'pdsi', 'aet',
         'temp_range', 'moisture_index', 'aridity',
         'mar_catchment', 'mar_wr2012', 'mae',
         'pitman_ft', 'pitman_pi', 'pitman_tl', 'pitman_r', 'pitman_st', 'pitman_zmax',
         'chirps_ppt', 'hdi_ix_sav', 'ire_pc_use']
    )
    features = [c for c in features if c in train.columns]

    model_data = train[features + TARGETS + ['Latitude', 'Longitude']].copy()
    model_data = model_data.fillna(model_data.median(numeric_only=True))
    locs = model_data[['Latitude', 'Longitude']]

    return model_data, locs, features


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python feature_screening.py <candidate.csv> [candidate2.csv ...]")
        sys.exit(1)

    print("Loading pipeline data...")
    model_data, locs, features = load_data()
    print(f"Ready: {len(model_data)} samples, {len(features)} base features\n")

    for path in sys.argv[1:]:
        screen_file(path, model_data, features, locs)