import os, json
import numpy as np
import pandas as pd
from flask import current_app
from database import db
from models.preprocessing_report_model import PreprocessingReport
from models.dataset_model import Dataset
from services.dataset_service import read_dataframe
from scipy import stats as scipy_stats


def _load_df(dataset):
    ext = dataset.file_name.rsplit('.', 1)[1].lower()
    return read_dataframe(dataset.file_path, ext)


def _get_output_dir(dataset_id):
    base = current_app.config['PROCESSED_FOLDER']
    out = os.path.join(base, str(dataset_id))
    os.makedirs(out, exist_ok=True)
    return out


def _get_column_info(df):
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    categorical = list(df.select_dtypes(include=['object', 'category']).columns)
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(str(col))
    return numeric, categorical, date_cols


def handle_missing_values(df, method):
    before = int(df.isnull().sum().sum())
    result = {'method': method, 'columns_before': {}}
    for col in df.columns:
        c = int(df[col].isnull().sum())
        if c > 0:
            result['columns_before'][col] = c

    if method == 'drop':
        df = df.dropna().copy()
    elif method == 'mean':
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].fillna(df[numeric].mean())
    elif method == 'median':
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].fillna(df[numeric].median())
    elif method == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
    elif method == 'ffill':
        df = df.ffill().copy()
    elif method == 'bfill':
        df = df.bfill().copy()

    after = int(df.isnull().sum().sum())
    result['missing_before'] = before
    result['missing_after'] = after
    result['total_removed'] = before - after
    return df, result


def handle_outliers_iqr(df, columns):
    detected = 0
    removed = 0
    details = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        count = len(outliers)
        detected += count
        if count > 0:
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper
            removed += count
            details[col] = {'detected': count, 'lower': round(float(lower), 4), 'upper': round(float(upper), 4)}
    return df, {'method': 'iqr', 'columns': columns, 'total_detected': detected, 'total_removed': removed, 'details': details}


def handle_outliers_zscore(df, columns, threshold=3):
    detected = 0
    removed = 0
    details = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 4:
            continue
        z = np.abs(scipy_stats.zscore(s, nan_policy='omit'))
        outliers = s[z > threshold]
        count = len(outliers)
        detected += count
        if count > 0:
            mean_val = s.mean()
            std_val = s.std()
            lower = mean_val - threshold * std_val
            upper = mean_val + threshold * std_val
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper
            removed += count
            details[col] = {'detected': count, 'threshold': threshold}
    return df, {'method': 'zscore', 'threshold': threshold, 'columns': columns, 'total_detected': detected, 'total_removed': removed, 'details': details}


def encode_label(df, columns):
    mappings = {}
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        unique = df[col].unique()
        mapping = {v: i for i, v in enumerate(unique)}
        df[col] = df[col].map(mapping)
        mappings[col] = {str(k): int(v) for k, v in mapping.items()}
    return df, {'method': 'label', 'columns': columns, 'mappings': mappings}


def encode_onehot(df, columns):
    created_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        created_cols.extend(dummies.columns.tolist())
    return df, {'method': 'onehot', 'columns': columns, 'created_columns': created_cols}


def scale_standard(df, columns):
    details = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 2:
            continue
        mean_val = s.mean()
        std_val = s.std()
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
            details[col] = {'mean': round(float(mean_val), 4), 'std': round(float(std_val), 4)}
    return df, {'method': 'standard', 'columns': columns, 'details': details}


def scale_minmax(df, columns):
    details = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 2:
            continue
        min_val = s.min()
        max_val = s.max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            details[col] = {'min': round(float(min_val), 4), 'max': round(float(max_val), 4)}
    return df, {'method': 'minmax', 'columns': columns, 'details': details}


def scale_robust(df, columns):
    details = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 2:
            continue
        median_val = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            df[col] = (df[col] - median_val) / iqr
            details[col] = {'median': round(float(median_val), 4), 'iqr': round(float(iqr), 4)}
    return df, {'method': 'robust', 'columns': columns, 'details': details}


def create_date_features(df, date_col, features):
    created = []
    try:
        dt_series = pd.to_datetime(df[date_col])
    except Exception:
        return df, {'error': f'Cannot parse {date_col} as date'}

    feature_map = {
        'year': dt_series.dt.year,
        'month': dt_series.dt.month,
        'day': dt_series.dt.day,
        'week': dt_series.dt.isocalendar().week.astype(int) if hasattr(dt_series.dt, 'isocalendar') else dt_series.dt.week,
        'quarter': dt_series.dt.quarter,
        'dayofweek': dt_series.dt.dayofweek
    }

    for feat in features:
        if feat in feature_map:
            col_name = f'{date_col}_{feat}'
            df[col_name] = feature_map[feat]
            created.append(col_name)

    return df, {'date_column': date_col, 'features_created': created}


def save_processed_dataset(df, dataset_id, output_dir):
    csv_path = os.path.join(output_dir, 'processed_dataset.csv')
    xlsx_path = os.path.join(output_dir, 'processed_dataset.xlsx')
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path


def run_automatic_preprocessing(dataset_id, user_id):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, 'Dataset not found'
    df = _load_df(dataset)
    if df is None:
        return None, 'Unable to read dataset'

    output_dir = _get_output_dir(dataset_id)
    original_shape = list(df.shape)
    numeric, categorical, date_cols = _get_column_info(df)
    steps = {}

    if numeric:
        df, mv_result = handle_missing_values(df, 'median')
        steps['missing_values'] = mv_result
    else:
        df, mv_result = handle_missing_values(df, 'mode')
        steps['missing_values'] = mv_result

    if numeric:
        df, out_result = handle_outliers_iqr(df, numeric)
        steps['outliers'] = out_result

    if categorical:
        df, enc_result = encode_label(df, categorical)
        steps['encoding'] = enc_result

    if numeric:
        df, scale_result = scale_standard(df, numeric)
        steps['scaling'] = scale_result

    if date_cols:
        df, date_result = create_date_features(df, date_cols[0], ['year', 'month', 'day', 'week', 'quarter', 'dayofweek'])
        steps['date_features'] = date_result

    output_file = save_processed_dataset(df, dataset_id, output_dir)
    processed_shape = list(df.shape)

    report = PreprocessingReport(
        dataset_id=dataset_id,
        mode='automatic',
        steps_applied=json.dumps(steps),
        original_shape=f'{original_shape[0]}x{original_shape[1]}',
        processed_shape=f'{processed_shape[0]}x{processed_shape[1]}',
        output_file=output_file
    )
    db.session.add(report)
    db.session.commit()

    return {
        'report_id': report.id,
        'original_shape': original_shape,
        'processed_shape': processed_shape,
        'steps': steps,
        'output_file': output_file,
        'output_filename': 'processed_dataset.csv'
    }, None


def run_manual_preprocessing(dataset_id, user_id, config):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, 'Dataset not found'
    df = _load_df(dataset)
    if df is None:
        return None, 'Unable to read dataset'

    output_dir = _get_output_dir(dataset_id)
    original_shape = list(df.shape)
    numeric, categorical, date_cols = _get_column_info(df)
    steps = {}

    mv_method = config.get('missing_method')
    if mv_method and mv_method != 'none':
        df, mv_result = handle_missing_values(df, mv_method)
        steps['missing_values'] = mv_result

    outlier_method = config.get('outlier_method')
    outlier_cols = config.get('outlier_columns', [])
    if outlier_method and outlier_method != 'none' and outlier_cols:
        if outlier_method == 'iqr':
            df, out_result = handle_outliers_iqr(df, outlier_cols)
        elif outlier_method == 'zscore':
            df, out_result = handle_outliers_zscore(df, outlier_cols)
        steps['outliers'] = out_result

    enc_method = config.get('encoding_method')
    enc_cols = config.get('encoding_columns', [])
    if enc_method and enc_method != 'none' and enc_cols:
        if enc_method == 'label':
            df, enc_result = encode_label(df, enc_cols)
        elif enc_method == 'onehot':
            df, enc_result = encode_onehot(df, enc_cols)
        steps['encoding'] = enc_result

    scale_method = config.get('scaling_method')
    scale_cols = config.get('scaling_columns', [])
    if scale_method and scale_method != 'none' and scale_cols:
        if scale_method == 'standard':
            df, scale_result = scale_standard(df, scale_cols)
        elif scale_method == 'minmax':
            df, scale_result = scale_minmax(df, scale_cols)
        elif scale_method == 'robust':
            df, scale_result = scale_robust(df, scale_cols)
        steps['scaling'] = scale_result

    date_features = config.get('date_features', [])
    date_col_selected = config.get('date_column')
    if date_features and date_col_selected:
        df, date_result = create_date_features(df, date_col_selected, date_features)
        steps['date_features'] = date_result

    output_file = save_processed_dataset(df, dataset_id, output_dir)
    processed_shape = list(df.shape)

    report = PreprocessingReport(
        dataset_id=dataset_id,
        mode='manual',
        steps_applied=json.dumps(steps),
        original_shape=f'{original_shape[0]}x{original_shape[1]}',
        processed_shape=f'{processed_shape[0]}x{processed_shape[1]}',
        output_file=output_file
    )
    db.session.add(report)
    db.session.commit()

    return {
        'report_id': report.id,
        'original_shape': original_shape,
        'processed_shape': processed_shape,
        'steps': steps,
        'output_file': output_file,
        'output_filename': 'processed_dataset.csv'
    }, None


def get_preprocessing_report(dataset_id):
    return PreprocessingReport.query.filter_by(dataset_id=dataset_id).order_by(
        PreprocessingReport.created_at.desc()
    ).first()


def get_column_details(dataset_id, user_id):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, None, None, None
    df = _load_df(dataset)
    if df is None:
        return None, None, None, None
    numeric, categorical, date_cols = _get_column_info(df)
    missing_info = {}
    for col in df.columns:
        c = int(df[col].isnull().sum())
        if c > 0:
            missing_info[col] = c
    return numeric, categorical, date_cols, missing_info
