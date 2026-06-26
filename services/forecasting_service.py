import os, json
import numpy as np
import pandas as pd
from flask import current_app
from database import db
from models.forecast_report_model import ForecastReport
from models.dataset_model import Dataset
from models.preprocessing_report_model import PreprocessingReport
from services.dataset_service import read_dataframe

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SARIMAX = True
except ImportError:
    _HAS_SARIMAX = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_HW = True
except ImportError:
    _HAS_HW = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

_HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KERAS_LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except ImportError:
    pass


FORECAST_FOLDER = None


def _get_forecast_dir(dataset_id):
    global FORECAST_FOLDER
    if FORECAST_FOLDER is None:
        FORECAST_FOLDER = os.path.join(
            current_app.config.get('PROCESSED_FOLDER', 'data/processed'),
            '..', 'forecasts'
        )
        FORECAST_FOLDER = os.path.normpath(FORECAST_FOLDER)
    out = os.path.join(FORECAST_FOLDER, str(dataset_id))
    os.makedirs(out, exist_ok=True)
    return out


def _load_data(dataset_id, user_id):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, None, 'Dataset not found'

    prep = PreprocessingReport.query.filter_by(dataset_id=dataset_id).order_by(
        PreprocessingReport.created_at.desc()
    ).first()
    if prep and prep.output_file and os.path.exists(prep.output_file):
        file_path = prep.output_file
        ext = 'csv'
    else:
        file_path = dataset.file_path
        ext = dataset.file_name.rsplit('.', 1)[1].lower()

    df = read_dataframe(file_path, ext)
    if df is None:
        return None, None, 'Unable to read dataset'
    return dataset, df, None


def _is_date_column(series, threshold=0.7):
    if len(series) == 0:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    s = series.dropna()
    if len(s) == 0:
        return False
    try:
        parsed = pd.to_datetime(s, errors='coerce')
        rate = parsed.notna().sum() / len(s)
        return rate >= threshold
    except Exception:
        return False


def _is_numeric_timestamp(series, threshold=0.7):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna()
    if len(s) == 0:
        return False
    mean_val = abs(s.mean())
    if mean_val < 1e8 or mean_val > 1e16:
        return False
    for unit in ('s', 'ms', 'us', 'ns'):
        try:
            parsed = pd.to_datetime(s, unit=unit, errors='coerce')
            rate = parsed.notna().sum() / len(s)
            if rate >= threshold:
                return True
        except Exception:
            continue
    return False


COMMON_DATE_NAMES = {
    'date', 'dates', 'timestamp', 'time', 'datetime', 'ds',
    'time_stamp', 'date_time', 'order_date', 'transaction_date',
    'created_at', 'updated_at', 'orderdate', 'transdate',
    'date_created', 'date_updated', 'effective_date',
    'expiration_date', 'start_date', 'end_date', 'period',
    'year_month', 'month_year', 'date_key', 'calendar_date'
}


def _detect_date_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        lower = col.lower().replace(' ', '_').replace('-', '_')
        if lower in COMMON_DATE_NAMES:
            if _is_date_column(df[col]):
                return col
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            if _is_date_column(df[col]):
                return col
    for col in df.columns:
        if _is_numeric_timestamp(df[col]):
            return col
    return None


def _detect_target_column(df, date_col):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col in numeric:
        numeric.remove(date_col)
    if not numeric:
        return None
    best_col = numeric[0]
    best_var = -1
    for col in numeric:
        s = df[col].dropna()
        if len(s) > 0:
            var_val = s.var()
            if var_val > best_var:
                best_var = var_val
                best_col = col
    return best_col


def _get_numeric_features(df, date_col, target_col):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {date_col, target_col}
    return [c for c in numeric if c not in exclude]


def _create_date_features_df(df, date_col):
    if pd.api.types.is_numeric_dtype(df[date_col]):
        return None
    result = pd.DataFrame(index=df.index)
    try:
        dates = pd.to_datetime(df[date_col])
    except Exception:
        return None
    result['year'] = dates.dt.year
    result['month'] = dates.dt.month
    result['day'] = dates.dt.day
    result['dayofweek'] = dates.dt.dayofweek
    result['quarter'] = dates.dt.quarter
    result['dayofyear'] = dates.dt.dayofyear
    result['weekofyear'] = dates.dt.isocalendar().week.astype(int)
    result['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    return result


def _apply_train_test_split(df, target_col, test_ratio=0.2):
    df_sorted = df.sort_values('date_idx').reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * (1 - test_ratio))
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    if len(test) < 2:
        test = df_sorted.iloc[max(0, split_idx - 2):]
        train = df_sorted.iloc[:max(0, split_idx - 2)]
    return train, test


def _evaluate(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {'mae': None, 'rmse': None, 'mape': None, 'r2': None}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'mape': round(mape, 4), 'r2': round(r2, 4)}


def _forecast_future_dates(last_date, horizon):
    if isinstance(last_date, (int, float, np.integer, np.floating)):
        return list(range(int(last_date) + 1, int(last_date) + horizon + 1))
    try:
        last_date = pd.Timestamp(last_date)
    except Exception:
        last_date = pd.Timestamp.now()
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    return dates


def _train_arima(train_vals, test_vals, **kwargs):
    if not _HAS_STATSMODELS:
        return None, 'statsmodels not available'
    try:
        order = kwargs.get('order', (1, 1, 1))
        model = ARIMA(train_vals, order=order)
        fitted = model.fit()
        test_pred = fitted.forecast(steps=len(test_vals))
        future_pred = fitted.forecast(steps=kwargs.get('horizon', 30))
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'ARIMA',
                'params': {'order': list(order)}}, None
    except Exception as e:
        return None, str(e)


def _train_sarima(train_vals, test_vals, **kwargs):
    if not _HAS_SARIMAX:
        return None, 'SARIMAX not available'
    try:
        order = kwargs.get('order', (1, 1, 1))
        s_order = kwargs.get('seasonal_order', (1, 1, 1, 12))
        model = SARIMAX(train_vals, order=order, seasonal_order=s_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)
        test_pred = fitted.forecast(steps=len(test_vals))
        future_pred = fitted.forecast(steps=kwargs.get('horizon', 30))
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'SARIMA',
                'params': {'order': list(order),
                           'seasonal_order': list(s_order)}}, None
    except Exception as e:
        return None, str(e)


def _train_ets(train_vals, test_vals, **kwargs):
    if not _HAS_HW:
        return None, 'ExponentialSmoothing not available'
    try:
        seasonal_periods = kwargs.get('seasonal_periods', 12)
        if len(train_vals) < seasonal_periods * 2:
            seasonal_periods = min(seasonal_periods, max(2, len(train_vals) // 2))
        model = ExponentialSmoothing(train_vals, seasonal_periods=seasonal_periods,
                                     trend='add', seasonal='add')
        fitted = model.fit()
        test_pred = fitted.forecast(steps=len(test_vals))
        future_pred = fitted.forecast(steps=kwargs.get('horizon', 30))
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'ExponentialSmoothing',
                'params': {'seasonal_periods': seasonal_periods}}, None
    except Exception as e:
        try:
            model = ExponentialSmoothing(train_vals, trend='add', seasonal=None)
            fitted = model.fit()
            test_pred = fitted.forecast(steps=len(test_vals))
            future_pred = fitted.forecast(steps=kwargs.get('horizon', 30))
            return {'test_predictions': test_pred.tolist(),
                    'future_predictions': future_pred.tolist(),
                    'model': 'ExponentialSmoothing',
                    'params': {'seasonal_periods': None}}, None
        except Exception as e2:
            return None, str(e2)


def _train_linear_regression(X_train, y_train, X_test, X_future, **kwargs):
    if not _HAS_SKLEARN:
        return None, 'scikit-learn not available'
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        future_pred = model.predict(X_future) if X_future is not None else []
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'LinearRegression',
                'params': {}}, None
    except Exception as e:
        return None, str(e)


def _train_random_forest(X_train, y_train, X_test, X_future, **kwargs):
    if not _HAS_SKLEARN:
        return None, 'scikit-learn not available'
    try:
        model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        future_pred = model.predict(X_future) if X_future is not None else []
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'RandomForest',
                'params': {'n_estimators': 100, 'max_depth': 10}}, None
    except Exception as e:
        return None, str(e)


def _train_xgboost(X_train, y_train, X_test, X_future, **kwargs):
    if not _HAS_XGB:
        return None, 'XGBoost not available'
    try:
        model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        future_pred = model.predict(X_future) if X_future is not None else []
        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred.tolist(),
                'model': 'XGBoost',
                'params': {'n_estimators': 100, 'max_depth': 6,
                           'learning_rate': 0.1}}, None
    except Exception as e:
        return None, str(e)


def _train_lstm(train_vals, test_vals, **kwargs):
    if not _HAS_TF:
        return None, 'TensorFlow/Keras not available'
    try:
        lookback = kwargs.get('lookback', min(10, len(train_vals) // 3))
        horizon = kwargs.get('horizon', 30)

        def _create_sequences(data, lb):
            X, y = [], []
            for i in range(lb, len(data)):
                X.append(data[i - lb:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        train_arr = np.array(train_vals).reshape(-1, 1)
        test_arr = np.array(test_vals).reshape(-1, 1)
        full_arr = np.concatenate([train_arr, test_arr])

        if len(train_arr) <= lookback:
            lookback = max(1, len(train_arr) - 1)

        X_train, y_train = _create_sequences(train_arr, lookback)
        X_full, y_full = _create_sequences(full_arr, lookback)

        model = Sequential([
            KERAS_LSTM(50, activation='relu', input_shape=(lookback, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=16,
                  callbacks=[early_stop], verbose=0)

        split_point = len(train_arr) - lookback
        X_test_seq = X_full[split_point:]
        test_pred = model.predict(X_test_seq, verbose=0).flatten()

        future_input = full_arr[-lookback:].reshape(1, lookback, 1)
        future_pred = []
        for _ in range(horizon):
            next_val = model.predict(future_input, verbose=0)[0, 0]
            future_pred.append(float(next_val))
            future_input = np.roll(future_input, -1, axis=1)
            future_input[0, -1, 0] = next_val

        return {'test_predictions': test_pred.tolist(),
                'future_predictions': future_pred,
                'model': 'LSTM',
                'params': {'lookback': lookback, 'layers': 'LSTM(50)->Dropout->Dense(1)'}}, None
    except Exception as e:
        return None, str(e)


TRADITIONAL_MODELS = {
    'ARIMA': _train_arima,
    'SARIMA': _train_sarima,
    'ExponentialSmoothing': _train_ets
}

ML_MODELS = {
    'LinearRegression': _train_linear_regression,
    'RandomForest': _train_random_forest,
    'XGBoost': _train_xgboost
}

DL_MODELS = {
    'LSTM': _train_lstm
}

ALL_MODELS = {}
ALL_MODELS.update(TRADITIONAL_MODELS)
ALL_MODELS.update(ML_MODELS)
ALL_MODELS.update(DL_MODELS)


def _run_traditional_model(name, train_vals, test_vals, horizon):
    fn = TRADITIONAL_MODELS.get(name)
    if not fn:
        return None, f'Unknown model: {name}'
    return fn(train_vals, test_vals, horizon=horizon)


def _run_ml_model(name, X_train, y_train, X_test, X_future):
    fn = ML_MODELS.get(name)
    if not fn:
        return None, f'Unknown model: {name}'
    return fn(X_train, y_train, X_test, X_future)


def get_forecast_report(dataset_id):
    return ForecastReport.query.filter_by(dataset_id=dataset_id).order_by(
        ForecastReport.created_at.desc()
    ).first()


def _get_results_path(dataset_id):
    d = _get_forecast_dir(dataset_id)
    return os.path.join(d, 'forecast_results.json')


def save_forecast_results(dataset_id, results):
    path = _get_results_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_data = {}
    for k, v in results.items():
        if k == 'forecast_dir':
            continue
        try:
            json.dumps(v)
            save_data[k] = v
        except (TypeError, ValueError):
            save_data[k] = str(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)


def load_forecast_results(dataset_id):
    path = _get_results_path(dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_automatic_forecasting(dataset_id, user_id, horizon=30, test_ratio=0.2):
    dataset, df, error = _load_data(dataset_id, user_id)
    if error:
        return None, error

    date_col = _detect_date_column(df)
    if not date_col:
        date_col = '__row_index__'

    target_col = _detect_target_column(df, date_col)
    if not target_col:
        return None, 'No numeric target column detected. Please use manual mode.'

    return _run_forecasting_pipeline(dataset, df, date_col, target_col, None,
                                     horizon, test_ratio, auto_mode=True)


def run_manual_forecasting(dataset_id, user_id, target_col, date_col,
                           model_name=None, horizon=30, test_ratio=0.2):
    dataset, df, error = _load_data(dataset_id, user_id)
    if error:
        return None, error

    if not date_col:
        date_col = '__row_index__'

    if date_col != '__row_index__' and date_col not in df.columns:
        return None, f'Date column "{date_col}" not found in dataset.'
    if target_col not in df.columns:
        return None, f'Target column "{target_col}" not found in dataset.'

    return _run_forecasting_pipeline(dataset, df, date_col, target_col,
                                     model_name, horizon, test_ratio, auto_mode=False)


def _run_forecasting_pipeline(dataset, df, date_col, target_col, model_name,
                              horizon, test_ratio, auto_mode=False):
    forecast_dir = _get_forecast_dir(dataset.id)
    use_row_index = (date_col == '__row_index__')

    if use_row_index:
        df = df.reset_index(drop=True)
        df['__row_index__'] = np.arange(len(df))
        dates = df['__row_index__'].values
        df_model = pd.DataFrame({
            'date_idx': dates,
            'target': pd.to_numeric(df[target_col], errors='coerce')
        }).dropna().sort_values('date_idx')
    else:
        try:
            dates = pd.to_datetime(df[date_col])
        except Exception:
            return None, f'Cannot parse date column "{date_col}".'
        df_model = pd.DataFrame({
            'date_idx': dates,
            'target': pd.to_numeric(df[target_col], errors='coerce')
        }).dropna().sort_values('date_idx')

    if len(df_model) < 10:
        return None, f'Not enough data points ({len(df_model)}). Minimum 10 required.'

    target_vals = df_model['target'].values
    full_dates = df_model['date_idx'].values

    train_df, test_df = _apply_train_test_split(df_model, 'target', test_ratio)
    train_vals = train_df['target'].values
    test_vals = test_df['target'].values
    train_dates = train_df['date_idx'].values
    test_dates = test_df['date_idx'].values

    numeric_features = _get_numeric_features(df, date_col, target_col)
    date_features_df = _create_date_features_df(df, date_col) if not use_row_index else None

    if date_features_df is not None:
        feature_cols = date_features_df.columns.tolist()
        for feat_col in numeric_features:
            if feat_col in df.columns:
                feature_cols.append(feat_col)

        full_features = date_features_df.copy()
        for feat_col in numeric_features:
            if feat_col in df.columns:
                full_features[feat_col] = pd.to_numeric(df[feat_col], errors='coerce')

        full_features = full_features.fillna(0)
        train_features = full_features.iloc[train_df.index]
        test_features = full_features.iloc[test_df.index]

        last_date = pd.Timestamp(df_model['date_idx'].max())
        future_dates_raw = _forecast_future_dates(last_date, horizon)
        future_feat_df = _create_date_features_df(
            pd.DataFrame({date_col: future_dates_raw}), date_col
        )
        if future_feat_df is not None:
            for feat_col in numeric_features:
                if feat_col in df.columns:
                    future_feat_df[feat_col] = 0
            future_feat_df = future_feat_df.fillna(0)
            X_future = future_feat_df.values
        else:
            X_future = None
    else:
        train_features = None
        test_features = None
        X_future = None
        feature_cols = []

    if use_row_index:
        future_dates_raw = _forecast_future_dates(int(df_model['date_idx'].max()), horizon)
    else:
        last_date = pd.Timestamp(df_model['date_idx'].max())
        future_dates_raw = _forecast_future_dates(last_date, horizon)

    ml_X_train = train_features.values if train_features is not None else None
    ml_X_test = test_features.values if test_features is not None else None

    if use_row_index:
        future_dates_str = [str(d) for d in future_dates_raw]
    else:
        future_dates_str = [str(d.date()) if hasattr(d, 'date') else str(d) for d in future_dates_raw]

    results = {
        'date_column': date_col,
        'target_column': target_col,
        'horizon': horizon,
        'test_ratio': test_ratio,
        'full_data_count': len(df_model),
        'train_count': len(train_vals),
        'test_count': len(test_vals),
        'models': {},
        'best_model': None,
        'insights': {},
        'train_dates': [str(d) for d in train_dates],
        'test_dates': [str(d) for d in test_dates],
        'train_values': train_vals.tolist(),
        'test_values': test_vals.tolist(),
        'feature_columns': feature_cols,
        'forecast_dir': forecast_dir
    }

    models_to_run = []
    if auto_mode:
        models_to_run = list(ALL_MODELS.keys())
    elif model_name:
        if model_name in ALL_MODELS:
            models_to_run = [model_name]
        else:
            return None, f'Unknown model: {model_name}. Available: {", ".join(ALL_MODELS.keys())}'
    else:
        models_to_run = list(ALL_MODELS.keys())

    best_rmse = float('inf')
    best_model_name = None

    for mname in models_to_run:
        model_result = None
        model_error = None

        if mname in TRADITIONAL_MODELS:
            model_result, model_error = _run_traditional_model(
                mname, train_vals.copy(), test_vals.copy(), horizon)
        elif mname in ML_MODELS:
            if ml_X_train is not None and ml_X_test is not None and X_future is not None:
                model_result, model_error = _run_ml_model(
                    mname, ml_X_train, train_vals.copy(),
                    ml_X_test, X_future)
            else:
                model_error = 'Date features not available for ML model'
        elif mname in DL_MODELS:
            model_result, model_error = _run_traditional_model(
                mname, train_vals.copy(), test_vals.copy(), horizon)

        if model_error or model_result is None:
            results['models'][mname] = {
                'error': model_error or 'Unknown error',
                'status': 'failed'
            }
            continue

        test_pred = model_result['test_predictions']
        future_pred = model_result['future_predictions']

        metrics = _evaluate(test_vals[:len(test_pred)], test_pred)
        if metrics.get('rmse') is not None and metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model_name = mname

        results['models'][mname] = {
            'status': 'success',
            'test_predictions': test_pred,
            'future_predictions': future_pred,
            'metrics': metrics,
            'params': model_result.get('params', {})
        }

    if not results['models']:
        return None, 'All models failed to train. Check data quality.'

    if best_model_name is None:
        successful = [n for n, r in results['models'].items()
                      if r.get('status') == 'success']
        best_model_name = successful[0] if successful else None

    best = results['models'].get(best_model_name, {})
    best_metrics = best.get('metrics', {})
    best_future = best.get('future_predictions', [])

    results['best_model'] = best_model_name

    results['future_dates'] = future_dates_str

    forecasts_df = pd.DataFrame({
        'date': future_dates_str,
        'forecast': best_future[:horizon] if len(best_future) >= horizon
                    else best_future + [None] * (horizon - len(best_future))
    })
    forecast_csv = os.path.join(forecast_dir, 'forecast_output.csv')
    forecasts_df.to_csv(forecast_csv, index=False)

    report = ForecastReport(
        dataset_id=dataset.id,
        model_name=best_model_name,
        forecast_horizon=horizon,
        target_column=target_col,
        date_column=date_col,
        mae=best_metrics.get('mae'),
        rmse=best_metrics.get('rmse'),
        mape=best_metrics.get('mape'),
        r2_score=best_metrics.get('r2'),
        forecast_file=forecast_csv
    )
    db.session.add(report)
    db.session.commit()

    train_metrics = {
        mname: r['metrics'] for mname, r in results['models'].items()
        if r.get('status') == 'success' and r.get('metrics')
    }

    trend_direction = 'upward' if best_future and len(best_future) > 1 and (
        np.mean(best_future[-3:]) > np.mean(best_future[:3])
    ) else 'downward' if best_future and len(best_future) > 1 else 'stable'

    total_growth = 0
    if best_future and len(best_future) > 1:
        total_growth = ((best_future[-1] - best_future[0]) / (abs(best_future[0]) + 1e-10)) * 100

    results['insights'] = {
        'trend_direction': trend_direction,
        'total_growth_pct': round(float(total_growth), 2),
        'models_trained': len(models_to_run),
        'successful_models': len([m for m in results['models'].values()
                                   if m.get('status') == 'success']),
        'failed_models': len([m for m in results['models'].values()
                               if m.get('status') == 'failed']),
        'all_metrics': train_metrics
    }
    results['report_id'] = report.id

    save_forecast_results(dataset.id, results)
    return results, None
