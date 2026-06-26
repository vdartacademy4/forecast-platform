import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestForecastModel:
    def test_forecast_report_model(self, db):
        from models.forecast_report_model import ForecastReport
        r = ForecastReport(
            dataset_id=1,
            model_name='ARIMA',
            forecast_horizon=30,
            target_column='value',
            date_column='date',
            mae=1.5,
            rmse=2.0,
            mape=5.0,
            r2_score=0.95,
            forecast_file='/tmp/forecast.csv'
        )
        db.session.add(r)
        db.session.commit()
        assert r.id is not None
        assert r.model_name == 'ARIMA'
        assert r.forecast_horizon == 30
        assert r.mae == 1.5
        assert r.r2_score == 0.95
        assert repr(r) == '<ForecastReport dataset=1 model=ARIMA>'


class TestForecastingService:
    def test_detect_date_column_datetime_type(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        assert _detect_date_column(df) == 'date'

    def test_detect_date_column_string_parseable(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1, 2, 3]
        })
        col = _detect_date_column(df)
        assert col == 'timestamp'

    def test_detect_date_column_named_date(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'Date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1, 2, 3]
        })
        col = _detect_date_column(df)
        assert col == 'Date'

    def test_detect_date_column_none(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert _detect_date_column(df) is None

    def test_detect_target_column(self):
        from services.forecasting_service import _detect_target_column
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'const': [1] * 10
        })
        col = _detect_target_column(df, 'date')
        assert col == 'value'

    def test_detect_target_column_no_numeric(self):
        from services.forecasting_service import _detect_target_column
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['p', 'q']})
        assert _detect_target_column(df, None) is None

    def test_evaluate_metrics(self):
        from services.forecasting_service import _evaluate
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.0, 2.9, 4.1, 5.0]
        m = _evaluate(y_true, y_pred)
        assert 'mae' in m
        assert 'rmse' in m
        assert 'mape' in m
        assert 'r2' in m
        assert m['mae'] >= 0
        assert m['r2'] > 0

    def test_evaluate_perfect_prediction(self):
        from services.forecasting_service import _evaluate
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        m = _evaluate(y_true, y_pred)
        assert m['mae'] == 0
        assert m['rmse'] == 0
        assert m['r2'] == 1.0

    def test_apply_train_test_split_default(self):
        from services.forecasting_service import _apply_train_test_split
        df = pd.DataFrame({
            'date_idx': pd.date_range('2020-01-01', periods=100),
            'target': range(100)
        })
        train, test = _apply_train_test_split(df, 'target', test_ratio=0.2)
        assert len(train) == 80
        assert len(test) == 20

    def test_apply_train_test_split_7030(self):
        from services.forecasting_service import _apply_train_test_split
        df = pd.DataFrame({
            'date_idx': pd.date_range('2020-01-01', periods=100),
            'target': range(100)
        })
        train, test = _apply_train_test_split(df, 'target', test_ratio=0.3)
        assert len(train) == 70
        assert len(test) == 30

    def test_forecast_future_dates(self):
        from services.forecasting_service import _forecast_future_dates
        dates = _forecast_future_dates('2024-01-01', 30)
        assert len(dates) == 30
        assert str(dates[0])[:10] == '2024-01-02'

    def test_create_date_features_df(self):
        from services.forecasting_service import _create_date_features_df
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10)
        })
        feat = _create_date_features_df(df, 'date')
        assert feat is not None
        assert 'year' in feat.columns
        assert 'month' in feat.columns
        assert 'day' in feat.columns
        assert 'dayofweek' in feat.columns
        assert 'quarter' in feat.columns
        assert len(feat) == 10

    def test_get_numeric_features(self):
        from services.forecasting_service import _get_numeric_features
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'target': [1, 2, 3, 4, 5],
            'feature1': [10, 20, 30, 40, 50],
            'cat': ['a', 'b', 'a', 'b', 'a']
        })
        features = _get_numeric_features(df, 'date', 'target')
        assert 'feature1' in features
        assert 'target' not in features
        assert 'date' not in features

    @patch('services.forecasting_service._HAS_STATSMODELS', True)
    @patch('services.forecasting_service.ARIMA')
    def test_arima_training(self, mock_arima):
        from services.forecasting_service import _train_arima
        mock_instance = MagicMock()
        mock_fitted = MagicMock()
        forecast_call_count = {'count': 0}
        def forecast_side_effect(steps=1):
            forecast_call_count['count'] += 1
            if forecast_call_count['count'] == 1:
                return np.array([10, 11, 12])
            return np.array([20, 21, 22, 23, 24])
        mock_fitted.forecast.side_effect = forecast_side_effect
        mock_instance.fit.return_value = mock_fitted
        mock_arima.return_value = mock_instance

        train_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        test_vals = np.array([11, 12, 13])
        result, error = _train_arima(train_vals, test_vals, horizon=5)
        assert error is None
        assert result['model'] == 'ARIMA'
        assert len(result['test_predictions']) == 3
        assert len(result['future_predictions']) == 5

    def test_is_date_column_datetime64(self):
        from services.forecasting_service import _is_date_column
        s = pd.Series(pd.date_range('2020-01-01', periods=10))
        assert _is_date_column(s) is True

    def test_is_date_column_string_dates(self):
        from services.forecasting_service import _is_date_column
        s = pd.Series(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                        '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'])
        assert bool(_is_date_column(s)) is True

    def test_is_date_column_mixed_formats(self):
        from services.forecasting_service import _is_date_column
        s = pd.Series(['2024-01-01', '2024-01-15', '2024-03-01',
                        '2024-04-01', '2024-05-01'])
        assert bool(_is_date_column(s)) is True

    def test_is_date_column_below_threshold(self):
        from services.forecasting_service import _is_date_column
        s = pd.Series(['2024-01-01', 'not_a_date', 'also_bad', '2024-01-04', 'nope'])
        assert bool(_is_date_column(s, threshold=0.7)) is False

    def test_is_date_column_above_threshold(self):
        from services.forecasting_service import _is_date_column
        s = pd.Series(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', 'not_a_date'])
        assert bool(_is_date_column(s, threshold=0.7)) is True

    def test_is_date_column_empty(self):
        from services.forecasting_service import _is_date_column
        assert _is_date_column(pd.Series([], dtype='object')) is False

    def test_is_numeric_timestamp_seconds(self):
        from services.forecasting_service import _is_numeric_timestamp
        s = pd.Series([1704067200, 1704153600, 1704240000, 1704326400, 1704412800])
        assert _is_numeric_timestamp(s) is True

    def test_is_numeric_timestamp_milliseconds(self):
        from services.forecasting_service import _is_numeric_timestamp
        s = pd.Series([1704067200000, 1704153600000, 1704240000000])
        assert _is_numeric_timestamp(s) is True

    def test_is_numeric_timestamp_small_numbers(self):
        from services.forecasting_service import _is_numeric_timestamp
        s = pd.Series([1, 2, 3, 4, 5])
        assert _is_numeric_timestamp(s) is False

    def test_detect_date_column_numeric_timestamp(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'ts': [1704067200, 1704153600, 1704240000, 1704326400, 1704412800],
            'value': [10, 20, 30, 40, 50]
        })
        col = _detect_date_column(df)
        assert col == 'ts'

    def test_detect_date_column_all_fail(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        assert _detect_date_column(df) is None

    def test_automatic_forecasting_fallback_row_index(self, db):
        from services.forecasting_service import run_automatic_forecasting
        from models.dataset_model import Dataset
        import tempfile, os

        ds = Dataset(user_id=1, file_name='test.csv',
                     file_path=os.path.join(tempfile.gettempdir(), 'test_no_date.csv'),
                     file_size=100, rows_count=10, columns_count=2)
        db.session.add(ds)
        db.session.commit()

        df = pd.DataFrame({'value': [10, 12, 15, 14, 18, 20, 22, 25, 24, 28]})
        df.to_csv(ds.file_path, index=False)

        results, error = run_automatic_forecasting(ds.id, 1, horizon=5, test_ratio=0.2)
        assert results is not None, f'Error: {error}'
        assert results['date_column'] == '__row_index__'
        assert results['best_model'] is not None

    def test_manual_forecasting_empty_date_col(self, db):
        from services.forecasting_service import run_manual_forecasting
        from models.dataset_model import Dataset
        import tempfile, os

        ds = Dataset(user_id=1, file_name='test.csv',
                     file_path=os.path.join(tempfile.gettempdir(), 'test_manual_no_date.csv'),
                     file_size=100, rows_count=10, columns_count=1)
        db.session.add(ds)
        db.session.commit()

        df = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        df.to_csv(ds.file_path, index=False)

        results, error = run_manual_forecasting(ds.id, 1, target_col='target',
                                                 date_col='', horizon=5, test_ratio=0.2)
        assert results is not None, f'Error: {error}'
        assert results['date_column'] == '__row_index__'

    def test_forecast_future_dates_int(self):
        from services.forecasting_service import _forecast_future_dates
        dates = _forecast_future_dates(100, 5)
        assert dates == [101, 102, 103, 104, 105]

    def test_forecast_future_dates_float(self):
        from services.forecasting_service import _forecast_future_dates
        dates = _forecast_future_dates(100.0, 3)
        assert dates == [101.0, 102.0, 103.0]

    def test_detect_date_column_formatted_strings(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'datetime_str': ['2024-01-01 12:00:00', '2024-01-02 13:30:00',
                              '2024-01-03 14:45:00', '2024-01-04 16:00:00',
                              '2024-01-05 17:15:00'],
            'val': [1, 2, 3, 4, 5]
        })
        col = _detect_date_column(df)
        assert col == 'datetime_str'

    def test_detect_date_column_offset_format(self):
        from services.forecasting_service import _detect_date_column
        df = pd.DataFrame({
            'ts': ['2006-04-01 00:00:00.000 +0200', '2006-04-02 00:00:00.000 +0200',
                    '2006-04-03 00:00:00.000 +0200'],
            'v': [10, 20, 30]
        })
        col = _detect_date_column(df)
        assert col is not None


class TestForecastRoutes:
    def test_forecast_mode_requires_login(self, client):
        resp = client.get('/forecasting/1', follow_redirects=True)
        assert resp.status_code == 200
        assert b'Please log in' in resp.data or b'Login' in resp.data or b'login' in resp.data.lower()

    def test_forecast_mode_page(self, auth_client, db, sample_dataset):
        resp = auth_client.get(f'/forecasting/{sample_dataset.id}', follow_redirects=True)
        assert resp.status_code in (200, 302)

    def test_forecast_setup_requires_login(self, client):
        resp = client.get('/forecasting/setup/1', follow_redirects=True)
        assert resp.status_code == 200

    def test_forecast_dashboard_requires_login(self, client):
        resp = client.get('/forecasting/dashboard/1', follow_redirects=True)
        assert resp.status_code == 200

    def test_forecast_results_requires_login(self, client):
        resp = client.get('/forecasting/results/1', follow_redirects=True)
        assert resp.status_code == 200

    def test_forecast_nonexistent_dataset(self, auth_client):
        resp = auth_client.get('/forecasting/9999', follow_redirects=True)
        assert resp.status_code == 200
        assert b'Dataset not found' in resp.data or b'dataset' in resp.data.lower()


class TestForecastPersistence:
    def test_save_and_load_results(self, tmp_path):
        from services.forecasting_service import save_forecast_results, _get_forecast_dir
        import services.forecasting_service as fs

        results = {
            'best_model': 'ARIMA',
            'horizon': 30,
            'test_ratio': 0.2,
            'models': {'ARIMA': {'status': 'success', 'metrics': {'mae': 1.0}}},
            'insights': {'trend_direction': 'upward', 'total_growth_pct': 5.0}
        }

        fs.FORECAST_FOLDER = str(tmp_path)
        save_forecast_results(1, results)
        loaded = fs.load_forecast_results(1)
        assert loaded is not None
        assert loaded['best_model'] == 'ARIMA'
        assert loaded['models']['ARIMA']['status'] == 'success'

    def test_get_forecast_report_none(self, db):
        from services.forecasting_service import get_forecast_report
        report = get_forecast_report(9999)
        assert report is None


class TestWorkflowIntegration:
    def test_forecast_step_in_workflow(self, db):
        from services.workflow_service import get_workflow_state, STEPS
        state = get_workflow_state(1)
        step5 = STEPS[4]
        assert step5['number'] == 5
        assert step5['name'] == 'Forecasting'
        assert step5['route'] == 'forecasting.mode_selection'

    def test_forecast_locked_without_report(self, db):
        from services.workflow_service import get_workflow_state
        state = get_workflow_state(999)
        assert 5 in state['locked']

    def test_forecast_unlocked_with_report(self, db):
        from services.workflow_service import get_workflow_state
        from models.forecast_report_model import ForecastReport
        from models.preprocessing_report_model import PreprocessingReport
        from models.dataset_model import Dataset

        ds = Dataset(user_id=1, file_name='test.csv', file_path='/tmp/test.csv',
                     file_size=100, rows_count=10, columns_count=3)
        db.session.add(ds)
        db.session.commit()

        prep = PreprocessingReport(dataset_id=ds.id, mode='automatic',
                                    original_shape='100x5', processed_shape='100x5')
        db.session.add(prep)
        db.session.commit()

        r = ForecastReport(dataset_id=ds.id, model_name='ARIMA',
                           forecast_horizon=30)
        db.session.add(r)
        db.session.commit()

        state = get_workflow_state(ds.id)
        assert 5 in state['completed']
        assert 5 not in state['locked']

    def test_forecast_unlocked_after_preprocessing(self, db):
        from services.workflow_service import get_workflow_state
        from models.preprocessing_report_model import PreprocessingReport
        from models.eda_report_model import EDAReport
        from models.validation_report_model import ValidationReport
        from models.dataset_model import Dataset

        ds = Dataset(user_id=1, file_name='test.csv', file_path='/tmp/test.csv',
                     file_size=100, rows_count=10, columns_count=3)
        db.session.add(ds)
        db.session.commit()

        vr = ValidationReport(dataset_id=ds.id, validation_status='completed',
                               total_rows=10, total_columns=3)
        db.session.add(vr)
        db.session.commit()

        eda = EDAReport(dataset_id=ds.id, eda_mode='automatic')
        db.session.add(eda)
        db.session.commit()

        prep = PreprocessingReport(dataset_id=ds.id, mode='automatic',
                                    original_shape='100x5', processed_shape='100x5')
        db.session.add(prep)
        db.session.commit()

        state = get_workflow_state(ds.id)
        assert 5 not in state['locked']
        assert 5 not in state['completed']
        assert state['current'] == 5

    def test_compare_locked_after_forecast(self, db):
        from services.workflow_service import get_workflow_state
        state = get_workflow_state(999)
        assert 6 in state['locked']

    def test_reports_locked_after_forecast(self, db):
        from services.workflow_service import get_workflow_state
        state = get_workflow_state(999)
        assert 7 in state['locked']
