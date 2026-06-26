import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestComparisonReportModel:
    def test_comparison_report_model(self, db):
        from models.comparison_report_model import ComparisonReport
        r = ComparisonReport(
            dataset_id=1,
            best_model='RandomForest',
            ranking=json.dumps([{'name': 'RandomForest', 'rank': 1}]),
            comparison_metrics=json.dumps({'rmse': [1.0], 'mae': [0.5]})
        )
        db.session.add(r)
        db.session.commit()
        assert r.id is not None
        assert r.best_model == 'RandomForest'
        assert r.dataset_id == 1
        assert r.ranking is not None
        assert r.comparison_metrics is not None

    def test_comparison_report_repr(self, db):
        from models.comparison_report_model import ComparisonReport
        r = ComparisonReport(dataset_id=1, best_model='ARIMA',
                             ranking='[]', comparison_metrics='{}')
        db.session.add(r)
        db.session.commit()
        assert 'ComparisonReport' in repr(r)
        assert 'dataset=1' in repr(r)
        assert 'best=ARIMA' in repr(r)


class TestBuildComparison:
    def test_build_no_forecast_results(self, db, sample_dataset):
        from services.comparison_service import build_comparison
        from unittest.mock import patch
        with patch('services.comparison_service.load_forecast_results', return_value=None):
            results, error = build_comparison(sample_dataset.id)
            assert results is None
            assert error is not None
            assert 'No forecast results found' in error

    def test_build_with_forecast_results(self, db, sample_dataset):
        from services.comparison_service import build_comparison
        from services.forecasting_service import save_forecast_results

        forecast_data = {
            'models': {
                'ARIMA': {
                    'status': 'success',
                    'metrics': {'mae': 0.5, 'rmse': 0.8, 'mape': 2.5, 'r2': 0.95},
                    'params': {'training_time': 0.5},
                    'test_predictions': [1, 2, 3],
                    'future_predictions': [4, 5],
                },
                'RandomForest': {
                    'status': 'success',
                    'metrics': {'mae': 0.3, 'rmse': 0.5, 'mape': 1.5, 'r2': 0.98},
                    'params': {'training_time': 1.2},
                    'test_predictions': [1.1, 2.1, 3.1],
                    'future_predictions': [4.1, 5.1],
                },
                'FailedModel': {
                    'status': 'failed',
                    'error': 'Something went wrong',
                },
            },
            'best_model': 'RandomForest',
            'future_dates': ['2024-01-01', '2024-01-02'],
            'horizon': 2,
            'target_col': 'value',
            'date_col': 'date',
            'insights': {'trend_direction': 'upward', 'total_growth_pct': 5.0},
        }
        save_forecast_results(sample_dataset.id, forecast_data)

        results, error = build_comparison(sample_dataset.id)
        assert results is not None, f'Error: {error}'
        assert error is None
        assert results['best_model'] == 'RandomForest'
        assert len(results['ranking']) == 3
        assert results['ranking'][0]['name'] == 'RandomForest'
        assert results['ranking'][0]['rank'] == 1
        assert results['ranking'][2]['status'] == 'failed'

    def test_best_model_selection(self):
        from services.comparison_service import _rank_models
        models = [
            {'name': 'ARIMA', 'mae': 0.5, 'rmse': 0.8, 'mape': 2.5, 'r2': 0.95,
             'status': 'success', 'training_time': 0.5},
            {'name': 'RandomForest', 'mae': 0.3, 'rmse': 0.5, 'mape': 1.5, 'r2': 0.98,
             'status': 'success', 'training_time': 1.2},
            {'name': 'XGBoost', 'mae': 0.6, 'rmse': 0.9, 'mape': 3.0, 'r2': 0.92,
             'status': 'success', 'training_time': 0.8},
        ]
        ranked = _rank_models(models)
        assert ranked[0]['name'] == 'RandomForest'
        assert ranked[1]['name'] == 'ARIMA'
        assert ranked[2]['name'] == 'XGBoost'

    def test_best_model_rmse_tie_mae_breaker(self):
        from services.comparison_service import _rank_models
        models = [
            {'name': 'ModelA', 'mae': 0.3, 'rmse': 0.5, 'mape': 2.0, 'r2': 0.96,
             'status': 'success', 'training_time': 0.5},
            {'name': 'ModelB', 'mae': 0.4, 'rmse': 0.5, 'mape': 2.5, 'r2': 0.94,
             'status': 'success', 'training_time': 0.6},
        ]
        ranked = _rank_models(models)
        assert ranked[0]['name'] == 'ModelA'

    def test_best_model_tie_mae_r2_breaker(self):
        from services.comparison_service import _rank_models
        models = [
            {'name': 'ModelA', 'mae': 0.3, 'rmse': 0.5, 'mape': 2.0, 'r2': 0.97,
             'status': 'success', 'training_time': 0.5},
            {'name': 'ModelB', 'mae': 0.3, 'rmse': 0.5, 'mape': 2.5, 'r2': 0.94,
             'status': 'success', 'training_time': 0.6},
        ]
        ranked = _rank_models(models)
        assert ranked[0]['name'] == 'ModelA'

    def test_rank_reason_best(self):
        from services.comparison_service import _rank_reason
        model = {'name': 'Best', 'rmse': 0.5, 'mae': 0.3, 'r2': 0.98}
        all_models = [model, {'name': 'Other', 'rmse': 0.8, 'mae': 0.5, 'r2': 0.95}]
        reason = _rank_reason(model, 0, all_models)
        assert 'Lowest RMSE' in reason

    def test_rank_reason_second(self):
        from services.comparison_service import _rank_reason
        best = {'name': 'Best', 'rmse': 0.5, 'mae': 0.3, 'r2': 0.98}
        second = {'name': 'Second', 'rmse': 0.8, 'mae': 0.5, 'r2': 0.95}
        all_models = [best, second]
        reason = _rank_reason(second, 1, all_models)
        assert 'higher than' in reason or 'lower than' in reason

    def test_build_chart_data(self):
        from services.comparison_service import _build_chart_data
        ranked = [
            {'name': 'A', 'status': 'success', 'rmse': 0.5, 'mae': 0.3, 'mape': 1.5,
             'r2': 0.98, 'training_time': 0.5},
            {'name': 'B', 'status': 'success', 'rmse': 0.8, 'mae': 0.5, 'mape': 2.5,
             'r2': 0.95, 'training_time': 1.2},
            {'name': 'C', 'status': 'failed'},
        ]
        cd = _build_chart_data(ranked)
        assert cd['labels'] == ['A', 'B']
        assert cd['rmse'] == [0.5, 0.8]
        assert cd['mae'] == [0.3, 0.5]

    def test_chart_data_empty(self):
        from services.comparison_service import _build_chart_data
        cd = _build_chart_data([])
        assert cd['labels'] == []

    def test_generate_insights_no_models(self):
        from services.comparison_service import _generate_insights
        ins = _generate_insights([], {})
        assert len(ins) == 1
        assert ins[0]['type'] == 'warning'

    def test_generate_insights_best(self):
        from services.comparison_service import _generate_insights
        ranked = [{'name': 'RF', 'status': 'success', 'rmse': 0.5, 'mae': 0.3,
                   'mape': 1.5, 'r2': 0.98, 'training_time': 1.2,
                   'model_category': 'Machine Learning'}]
        fc_insights = {'trend_direction': 'upward', 'total_growth_pct': 5.0}
        ins = _generate_insights(ranked, {'insights': fc_insights})
        types = [i['type'] for i in ins]
        assert 'best' in types
        assert 'trend' in types

    def test_generate_insights_negative_r2(self):
        from services.comparison_service import _generate_insights
        ranked = [
            {'name': 'M1', 'status': 'success', 'rmse': 1.0, 'mae': 0.8,
             'mape': 5.0, 'r2': -0.5, 'training_time': 0.5,
             'model_category': 'Traditional'},
            {'name': 'M2', 'status': 'success', 'rmse': 2.0, 'mae': 1.5,
             'mape': 10.0, 'r2': 0.9, 'training_time': 1.0,
             'model_category': 'ML'},
        ]
        ins = _generate_insights(ranked, {'insights': {}})
        types = [i['type'] for i in ins]
        assert 'warning' in types

    def test_get_model_category(self):
        from services.comparison_service import _get_model_category
        assert _get_model_category('ARIMA') == 'Traditional'
        assert _get_model_category('RandomForest') == 'Machine Learning'
        assert _get_model_category('LSTM') == 'Deep Learning'
        assert _get_model_category('Unknown') == 'Unknown'


class TestGenerateComparison:
    def test_generate_csv_no_results(self, db):
        from services.comparison_service import generate_comparison_csv
        path, error = generate_comparison_csv(999)
        assert path is None
        assert error is not None

    def test_generate_csv_with_results(self, db, sample_dataset):
        from services.comparison_service import (
            build_comparison, generate_comparison_csv, save_comparison_results
        )
        results = {
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
                {'rank_label': '#2', 'name': 'ARIMA', 'model_category': 'Traditional',
                 'status': 'success', 'mae': 0.5, 'rmse': 0.8, 'mape': '2.5%',
                 'r2': 0.95, 'training_time': '0.50s'},
            ]
        }
        save_comparison_results(sample_dataset.id, results)
        path, error = generate_comparison_csv(sample_dataset.id)
        assert path is not None
        assert error is None
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) == 2
        assert 'Model Name' in df.columns
        assert 'RMSE' in df.columns

    def test_generate_excel_no_results(self, db):
        from services.comparison_service import generate_comparison_excel
        path, error = generate_comparison_excel(999)
        assert path is None
        assert error is not None

    def test_generate_excel_with_results(self, db, sample_dataset):
        from services.comparison_service import generate_comparison_excel, save_comparison_results
        results = {
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
            ]
        }
        save_comparison_results(sample_dataset.id, results)
        path, error = generate_comparison_excel(sample_dataset.id)
        assert path is not None
        assert error is None
        assert os.path.exists(path)

    def test_generate_pdf_no_results(self, db):
        from services.comparison_service import generate_comparison_pdf
        path, error = generate_comparison_pdf(999)
        assert path is None
        assert error is not None

    def test_generate_pdf_with_results(self, db, sample_dataset):
        from services.comparison_service import generate_comparison_pdf, save_comparison_results
        results = {
            'best_model': 'RF',
            'best_explanation': 'RF is best.',
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
            ]
        }
        save_comparison_results(sample_dataset.id, results)
        path, error = generate_comparison_pdf(sample_dataset.id)
        if path:
            assert os.path.exists(path)
        else:
            assert error is not None


class TestGetActualPredicted:
    def test_get_actual_predicted_no_results(self, db):
        from services.comparison_service import get_model_actual_predicted
        data, error = get_model_actual_predicted(999, 'ARIMA')
        assert data is None
        assert error is not None

    def test_get_actual_predicted_model_not_found(self, db, sample_dataset):
        from services.comparison_service import get_model_actual_predicted
        from services.forecasting_service import save_forecast_results
        save_forecast_results(sample_dataset.id, {'models': {}})
        data, error = get_model_actual_predicted(sample_dataset.id, 'ARIMA')
        assert data is None
        assert error is not None

    def test_get_actual_predicted_success(self, db, sample_dataset):
        from services.comparison_service import get_model_actual_predicted
        from services.forecasting_service import save_forecast_results
        forecast_data = {
            'models': {
                'ARIMA': {
                    'status': 'success',
                    'test_predictions': [1, 2, 3],
                    'future_predictions': [4, 5],
                },
            },
            'target_col': 'value',
            'date_col': 'date',
            'future_dates': ['2024-01-01', '2024-01-02'],
        }
        save_forecast_results(sample_dataset.id, forecast_data)
        data, error = get_model_actual_predicted(sample_dataset.id, 'ARIMA')
        assert data is not None
        assert data['model_name'] == 'ARIMA'
        assert len(data['test_predictions']) == 3
        assert len(data['future_predictions']) == 2

    def test_get_actual_predicted_failed_model(self, db, sample_dataset):
        from services.comparison_service import get_model_actual_predicted
        from services.forecasting_service import save_forecast_results
        forecast_data = {
            'models': {
                'ARIMA': {
                    'status': 'failed',
                    'error': 'Failed to converge',
                },
            },
        }
        save_forecast_results(sample_dataset.id, forecast_data)
        data, error = get_model_actual_predicted(sample_dataset.id, 'ARIMA')
        assert data is None
        assert error is not None
        assert 'did not complete' in error


class TestComparisonPersistence:
    def test_save_and_load_results(self, db, sample_dataset):
        from services.comparison_service import save_comparison_results, load_comparison_results
        results = {'best_model': 'RF', 'ranking': [{'name': 'RF', 'rank': 1}]}
        save_comparison_results(sample_dataset.id, results)
        loaded = load_comparison_results(sample_dataset.id)
        assert loaded is not None
        assert loaded['best_model'] == 'RF'
        assert loaded['ranking'][0]['name'] == 'RF'

    def test_load_nonexistent(self, db):
        from services.comparison_service import load_comparison_results
        assert load_comparison_results(999) is None

    def test_get_report_nonexistent(self, db):
        from services.comparison_service import get_comparison_report
        assert get_comparison_report(999) is None

    def test_get_report_exists(self, db, sample_dataset):
        from models.comparison_report_model import ComparisonReport
        from services.comparison_service import get_comparison_report
        r = ComparisonReport(dataset_id=sample_dataset.id, best_model='RF',
                             ranking='[]', comparison_metrics='{}')
        db.session.add(r)
        db.session.commit()
        report = get_comparison_report(sample_dataset.id)
        assert report is not None
        assert report.best_model == 'RF'

    def test_get_comparison_context_existing(self, db, sample_dataset):
        from services.comparison_service import get_comparison_context, save_comparison_results
        results = {'best_model': 'XGB', 'ranking': []}
        save_comparison_results(sample_dataset.id, results)
        loaded, error = get_comparison_context(sample_dataset.id)
        assert loaded is not None
        assert loaded['best_model'] == 'XGB'

    def test_get_comparison_context_new(self, db, sample_dataset):
        from services.comparison_service import get_comparison_context
        forecast_data = {
            'models': {
                'ARIMA': {
                    'status': 'success',
                    'metrics': {'mae': 0.5, 'rmse': 0.8, 'mape': 2.5, 'r2': 0.95},
                    'params': {},
                    'test_predictions': [1, 2, 3],
                    'future_predictions': [4, 5],
                },
            },
            'best_model': 'ARIMA',
            'future_dates': ['2024-01-01'],
            'horizon': 1,
            'target_col': 'value',
            'date_col': 'date',
            'insights': {},
        }
        with patch('services.comparison_service.load_comparison_results',
                   return_value=None):
            with patch('services.comparison_service.load_forecast_results',
                       return_value=forecast_data):
                results, error = get_comparison_context(sample_dataset.id)
                assert results is not None
                assert results['best_model'] == 'ARIMA'


class TestComparisonRoutes:
    def test_compare_dashboard_requires_login(self, client):
        resp = client.get('/compare/1')
        assert resp.status_code == 302

    def test_compare_results_requires_login(self, client):
        resp = client.get('/compare/results/1')
        assert resp.status_code == 302

    def test_compare_download_requires_login(self, client):
        resp = client.get('/compare/download/1/csv')
        assert resp.status_code == 302

    def test_compare_dashboard_no_dataset(self, auth_client):
        resp = auth_client.get('/compare/999')
        assert resp.status_code == 302

    def test_compare_results_no_dataset(self, auth_client):
        resp = auth_client.get('/compare/results/999')
        assert resp.status_code == 302

    def test_compare_api_requires_login(self, client):
        resp = client.get('/compare/api/model/1/ARIMA')
        assert resp.status_code == 302

    def test_compare_api_not_found(self, auth_client):
        resp = auth_client.get('/compare/api/model/999/ARIMA')
        assert resp.status_code == 404


class TestWorkflowIntegration:
    def test_compare_step_in_workflow(self, db):
        from services.workflow_service import STEPS
        step6 = [s for s in STEPS if s['number'] == 6]
        assert len(step6) == 1
        assert step6[0]['name'] == 'Compare'
        assert step6[0]['route'] == 'comparison.compare_dashboard'

    def test_compare_locked_without_forecast(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        state = get_workflow_state(sample_dataset.id)
        assert 6 not in state['completed']
        assert 6 in state['locked']

    def test_compare_unlocked_with_forecast(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        from models.preprocessing_report_model import PreprocessingReport
        from models.forecast_report_model import ForecastReport
        prep = PreprocessingReport(dataset_id=sample_dataset.id, mode='auto')
        db.session.add(prep)
        fc = ForecastReport(dataset_id=sample_dataset.id, model_name='ARIMA',
                            forecast_horizon=10)
        db.session.add(fc)
        db.session.commit()
        state = get_workflow_state(sample_dataset.id)
        assert 6 not in state['locked'] or 6 in state['completed']

    def test_compare_completed_with_report(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        from models.preprocessing_report_model import PreprocessingReport
        from models.forecast_report_model import ForecastReport
        from models.comparison_report_model import ComparisonReport
        prep = PreprocessingReport(dataset_id=sample_dataset.id, mode='auto')
        db.session.add(prep)
        fc = ForecastReport(dataset_id=sample_dataset.id, model_name='ARIMA',
                            forecast_horizon=10)
        db.session.add(fc)
        comp = ComparisonReport(dataset_id=sample_dataset.id, best_model='RF',
                                ranking='[]', comparison_metrics='{}')
        db.session.add(comp)
        db.session.commit()
        state = get_workflow_state(sample_dataset.id)
        assert 6 in state['completed']

    def test_reports_locked_after_comparison(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        from models.forecast_report_model import ForecastReport
        from models.comparison_report_model import ComparisonReport
        fc = ForecastReport(dataset_id=sample_dataset.id, model_name='ARIMA',
                            forecast_horizon=10)
        db.session.add(fc)
        comp = ComparisonReport(dataset_id=sample_dataset.id, best_model='RF',
                                ranking='[]', comparison_metrics='{}')
        db.session.add(comp)
        db.session.commit()
        state = get_workflow_state(sample_dataset.id)
        assert 7 in state['locked']
