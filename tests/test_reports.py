import os
import json
import pytest
from unittest.mock import patch


class TestReportModel:
    def test_report_model(self, db):
        from models.report_model import Report
        r = Report(dataset_id=1)
        db.session.add(r)
        db.session.commit()
        assert r.id is not None
        assert r.dataset_id == 1

    def test_report_repr(self, db):
        from models.report_model import Report
        r = Report(dataset_id=1)
        db.session.add(r)
        db.session.commit()
        assert 'Report' in repr(r)
        assert 'dataset=1' in repr(r)


class TestGenerateFullReport:
    def test_generate_no_dataset(self, db):
        from services.report_service import generate_full_report
        results, error = generate_full_report(999)
        assert results is None
        assert error is not None
        assert 'not found' in error

    def test_generate_with_minimal_data(self, db, sample_dataset):
        from services.report_service import generate_full_report
        from services.forecasting_service import save_forecast_results
        from services.comparison_service import save_comparison_results
        from models.validation_report_model import ValidationReport

        v = ValidationReport(dataset_id=sample_dataset.id, validation_status='completed', total_rows=100, total_columns=5)
        db.session.add(v)
        db.session.commit()

        forecast_data = {
            'models': {
                'ARIMA': {'status': 'success', 'metrics': {'mae': 0.5, 'rmse': 0.8, 'mape': 2.5, 'r2': 0.95}, 'params': {'training_time': 0.5}, 'test_predictions': [1, 2, 3], 'future_predictions': [4, 5]},
                'RF': {'status': 'success', 'metrics': {'mae': 0.3, 'rmse': 0.5, 'mape': 1.5, 'r2': 0.98}, 'params': {'training_time': 1.2}, 'test_predictions': [1.1, 2.1, 3.1], 'future_predictions': [4.1, 5.1]},
            },
            'best_model': 'RF',
            'future_dates': ['2024-01-01', '2024-01-02'],
            'horizon': 2,
            'target_col': 'value',
            'date_col': 'date',
            'insights': {'trend_direction': 'upward', 'total_growth_pct': 5.0},
        }
        save_forecast_results(sample_dataset.id, forecast_data)

        comp_results = {
            'best_model': 'RF',
            'best_explanation': 'RF is best.',
            'ranking': [{'name': 'RF', 'rank': 1, 'status': 'success', 'rmse': 0.5, 'mae': 0.3, 'r2': 0.98}, {'name': 'ARIMA', 'rank': 2, 'status': 'success', 'rmse': 0.8, 'mae': 0.5, 'r2': 0.95}],
            'chart_data': {'labels': ['RF', 'ARIMA'], 'rmse': [0.5, 0.8], 'mae': [0.3, 0.5]},
            'insights': [{'type': 'best', 'text': 'RF ranks #1.'}],
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML', 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%', 'r2': 0.98, 'training_time': '1.20s'},
                {'rank_label': '#2', 'name': 'ARIMA', 'model_category': 'Traditional', 'status': 'success', 'mae': 0.5, 'rmse': 0.8, 'mape': '2.5%', 'r2': 0.95, 'training_time': '0.50s'},
            ],
        }
        save_comparison_results(sample_dataset.id, comp_results)

        results, error = generate_full_report(sample_dataset.id)
        assert results is not None, f'Error: {error}'
        assert error is None
        assert results['dataset_id'] == sample_dataset.id
        assert results['best_model'] == 'RF'
        assert results['dataset_name'] == 'test_data.csv'
        assert results['report_id'] is not None
        assert len(results['steps_summary']) == 7
        assert len(results['results_table']) == 2
        assert 'smart_insights' in results
        assert len(results['smart_insights']) > 0

    def test_generate_report_no_forecast(self, db, sample_dataset):
        from services.report_service import generate_full_report
        from unittest.mock import patch
        with patch('services.report_service.load_comparison_results', return_value=None), \
             patch('services.report_service.load_forecast_results', return_value=None):
            results, error = generate_full_report(sample_dataset.id)
            assert results is not None
            assert results['best_model'] == 'N/A'
            assert results['forecast_models'] == {}
            assert results['forecasts'] == []


class TestReportPersistence:
    def test_save_and_load_results(self, db, sample_dataset):
        from services.report_service import save_report_results, load_report_results
        results = {'best_model': 'RF', 'dataset_name': 'test.csv'}
        save_report_results(sample_dataset.id, results)
        loaded = load_report_results(sample_dataset.id)
        assert loaded is not None
        assert loaded['best_model'] == 'RF'

    def test_load_nonexistent(self, db):
        from services.report_service import load_report_results
        assert load_report_results(999) is None

    def test_get_report_nonexistent(self, db):
        from services.report_service import get_report
        assert get_report(999) is None

    def test_get_report_exists(self, db, sample_dataset):
        from models.report_model import Report
        from services.report_service import get_report
        r = Report(dataset_id=sample_dataset.id)
        db.session.add(r)
        db.session.commit()
        report = get_report(sample_dataset.id)
        assert report is not None
        assert report.dataset_id == sample_dataset.id

    def test_get_report_context_existing(self, db, sample_dataset):
        from services.report_service import get_report_context, save_report_results
        results = {'best_model': 'XGB', 'dataset_name': 'test.csv'}
        save_report_results(sample_dataset.id, results)
        loaded, error = get_report_context(sample_dataset.id)
        assert loaded is not None
        assert loaded['best_model'] == 'XGB'

    def test_get_report_context_new(self, db, sample_dataset):
        from services.report_service import get_report_context
        from unittest.mock import patch
        with patch('services.report_service.load_report_results', return_value=None), \
             patch('services.report_service.load_comparison_results', return_value=None), \
             patch('services.report_service.load_forecast_results', return_value=None):
            results, error = get_report_context(sample_dataset.id)
            assert results is not None
            assert results['best_model'] == 'N/A'


class TestReportDownloads:
    def test_generate_csv_no_results(self, db):
        from services.report_service import generate_report_csv
        path, error = generate_report_csv(999)
        assert path is None
        assert error is not None

    def test_generate_csv_with_results(self, db, sample_dataset):
        from services.report_service import generate_report_csv, save_report_results
        results = {
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
            ]
        }
        save_report_results(sample_dataset.id, results)
        path, error = generate_report_csv(sample_dataset.id)
        assert path is not None
        assert error is None
        assert os.path.exists(path)
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 1
        assert 'Model Name' in df.columns

    def test_generate_excel_no_results(self, db):
        from services.report_service import generate_report_excel
        path, error = generate_report_excel(999)
        assert path is None
        assert error is not None

    def test_generate_excel_with_results(self, db, sample_dataset):
        from services.report_service import generate_report_excel, save_report_results
        results = {
            'steps_summary': [{'number': 1, 'name': 'Upload', 'detail': 'test'}],
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
            ],
            'forecasts': [{'period': 'Period 1', 'value': 4.1}],
        }
        save_report_results(sample_dataset.id, results)
        path, error = generate_report_excel(sample_dataset.id)
        assert path is not None
        assert error is None
        assert os.path.exists(path)

    def test_generate_pdf_no_results(self, db):
        from services.report_service import generate_report_pdf
        path, error = generate_report_pdf(999)
        assert path is None
        assert error is not None

    def test_generate_pdf_with_results(self, db, sample_dataset):
        from services.report_service import generate_report_pdf, save_report_results
        results = {
            'dataset_name': 'test.csv',
            'generated_at': 'January 1, 2025',
            'best_model': 'RF',
            'best_explanation': 'RF is best.',
            'results_table': [
                {'rank_label': '#1', 'name': 'RF', 'model_category': 'ML',
                 'status': 'success', 'mae': 0.3, 'rmse': 0.5, 'mape': '1.5%',
                 'r2': 0.98, 'training_time': '1.20s'},
            ],
            'steps_summary': [{'number': 1, 'name': 'Upload', 'detail': 'test'}],
            'forecasts': [],
        }
        save_report_results(sample_dataset.id, results)
        path, error = generate_report_pdf(sample_dataset.id)
        if path:
            assert os.path.exists(path)
        else:
            assert error is not None


class TestReportRoutes:
    def test_view_report_requires_login(self, client):
        resp = client.get('/reports/1')
        assert resp.status_code == 302

    def test_download_report_requires_login(self, client):
        resp = client.get('/reports/download/1/csv')
        assert resp.status_code == 302

    def test_view_report_no_dataset(self, auth_client):
        resp = auth_client.get('/reports/999')
        assert resp.status_code == 302


class TestAnalysisHistory:
    def test_analysis_history_model(self, db, sample_dataset):
        from models.analysis_history_model import AnalysisHistory
        h = AnalysisHistory(dataset_id=sample_dataset.id, user_id=1,
                            dataset_name='test.csv', best_model='RF',
                            forecast_horizon=10, total_rows=100, total_columns=5,
                            total_workflow_time=45.0)
        db.session.add(h)
        db.session.commit()
        assert h.id is not None
        assert h.dataset_name == 'test.csv'
        assert h.best_model == 'RF'

    def test_save_analysis_history(self, db, sample_dataset, auth_client):
        from services.report_service import save_analysis_history
        h = save_analysis_history(sample_dataset.id, 1)
        assert h is not None
        assert h.id is not None

    def test_save_analysis_history_dedup(self, db, sample_dataset, auth_client):
        from services.report_service import save_analysis_history
        h1 = save_analysis_history(sample_dataset.id, 1)
        h2 = save_analysis_history(sample_dataset.id, 1)
        assert h1.id == h2.id

    def test_get_analysis_history(self, db, sample_dataset, auth_client):
        from services.report_service import save_analysis_history, get_analysis_history
        save_analysis_history(sample_dataset.id, 1)
        history = get_analysis_history(1, limit=10)
        assert len(history) >= 1


class TestSmartInsights:
    def test_smart_insights_generated(self, db, sample_dataset):
        from services.report_service import generate_full_report
        from services.forecasting_service import save_forecast_results
        from services.comparison_service import save_comparison_results
        from models.validation_report_model import ValidationReport

        v = ValidationReport(dataset_id=sample_dataset.id, validation_status='completed',
                             total_rows=100, total_columns=5)
        db.session.add(v)

        forecast_data = {
            'models': {
                'RF': {'status': 'success', 'metrics': {'mae': 0.3, 'rmse': 0.5, 'mape': 1.5, 'r2': 0.98},
                       'params': {'training_time': 1.2}, 'test_predictions': [1, 2, 3], 'future_predictions': [4, 5]},
            },
            'best_model': 'RF',
            'future_dates': ['2024-01-01', '2024-01-02'],
            'horizon': 2,
            'target_col': 'value',
            'date_col': 'date',
        }
        save_forecast_results(sample_dataset.id, forecast_data)

        comp_results = {
            'best_model': 'RF',
            'best_explanation': 'RF is best.',
            'ranking': [{'name': 'RF', 'rank': 1, 'status': 'success', 'rmse': 0.5, 'mae': 0.3, 'r2': 0.98}],
            'chart_data': {},
            'insights': [{'type': 'best', 'text': 'RF ranks #1.'}],
            'results_table': [],
        }
        save_comparison_results(sample_dataset.id, comp_results)

        results, error = generate_full_report(sample_dataset.id)
        assert error is None
        assert 'smart_insights' in results
        insights = results['smart_insights']
        assert len(insights) >= 3
        assert any('RF' in ins for ins in insights)

    def test_smart_insights_data_quality(self, db, sample_dataset):
        from services.report_service import _generate_smart_insights
        ctx = {
            'validation': {
                'missing_values': [{'count': 0}],
                'duplicate_rows': 0,
                'validation_status': 'completed',
            },
            'preprocessing': {'steps_applied': {'missing_method': 'mean', 'outlier_method': 'iqr'}},
            'best_model': 'XGB',
            'best_rmse': 1.5,
            'comparison': {'best_model': 'XGB', 'ranking': [{'name': 'XGB', 'rmse': 1.5}], 'results_table': []},
            'forecast': {'forecast_horizon': 30, 'models': {}},
        }
        insights = _generate_smart_insights(ctx)
        assert len(insights) >= 4
        assert any('XGB' in ins for ins in insights)


class TestWorkflowIntegration:
    def test_reports_step_in_workflow(self, db):
        from services.workflow_service import STEPS
        step7 = [s for s in STEPS if s['number'] == 7]
        assert len(step7) == 1
        assert step7[0]['name'] == 'Reports'
        assert step7[0]['route'] == 'reports.view_report'

    def test_reports_locked_without_comparison(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        state = get_workflow_state(sample_dataset.id)
        assert 7 not in state['completed']
        assert 7 in state['locked']

    def test_reports_unlock_with_comparison(self, db, sample_dataset):
        from services.workflow_service import get_workflow_state
        from models.preprocessing_report_model import PreprocessingReport
        from models.forecast_report_model import ForecastReport
        from models.comparison_report_model import ComparisonReport
        from models.report_model import Report
        prep = PreprocessingReport(dataset_id=sample_dataset.id, mode='auto')
        db.session.add(prep)
        fc = ForecastReport(dataset_id=sample_dataset.id, model_name='ARIMA', forecast_horizon=10)
        db.session.add(fc)
        comp = ComparisonReport(dataset_id=sample_dataset.id, best_model='RF', ranking='[]', comparison_metrics='{}')
        db.session.add(comp)
        rep = Report(dataset_id=sample_dataset.id)
        db.session.add(rep)
        db.session.commit()

        state = get_workflow_state(sample_dataset.id)
        assert 7 in state['completed']

    def test_reports_in_get_step_urls(self, db, app, sample_dataset):
        from services.workflow_service import get_workflow_state, get_step_urls, STEPS
        from models.preprocessing_report_model import PreprocessingReport
        from models.forecast_report_model import ForecastReport
        from models.comparison_report_model import ComparisonReport
        from models.report_model import Report
        prep = PreprocessingReport(dataset_id=sample_dataset.id, mode='auto')
        db.session.add(prep)
        fc = ForecastReport(dataset_id=sample_dataset.id, model_name='ARIMA', forecast_horizon=10)
        db.session.add(fc)
        comp = ComparisonReport(dataset_id=sample_dataset.id, best_model='RF', ranking='[]', comparison_metrics='{}')
        db.session.add(comp)
        rep = Report(dataset_id=sample_dataset.id)
        db.session.add(rep)
        db.session.commit()

        with app.test_request_context('/'):
            state = get_workflow_state(sample_dataset.id)
            urls = get_step_urls(sample_dataset.id, 7, state)
            assert urls['next'] is None
