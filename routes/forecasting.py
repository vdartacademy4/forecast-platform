import os
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, send_file)
from routes.auth import login_required
from services.dataset_service import get_dataset
from services.forecasting_service import (
    run_automatic_forecasting, run_manual_forecasting,
    get_forecast_report, load_forecast_results,
    ALL_MODELS, TRADITIONAL_MODELS, ML_MODELS, DL_MODELS,
    _is_date_column, _is_numeric_timestamp
)
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls as wf_get_step_urls
from services.dataset_service import read_dataframe
from models.preprocessing_report_model import PreprocessingReport
import pandas as pd

forecasting_bp = Blueprint('forecasting', __name__)


def _get_column_info(dataset_id, user_id):
    dataset = get_dataset(dataset_id, user_id)
    if not dataset:
        return None, None, None, None

    prep = PreprocessingReport.query.filter_by(dataset_id=dataset_id).order_by(
        PreprocessingReport.created_at.desc()
    ).first()
    if prep and prep.output_file and os.path.exists(prep.output_file):
        file_path = prep.output_file
        ext = 'csv'
    else:
        file_path = dataset.file_path
        ext = dataset.file_name.rsplit('.', 1)[1].lower()

    if ext == 'csv':
        df = read_dataframe(file_path, 'csv')
    else:
        df = read_dataframe(file_path, ext)

    if df is None:
        return None, None, None, None

    numeric = list(df.select_dtypes(include=['number']).columns)
    categorical = list(df.select_dtypes(include=['object', 'category']).columns)
    date_cols = []
    for col in df.columns:
        if _is_date_column(df[col]) or _is_numeric_timestamp(df[col]):
            date_cols.append(str(col))
    return numeric, categorical, date_cols, df.columns.tolist()


@forecasting_bp.route('/forecasting/<int:dataset_id>')
@login_required
def mode_selection(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_forecast_report(dataset_id)
    if report:
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    return redirect(url_for('forecasting.setup', dataset_id=dataset_id))


@forecasting_bp.route('/forecasting/setup/<int:dataset_id>')
@login_required
def setup(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_forecast_report(dataset_id)
    if report:
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    numeric, categorical, date_cols, all_cols = _get_column_info(dataset_id, session['user_id'])
    if numeric is None:
        flash('Unable to read dataset.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 5, workflow_state)
    return render_template('forecast_setup.html', dataset=dataset,
                           numeric_columns=numeric, date_columns=date_cols,
                           all_columns=all_cols,
                           traditional_models=list(TRADITIONAL_MODELS.keys()),
                           ml_models=list(ML_MODELS.keys()),
                           dl_models=list(DL_MODELS.keys()),
                           all_models=list(ALL_MODELS.keys()),
                           dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@forecasting_bp.route('/forecasting/train/<int:dataset_id>', methods=['POST'])
@login_required
def train(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_forecast_report(dataset_id)
    if report:
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    horizon = request.form.get('horizon', 30, type=int)
    test_ratio_str = request.form.get('test_ratio', '0.2')
    test_ratio = float(test_ratio_str) if test_ratio_str else 0.2
    target_col = request.form.get('target_column', '')
    date_col = request.form.get('date_column', '')
    model_name = request.form.get('model_name', '')

    if model_name:
        if not target_col:
            flash('Please select a target column.', 'warning')
            return redirect(url_for('forecasting.setup', dataset_id=dataset_id))
        results, error = run_manual_forecasting(
            dataset_id, session['user_id'],
            target_col=target_col, date_col=date_col,
            model_name=model_name,
            horizon=horizon, test_ratio=test_ratio
        )
        log_type = 'forecast_manual_completed'
    else:
        results, error = run_automatic_forecasting(
            dataset_id, session['user_id'], horizon=horizon,
            test_ratio=test_ratio
        )
        log_type = 'forecast_auto_completed'

    if error:
        flash(error, 'danger')
        return redirect(url_for('forecasting.setup', dataset_id=dataset_id))

    log_activity(session['user_id'], log_type, 'forecasting', dataset_id,
                 f'Forecasting completed for {dataset.file_name}')
    flash('Forecasting completed successfully!', 'success')
    return redirect(url_for('forecasting.results', dataset_id=dataset_id))


@forecasting_bp.route('/forecasting/results/<int:dataset_id>')
@login_required
def results(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    results_data = load_forecast_results(dataset_id)
    if not results_data:
        report = get_forecast_report(dataset_id)
        if report:
            return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))
        flash('No forecasting results found. Please run forecasting first.', 'warning')
        return redirect(url_for('forecasting.setup', dataset_id=dataset_id))

    report = get_forecast_report(dataset_id)
    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 5, workflow_state)

    return render_template('forecast_results.html', dataset=dataset,
                           results=results_data, report=report,
                           dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@forecasting_bp.route('/forecasting/dashboard/<int:dataset_id>')
@login_required
def dashboard(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_forecast_report(dataset_id)
    if not report:
        flash('No forecasting results found.', 'warning')
        return redirect(url_for('forecasting.setup', dataset_id=dataset_id))

    results_data = load_forecast_results(dataset_id) or {}
    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 5, workflow_state)

    return render_template('forecast_dashboard.html', dataset=dataset,
                           report=report, results=results_data,
                           dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@forecasting_bp.route('/forecasting/download/<int:dataset_id>')
@login_required
def download_forecast(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_forecast_report(dataset_id)
    if not report or not report.forecast_file:
        flash('No forecast file available.', 'warning')
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    if not os.path.exists(report.forecast_file):
        flash('Forecast file not found.', 'warning')
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    return send_file(report.forecast_file, as_attachment=True,
                     download_name=f'forecast_{dataset.file_name.rsplit(".", 1)[0]}.csv')
