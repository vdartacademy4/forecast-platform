import os, json
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, send_file)
from routes.auth import login_required
from services.dataset_service import get_dataset
from services.preprocessing_service import (
    run_automatic_preprocessing, run_manual_preprocessing,
    get_preprocessing_report, get_column_details
)
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls

preprocessing_bp = Blueprint('preprocessing', __name__)


@preprocessing_bp.route('/preprocessing-mode/<int:dataset_id>')
@login_required
def mode_selection(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))
    # Auto-redirect to preprocessing dashboard if already completed
    report = get_preprocessing_report(dataset_id)
    if report:
        return redirect(url_for('preprocessing.dashboard', dataset_id=dataset_id))
    workflow_state = get_workflow_state(dataset_id)
    step_urls = get_step_urls(dataset_id, 4, workflow_state)
    return render_template('preprocessing_mode.html', dataset=dataset,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@preprocessing_bp.route('/preprocessing-auto/<int:dataset_id>', methods=['POST'])
@login_required
def auto_preprocessing(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    results, error = run_automatic_preprocessing(dataset_id, session['user_id'])
    if error:
        flash(error, 'danger')
        return redirect(url_for('dataset.detail', dataset_id=dataset_id))

    log_activity(session['user_id'], 'preprocessing_auto_completed', 'preprocessing', dataset_id,
                 f'Automatic preprocessing completed for {dataset.file_name}')
    flash('Automatic preprocessing completed successfully!', 'success')
    return redirect(url_for('preprocessing.dashboard', dataset_id=dataset_id))


@preprocessing_bp.route('/preprocessing-manual/<int:dataset_id>', methods=['GET', 'POST'])
@login_required
def manual_preprocessing(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    if request.method == 'POST':
        config = {
            'missing_method': request.form.get('missing_method', 'none'),
            'outlier_method': request.form.get('outlier_method', 'none'),
            'outlier_columns': request.form.getlist('outlier_columns'),
            'encoding_method': request.form.get('encoding_method', 'none'),
            'encoding_columns': request.form.getlist('encoding_columns'),
            'scaling_method': request.form.get('scaling_method', 'none'),
            'scaling_columns': request.form.getlist('scaling_columns'),
            'date_column': request.form.get('date_column', ''),
            'date_features': request.form.getlist('date_features')
        }
        results, error = run_manual_preprocessing(dataset_id, session['user_id'], config)
        if error:
            flash(error, 'danger')
            return redirect(url_for('dataset.detail', dataset_id=dataset_id))
        log_activity(session['user_id'], 'preprocessing_manual_completed', 'preprocessing', dataset_id,
                     f'Manual preprocessing completed for {dataset.file_name}')
        flash('Manual preprocessing completed successfully!', 'success')
        return redirect(url_for('preprocessing.dashboard', dataset_id=dataset_id))

    numeric, categorical, date_cols, missing_info = get_column_details(dataset_id, session['user_id'])
    if numeric is None:
        flash('Unable to read dataset.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = get_step_urls(dataset_id, 4, workflow_state)
    return render_template('manual_preprocessing.html', dataset=dataset,
                           numeric_columns=numeric, categorical_columns=categorical,
                           date_columns=date_cols, missing_info=missing_info,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@preprocessing_bp.route('/preprocessing-dashboard/<int:dataset_id>')
@login_required
def dashboard(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_preprocessing_report(dataset_id)
    if not report:
        flash('No preprocessing results found. Please run preprocessing first.', 'warning')
        return redirect(url_for('preprocessing.mode_selection', dataset_id=dataset_id))

    steps = json.loads(report.steps_applied) if report.steps_applied else {}
    workflow_state = get_workflow_state(dataset_id)
    step_urls = get_step_urls(dataset_id, 4, workflow_state)

    return render_template('preprocessing_dashboard.html', dataset=dataset,
                           report=report, steps=steps, dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@preprocessing_bp.route('/preprocessing-summary/<int:dataset_id>')
@login_required
def summary(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_preprocessing_report(dataset_id)
    if not report:
        flash('No preprocessing results found.', 'warning')
        return redirect(url_for('preprocessing.mode_selection', dataset_id=dataset_id))

    steps = json.loads(report.steps_applied) if report.steps_applied else {}
    workflow_state = get_workflow_state(dataset_id)
    step_urls = get_step_urls(dataset_id, 4, workflow_state)

    return render_template('preprocessing_summary.html', dataset=dataset,
                           report=report, steps=steps, dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@preprocessing_bp.route('/download-processed/<int:dataset_id>')
@login_required
def download_processed(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = get_preprocessing_report(dataset_id)
    if not report or not report.output_file:
        flash('No processed dataset available.', 'warning')
        return redirect(url_for('preprocessing.mode_selection', dataset_id=dataset_id))

    fmt = request.args.get('format', 'csv')
    file_path = report.output_file
    if fmt == 'xlsx':
        file_path = file_path.replace('.csv', '.xlsx')

    if not os.path.exists(file_path):
        flash('Processed file not found.', 'warning')
        return redirect(url_for('preprocessing.summary', dataset_id=dataset_id))

    download_name = f'processed_{dataset.file_name.rsplit(".", 1)[0]}.{fmt}'
    return send_file(file_path, as_attachment=True, download_name=download_name)
