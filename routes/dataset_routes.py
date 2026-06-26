import json
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, current_app)
from routes.auth import login_required
from services.dataset_service import (
    upload_dataset, get_user_datasets, get_dataset,
    delete_dataset, get_preview_data
)
from services.validation_service import run_validation, get_report
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls as wf_get_step_urls
from utils.file_utils import get_file_extension

dataset_bp = Blueprint('dataset', __name__)


@dataset_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return render_template('upload_dataset.html')

        file = request.files['file']
        if not file.filename:
            flash('No file selected.', 'danger')
            return render_template('upload_dataset.html')

        dataset, error = upload_dataset(file, session['user_id'])
        if error:
            flash(error, 'danger')
            return render_template('upload_dataset.html')

        log_activity(
            session['user_id'], 'uploaded_dataset', 'dataset', dataset.id,
            f'Uploaded {dataset.file_name} ({dataset.rows_count} rows, {dataset.columns_count} columns)'
        )
        flash(f'Dataset "{dataset.file_name}" uploaded successfully! '
              f'({dataset.rows_count} rows, {dataset.columns_count} columns)', 'success')
        return redirect(url_for('dataset.preview', dataset_id=dataset.id))

    return render_template('upload_dataset.html')


@dataset_bp.route('/datasets')
@login_required
def list_datasets():
    page = request.args.get('page', 1, type=int)
    datasets, total = get_user_datasets(session['user_id'], page=page)

    for ds in datasets:
        report = get_report(ds.id)
        ds.validation_status = report.validation_status if report else 'pending'

    return render_template('upload_history.html', datasets=datasets, total=total, page=page)


@dataset_bp.route('/dataset/<int:dataset_id>')
@login_required
def detail(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    extension = get_file_extension(dataset.file_name)
    columns, data, dtypes = get_preview_data(dataset.file_path, extension, rows=10)
    report = get_report(dataset.id)

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 2, workflow_state)
    return render_template('dataset_preview.html',
                           dataset=dataset,
                           columns=columns,
                           data=data,
                           dtypes=dtypes,
                           report=report,
                           dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state,
                           step_urls=step_urls)


@dataset_bp.route('/dataset/<int:dataset_id>/preview')
@login_required
def preview(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    extension = get_file_extension(dataset.file_name)
    columns, data, dtypes = get_preview_data(dataset.file_path, extension, rows=50)
    report = get_report(dataset.id)

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 2, workflow_state)
    return render_template('dataset_preview.html',
                           dataset=dataset,
                           columns=columns,
                           data=data,
                           dtypes=dtypes,
                           report=report,
                           full_preview=True,
                           dataset_id=dataset_id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state,
                           step_urls=step_urls)


@dataset_bp.route('/dataset/<int:dataset_id>/validate', methods=['POST'])
@login_required
def validate(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    report = run_validation(dataset)
    if report.validation_status == 'failed':
        log_activity(session['user_id'], 'validation_failed', 'dataset', dataset_id,
                     f'Validation failed for {dataset.file_name}')
        flash('Validation failed. The file may be corrupted.', 'danger')
        return redirect(url_for('dataset.preview', dataset_id=dataset_id))

    log_activity(session['user_id'], 'validation_completed', 'dataset', dataset_id,
                 f'Validation completed for {dataset.file_name}')
    flash('Validation completed successfully!', 'success')
    return redirect(url_for('dataset.validation_report', report_id=report.id))


@dataset_bp.route('/validation-report/<int:report_id>')
@login_required
def validation_report(report_id):
    from models.validation_report_model import ValidationReport
    report = ValidationReport.query.get(report_id)
    if not report:
        flash('Validation report not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    dataset = get_dataset(report.dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    missing_values = json.loads(report.missing_values) if report.missing_values else []
    empty_columns = json.loads(report.empty_columns) if report.empty_columns else []
    duplicate_columns = json.loads(report.duplicate_columns) if report.duplicate_columns else []
    date_columns = json.loads(report.date_columns) if report.date_columns else []
    numeric_columns = json.loads(report.numeric_columns) if report.numeric_columns else []
    categorical_columns = json.loads(report.categorical_columns) if report.categorical_columns else []
    column_types = json.loads(report.column_types) if report.column_types else {}

    total_missing = sum(m['count'] for m in missing_values)
    issues_count = len(missing_values) + (1 if report.duplicate_rows > 0 else 0) + len(empty_columns) + len(duplicate_columns)

    workflow_state = get_workflow_state(dataset.id)
    step_urls = wf_get_step_urls(dataset.id, 2, workflow_state)
    return render_template('validation_report.html',
                           report=report,
                           dataset=dataset,
                           missing_values=missing_values,
                           total_missing=total_missing,
                           empty_columns=empty_columns,
                           duplicate_columns=duplicate_columns,
                           date_columns=date_columns,
                           numeric_columns=numeric_columns,
                           categorical_columns=categorical_columns,
                           column_types=column_types,
                           issues_count=issues_count,
                           dataset_id=dataset.id,
                           current_step=workflow_state['current'],
                           workflow_state=workflow_state,
                           step_urls=step_urls)


@dataset_bp.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    file_name = dataset.file_name
    log_activity(session['user_id'], 'deleted_dataset', 'dataset', dataset_id,
                 f'Deleted dataset {file_name}')
    delete_dataset(dataset)
    flash(f'Dataset "{file_name}" deleted successfully.', 'info')
    return redirect(url_for('dataset.list_datasets'))
