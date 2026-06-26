import os
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, send_file, jsonify)
from routes.auth import login_required
from services.dataset_service import get_dataset
from services.report_service import (
    get_report_context, generate_report_csv,
    generate_report_excel, generate_report_pdf,
    save_analysis_history
)
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls as wf_get_step_urls
from datetime import datetime

reports_bp = Blueprint('reports', __name__)


@reports_bp.route('/reports/<int:dataset_id>')
@login_required
def view_report(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    context, error = get_report_context(dataset_id)
    if error:
        flash(error, 'danger')
        return redirect(url_for('dataset.detail', dataset_id=dataset_id))
    if context is None:
        flash('No report data available. Complete the workflow first.', 'warning')
        return redirect(url_for('dataset.detail', dataset_id=dataset_id))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 7, workflow_state)

    prev_url = step_urls.get('prev')
    next_url = step_urls.get('next')

    log_activity(session['user_id'], 'viewed_report', 'report', dataset_id)
    save_analysis_history(dataset_id, session['user_id'])

    return render_template(
        'report_view.html',
        dataset=dataset,
        report=context,
        workflow_state=workflow_state,
        step_urls=step_urls,
        prev_url=prev_url,
        next_url=next_url,
        now=datetime.utcnow(),
        enumerate=enumerate,
    )


@reports_bp.route('/workflow/completed/<int:dataset_id>')
@login_required
def workflow_completed(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    context, error = get_report_context(dataset_id)
    if error or not context:
        flash('Complete the workflow first.', 'warning')
        return redirect(url_for('dataset.detail', dataset_id=dataset_id))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 7, workflow_state)

    prev_url = step_urls.get('prev')
    next_url = step_urls.get('next')

    log_activity(session['user_id'], 'workflow_completed', 'workflow', dataset_id)
    save_analysis_history(dataset_id, session['user_id'])

    return render_template(
        'workflow_completed.html',
        dataset=dataset,
        report=context,
        workflow_state=workflow_state,
        step_urls=step_urls,
        prev_url=prev_url,
        next_url=next_url,
        now=datetime.utcnow(),
    )


@reports_bp.route('/reports/download/<int:dataset_id>/<file_format>')
@login_required
def download_report(dataset_id, file_format):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    generators = {
        'csv': generate_report_csv,
        'xlsx': generate_report_excel,
        'pdf': generate_report_pdf,
    }

    gen = generators.get(file_format)
    if not gen:
        flash(f'Unsupported format: {file_format}', 'danger')
        return redirect(url_for('reports.view_report', dataset_id=dataset_id))

    path, error = gen(dataset_id)
    if error:
        flash(error, 'danger')
        return redirect(url_for('reports.view_report', dataset_id=dataset_id))

    mime_types = {
        'csv': 'text/csv',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pdf': 'application/pdf',
    }
    return send_file(
        path,
        mimetype=mime_types.get(file_format, 'application/octet-stream'),
        as_attachment=True,
        download_name=f'forecast_report_{dataset_id}.{file_format}'
    )
