import os
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session,
                   current_app, send_file)
from routes.auth import login_required
from services.dataset_service import get_dataset
from services.eda_service import (run_automatic_eda, run_manual_eda,
                                  generate_html_report, get_eda_report,
                                  save_eda_results, load_eda_results,
                                  _get_output_dir)
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls as wf_get_step_urls

eda_bp = Blueprint('eda', __name__)


def _get_results(dataset_id, user_id):
    dataset = get_dataset(dataset_id, user_id)
    if not dataset:
        flash('Dataset not found.', 'danger')
        return None, None
    results = load_eda_results(dataset_id)
    if not results:
        flash('No EDA results found. Please run EDA first.', 'warning')
        return None, None
    results['output_dir'] = _get_output_dir(dataset_id)
    return dataset, results


@eda_bp.route('/eda-mode/<int:dataset_id>')
@login_required
def mode_selection(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))
    # Auto-redirect to EDA dashboard if EDA already completed
    from models.eda_report_model import EDAReport
    if EDAReport.query.filter_by(dataset_id=dataset_id).first():
        return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))
    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
    return render_template('eda_mode.html', dataset=dataset,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@eda_bp.route('/eda-auto/<int:dataset_id>', methods=['POST'])
@login_required
def auto_eda(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))
    results, error = run_automatic_eda(dataset_id, session['user_id'])
    if error:
        flash(error, 'danger')
        return redirect(url_for('dataset.detail', dataset_id=dataset_id))
    save_eda_results(dataset_id, results)
    log_activity(session['user_id'], 'eda_auto_completed', 'eda', dataset_id,
                 f'Automatic EDA completed for {dataset.file_name}')
    flash('Automatic EDA completed successfully!', 'success')
    return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))


@eda_bp.route('/eda-manual/<int:dataset_id>', methods=['GET', 'POST'])
@login_required
def manual_eda(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    if request.method == 'POST':
        selections = {
            'statistics': request.form.getlist('statistics'),
            'analysis': request.form.getlist('analysis'),
            'charts': request.form.getlist('charts')
        }
        if not any(selections.values()):
            flash('Please select at least one analysis option.', 'warning')
            workflow_state = get_workflow_state(dataset_id)
            step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
            return render_template('manual_eda.html', dataset=dataset,
                                   dataset_id=dataset_id, current_step=workflow_state['current'],
                                   workflow_state=workflow_state, step_urls=step_urls)
        results, error = run_manual_eda(dataset_id, session['user_id'], selections)
        if error:
            flash(error, 'danger')
            return redirect(url_for('dataset.detail', dataset_id=dataset_id))
        save_eda_results(dataset_id, results)
        log_activity(session['user_id'], 'eda_manual_completed', 'eda', dataset_id,
                     f'Manual EDA completed for {dataset.file_name}')
        flash('Manual EDA completed successfully!', 'success')
        return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
    return render_template('manual_eda.html', dataset=dataset,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@eda_bp.route('/eda-dashboard/<int:dataset_id>')
@login_required
def eda_dashboard(dataset_id):
    dataset, results = _get_results(dataset_id, session['user_id'])
    if not dataset or not results:
        return redirect(url_for('eda.mode_selection', dataset_id=dataset_id))
    charts = results.get('charts', {})
    eda_mode = 'automatic'
    report = None
    if results.get('report_id'):
        report = get_eda_report(results['report_id'])
        if report:
            eda_mode = report.eda_mode

    # Ensure dataset_overview exists with fallbacks from dataset model
    overview = results.get('dataset_overview', {})
    if not overview or not overview.get('rows'):
        overview = {
            'rows': dataset.rows_count or 0,
            'columns': dataset.columns_count or 0,
            'memory_kb': 0,
            'numeric_count': len(results.get('statistics', {})),
            'categorical_count': len(results.get('categorical_analysis', {})),
            'date_count': 0
        }
    results['dataset_overview'] = overview

    # Compute data quality score and smart insights
    from services.eda_service import compute_data_quality_score, generate_smart_insights
    quality_score = compute_data_quality_score(results)
    smart_insights = generate_smart_insights(results)

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
    return render_template('eda_dashboard.html', dataset=dataset, results=results,
                           charts=charts, eda_mode=eda_mode, report=report,
                           quality_score=quality_score, smart_insights=smart_insights,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@eda_bp.route('/eda-summary/<int:dataset_id>')
@login_required
def eda_summary(dataset_id):
    dataset, results = _get_results(dataset_id, session['user_id'])
    if not dataset or not results:
        return redirect(url_for('eda.mode_selection', dataset_id=dataset_id))
    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
    return render_template('eda_summary.html', dataset=dataset, results=results,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@eda_bp.route('/eda-charts/<int:dataset_id>')
@login_required
def eda_charts(dataset_id):
    dataset, results = _get_results(dataset_id, session['user_id'])
    if not dataset or not results:
        return redirect(url_for('eda.mode_selection', dataset_id=dataset_id))
    charts = results.get('charts', {})
    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 3, workflow_state)
    return render_template('eda_charts.html', dataset=dataset, charts=charts,
                           dataset_id=dataset_id, current_step=workflow_state['current'],
                           workflow_state=workflow_state, step_urls=step_urls)


@eda_bp.route('/generate-eda-report/<int:dataset_id>', methods=['POST'])
@login_required
def generate_report(dataset_id):
    dataset, results = _get_results(dataset_id, session['user_id'])
    if not dataset or not results:
        return redirect(url_for('eda.mode_selection', dataset_id=dataset_id))

    output_dir = results.get('output_dir')
    if not output_dir:
        flash('Report directory not found.', 'danger')
        return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))

    report_path = generate_html_report(dataset, results, output_dir)
    eda_report = get_eda_report(results.get('report_id'))
    if eda_report:
        eda_report.report_path = report_path
        from database import db
        db.session.commit()

    flash('EDA report generated successfully!', 'success')
    return redirect(url_for('eda.download_report', dataset_id=dataset_id))


@eda_bp.route('/download-eda-report/<int:dataset_id>')
@login_required
def download_report(dataset_id):
    dataset, results = _get_results(dataset_id, session['user_id'])
    if not dataset or not results:
        return redirect(url_for('eda.mode_selection', dataset_id=dataset_id))
    output_dir = results.get('output_dir')
    if not output_dir:
        flash('Report not found.', 'danger')
        return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))
    report_path = os.path.join(output_dir, 'eda_report.html')
    if not os.path.exists(report_path):
        flash('Report file not found. Please generate it first.', 'warning')
        return redirect(url_for('eda.eda_dashboard', dataset_id=dataset_id))
    return send_file(report_path, as_attachment=True,
                     download_name=f'EDA_Report_{dataset.file_name}.html')
