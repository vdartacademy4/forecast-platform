import os
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, session, send_file, jsonify)
from routes.auth import login_required
from services.dataset_service import get_dataset
from services.comparison_service import (
    get_comparison_context, get_model_actual_predicted,
    generate_comparison_csv, generate_comparison_excel,
    generate_comparison_pdf, load_comparison_results
)
from services.activity_service import log_activity
from services.workflow_service import get_workflow_state, get_step_urls as wf_get_step_urls
from services.forecasting_service import load_forecast_results

comparison_bp = Blueprint('comparison', __name__)


@comparison_bp.route('/compare/<int:dataset_id>')
@login_required
def compare_dashboard(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    forecast_data = load_forecast_results(dataset_id)
    if not forecast_data:
        flash('No forecast results found. Please run forecasting first.', 'warning')
        return redirect(url_for('forecasting.setup', dataset_id=dataset_id))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 6, workflow_state)

    existing = load_comparison_results(dataset_id)
    if existing:
        return redirect(url_for('comparison.compare_results', dataset_id=dataset_id))

    comparison, error = build_comparison(dataset_id)
    if error:
        flash(error, 'danger')
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))

    log_activity(session['user_id'], 'compared_models', 'comparison', dataset_id)
    return redirect(url_for('comparison.compare_results', dataset_id=dataset_id))


def build_comparison(dataset_id):
    from services.comparison_service import build_comparison as bc
    return bc(dataset_id)


@comparison_bp.route('/compare/results/<int:dataset_id>')
@login_required
def compare_results(dataset_id):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    comparison, error = get_comparison_context(dataset_id)
    if error:
        flash(error, 'danger')
        return redirect(url_for('forecasting.dashboard', dataset_id=dataset_id))
    if comparison is None:
        flash('No comparison data available.', 'warning')
        return redirect(url_for('forecasting.setup', dataset_id=dataset_id))

    workflow_state = get_workflow_state(dataset_id)
    step_urls = wf_get_step_urls(dataset_id, 6, workflow_state)

    chart_json = _chart_data_to_plotly(comparison.get('chart_data', {}))

    prev_url = step_urls.get('prev')
    next_url = step_urls.get('next')

    return render_template(
        'compare_results.html',
        dataset=dataset,
        comparison=comparison,
        chart_json=chart_json,
        workflow_state=workflow_state,
        step_urls=step_urls,
        prev_url=prev_url,
        next_url=next_url,
        enumerate=enumerate,
    )


def _chart_data_to_plotly(chart_data):
    import json as _json
    labels = chart_data.get('labels', [])
    charts = {}

    if labels:
        charts['rmse'] = _make_bar_chart(labels, chart_data.get('rmse', []),
                                         'RMSE Comparison', 'RMSE', '#FF6384')
        charts['mae'] = _make_bar_chart(labels, chart_data.get('mae', []),
                                        'MAE Comparison', 'MAE', '#36A2EB')
        charts['mape'] = _make_bar_chart(labels, chart_data.get('mape', []),
                                         'MAPE Comparison', 'MAPE (%)', '#FFCE56')
        charts['r2'] = _make_bar_chart(labels, chart_data.get('r2', []),
                                       'R\u00b2 Score Comparison', 'R\u00b2', '#4BC0C0')
        charts['training_time'] = _make_bar_chart(
            labels, chart_data.get('training_time', []),
            'Training Time Comparison', 'Time (s)', '#9966FF')

    return _json.dumps(charts, default=str)


def _make_bar_chart(labels, values, title, ylabel, color):
    return {
        'data': [{
            'x': labels,
            'y': values,
            'type': 'bar',
            'marker': {'color': color},
            'text': [f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in values],
            'textposition': 'outside',
        }],
        'layout': {
            'title': title,
            'yaxis': {'title': ylabel},
            'xaxis': {'title': 'Model'},
            'height': 400,
            'margin': {'t': 50, 'b': 80, 'l': 60, 'r': 20},
            'template': 'plotly_white',
        }
    }


@comparison_bp.route('/compare/api/model/<int:dataset_id>/<model_name>')
@login_required
def api_model_data(dataset_id, model_name):
    data, error = get_model_actual_predicted(dataset_id, model_name)
    if error:
        return jsonify({'error': error}), 404
    return jsonify(data)


@comparison_bp.route('/compare/download/<int:dataset_id>/<file_format>')
@login_required
def download_comparison(dataset_id, file_format):
    dataset = get_dataset(dataset_id, session['user_id'])
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dataset.list_datasets'))

    generators = {
        'csv': generate_comparison_csv,
        'xlsx': generate_comparison_excel,
        'pdf': generate_comparison_pdf,
    }

    gen = generators.get(file_format)
    if not gen:
        flash(f'Unsupported format: {file_format}', 'danger')
        return redirect(url_for('comparison.compare_results', dataset_id=dataset_id))

    path, error = gen(dataset_id)
    if error:
        flash(error, 'danger')
        return redirect(url_for('comparison.compare_results', dataset_id=dataset_id))

    mime_types = {
        'csv': 'text/csv',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pdf': 'application/pdf',
    }
    return send_file(
        path,
        mimetype=mime_types.get(file_format, 'application/octet-stream'),
        as_attachment=True,
        download_name=f'comparison_results_{dataset_id}.{file_format}'
    )
