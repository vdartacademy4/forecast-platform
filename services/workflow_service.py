from flask import url_for
from models.validation_report_model import ValidationReport
from models.eda_report_model import EDAReport
from models.preprocessing_report_model import PreprocessingReport
from models.forecast_report_model import ForecastReport
from models.comparison_report_model import ComparisonReport

STEPS = [
    {'number': 1, 'name': 'Upload', 'icon': 'fa-upload', 'route': 'dataset.upload'},
    {'number': 2, 'name': 'Validation', 'icon': 'fa-clipboard-check', 'route': 'dataset.preview'},
    {'number': 3, 'name': 'EDA', 'icon': 'fa-chart-pie', 'route': 'eda.mode_selection'},
    {'number': 4, 'name': 'Preprocessing', 'icon': 'fa-filter', 'route': 'preprocessing.mode_selection'},
    {'number': 5, 'name': 'Forecasting', 'icon': 'fa-chart-line', 'route': 'forecasting.mode_selection'},
    {'number': 6, 'name': 'Compare', 'icon': 'fa-balance-scale', 'route': 'comparison.compare_dashboard'},
    {'number': 7, 'name': 'Reports', 'icon': 'fa-file-alt', 'route': 'reports.view_report'},
]


def get_workflow_state(dataset_id):
    completed = [1]
    locked = []

    report = ValidationReport.query.filter_by(dataset_id=dataset_id).order_by(
        ValidationReport.created_at.desc()
    ).first()
    if report and report.validation_status == 'completed':
        completed.append(2)

    eda = EDAReport.query.filter_by(dataset_id=dataset_id).order_by(
        EDAReport.generated_at.desc()
    ).first()
    if eda:
        completed.append(3)

    prep = PreprocessingReport.query.filter_by(dataset_id=dataset_id).order_by(
        PreprocessingReport.created_at.desc()
    ).first()
    if prep:
        completed.append(4)

    if 4 in completed:
        forecast = ForecastReport.query.filter_by(dataset_id=dataset_id).order_by(
            ForecastReport.created_at.desc()
        ).first()
        if forecast:
            completed.append(5)
    else:
        locked.append(5)

    if 5 in completed:
        compare = ComparisonReport.query.filter_by(dataset_id=dataset_id).order_by(
            ComparisonReport.created_at.desc()
        ).first()
        if compare:
            completed.append(6)
        else:
            pass
    else:
        locked.append(6)

    if 6 in completed:
        from models.report_model import Report
        rep = Report.query.filter_by(dataset_id=dataset_id).order_by(
            Report.created_at.desc()
        ).first()
        if rep:
            completed.append(7)
    else:
        locked.append(7)

    current = 1
    for step in STEPS:
        if step['number'] not in completed and step['number'] not in locked:
            current = step['number']
            break
    else:
        current = 7

    return {'steps': STEPS, 'completed': completed, 'current': current, 'locked': locked}


def get_step_urls(dataset_id, page_step, workflow_state):
    completed = workflow_state['completed']
    locked = workflow_state['locked']

    prev = None
    for i in range(page_step - 2, -1, -1):
        step = STEPS[i]
        if step['number'] in locked:
            continue
        prev = _build_step_url(step, dataset_id, completed)
        break

    next_url = None
    for i in range(page_step, len(STEPS)):
        step = STEPS[i]
        if step['number'] in locked:
            continue
        next_url = _build_step_url(step, dataset_id, completed)
        break

    return {'prev': prev, 'next': next_url}


def _build_step_url(step, dataset_id, completed=None):
    if completed is None:
        completed = []

    route = step.get('route')
    if not route:
        return None
    if step['number'] == 1:
        return url_for('dataset.upload')
    elif step['number'] == 2:
        report = ValidationReport.query.filter_by(dataset_id=dataset_id).order_by(
            ValidationReport.created_at.desc()
        ).first()
        if report and report.validation_status == 'completed':
            return url_for('dataset.validation_report', report_id=report.id)
        return url_for('dataset.preview', dataset_id=dataset_id)
    elif step['number'] == 3:
        if 3 in completed:
            return url_for('eda.eda_dashboard', dataset_id=dataset_id)
        return url_for('eda.mode_selection', dataset_id=dataset_id)
    elif step['number'] == 4:
        if 4 in completed:
            return url_for('preprocessing.dashboard', dataset_id=dataset_id)
        return url_for('preprocessing.mode_selection', dataset_id=dataset_id)
    elif step['number'] == 5:
        from models.forecast_report_model import ForecastReport
        fc = ForecastReport.query.filter_by(dataset_id=dataset_id).order_by(
            ForecastReport.created_at.desc()
        ).first()
        if fc:
            return url_for('forecasting.dashboard', dataset_id=dataset_id)
        return url_for('forecasting.mode_selection', dataset_id=dataset_id)
    elif step['number'] == 6:
        from models.comparison_report_model import ComparisonReport
        comp = ComparisonReport.query.filter_by(dataset_id=dataset_id).order_by(
            ComparisonReport.created_at.desc()
        ).first()
        if comp:
            return url_for('comparison.compare_results', dataset_id=dataset_id)
        return url_for('comparison.compare_dashboard', dataset_id=dataset_id)
    elif step['number'] == 7:
        return url_for('reports.view_report', dataset_id=dataset_id)
    return url_for(route, dataset_id=dataset_id)
