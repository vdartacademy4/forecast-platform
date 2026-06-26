import os
import json
import ast
import numpy as np
import pandas as pd
from flask import current_app
from database import db
from models.report_model import Report
from services.forecasting_service import load_forecast_results, _get_forecast_dir
from services.comparison_service import load_comparison_results


def get_report(dataset_id):
    return Report.query.filter_by(dataset_id=dataset_id).order_by(
        Report.created_at.desc()
    ).first()


def _get_report_dir(dataset_id):
    d = _get_forecast_dir(dataset_id)
    report_dir = os.path.join(d, 'report')
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def _get_results_path(dataset_id):
    return os.path.join(_get_report_dir(dataset_id), 'report_results.json')


def load_report_results(dataset_id):
    path = _get_results_path(dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report_results(dataset_id, results):
    path = _get_results_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_data = {}
    for k, v in results.items():
        try:
            json.dumps(v)
            save_data[k] = v
        except (TypeError, ValueError):
            save_data[k] = str(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)


def _get_workflow_time(dataset_id):
    from models.activity_log_model import ActivityLog
    upload_log = ActivityLog.query.filter_by(
        dataset_id=dataset_id, action='uploaded_dataset'
    ).order_by(ActivityLog.timestamp.asc()).first()
    report_log = ActivityLog.query.filter_by(
        dataset_id=dataset_id, action='viewed_report'
    ).order_by(ActivityLog.timestamp.desc()).first()
    if upload_log and report_log:
        delta = report_log.timestamp - upload_log.timestamp
        return round(delta.total_seconds() / 60, 1)
    return None


def _generate_smart_insights(ctx):
    insights = []

    v = ctx.get('validation')
    if v:
        mv_count = sum(m.get('count', 0) for m in (v.get('missing_values') or []))
        if mv_count == 0:
            insights.append('Dataset quality is good — no missing values detected.')
        elif mv_count < 100:
            insights.append(f'Dataset has minor missing values ({mv_count} total) — handled during preprocessing.')
        else:
            insights.append(f'Dataset has {mv_count} missing values — successfully handled during preprocessing.')
        if v.get('duplicate_rows', 0) == 0:
            insights.append('No duplicate rows found — dataset integrity is intact.')
        if v.get('validation_status') == 'completed':
            insights.append('Validation completed successfully — data structure verified.')

    prep = ctx.get('preprocessing')
    if prep:
        steps = prep.get('steps_applied') or {}
        if steps.get('missing_method') and steps['missing_method'] != 'none':
            insights.append(f'Missing values handled using {steps["missing_method"].replace("_", " ")} method.')
        if steps.get('outlier_method') and steps['outlier_method'] != 'none':
            insights.append(f'Outliers treated with {steps["outlier_method"].replace("_", " ")} method.')
        insights.append('Data preprocessing completed — dataset is ready for modeling.')

    best = ctx.get('best_model')
    if best and best != 'N/A':
        insights.append(f'{best} performs best among all trained models.')
        rmse_val = ctx.get('best_rmse')
        if rmse_val is not None:
            insights.append(f'Best model RMSE: {rmse_val:.4f} — forecast confidence is high.')
    else:
        insights.append('Forecast modeling completed successfully.')

    if ctx.get('forecast_horizon'):
        insights.append(f'Forecast generated for {ctx["forecast_horizon"]} future periods — dataset is ready for business use.')

    ctx['smart_insights'] = insights
    return insights


def generate_full_report(dataset_id):
    from models.dataset_model import Dataset
    from models.validation_report_model import ValidationReport
    from models.eda_report_model import EDAReport
    from models.preprocessing_report_model import PreprocessingReport
    from models.forecast_report_model import ForecastReport
    from models.comparison_report_model import ComparisonReport
    from models.activity_log_model import ActivityLog
    from services.workflow_service import STEPS
    from datetime import datetime

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return None, 'Dataset not found.'

    valid = ValidationReport.query.filter_by(dataset_id=dataset_id).order_by(
        ValidationReport.created_at.desc()).first()
    eda = EDAReport.query.filter_by(dataset_id=dataset_id).order_by(
        EDAReport.generated_at.desc()).first()
    prep = PreprocessingReport.query.filter_by(dataset_id=dataset_id).order_by(
        PreprocessingReport.created_at.desc()).first()
    fc = ForecastReport.query.filter_by(dataset_id=dataset_id).order_by(
        ForecastReport.created_at.desc()).first()
    comp = ComparisonReport.query.filter_by(dataset_id=dataset_id).order_by(
        ComparisonReport.created_at.desc()).first()

    comp_results = load_comparison_results(dataset_id)
    forecast_results = load_forecast_results(dataset_id)

    best_model_name = comp.best_model if comp else (comp_results.get('best_model') if comp_results else None)
    best_explanation = ''
    ranking = []
    insights = []
    chart_data = {}
    results_table = []
    best_rmse = None
    best_mae = None
    best_mape = None
    best_r2 = None
    if comp_results:
        best_explanation = comp_results.get('best_explanation', '')
        ranking = comp_results.get('ranking', [])
        insights = comp_results.get('insights', [])
        chart_data = comp_results.get('chart_data', {})
        results_table = comp_results.get('results_table', [])
        if ranking:
            top = next((m for m in ranking if m.get('status') == 'success'), None)
            if top:
                best_rmse = top.get('rmse')
                best_mae = top.get('mae')
                best_mape = top.get('mape')
                best_r2 = top.get('r2')

    steps_summary = []
    for s in STEPS:
        detail = ''
        if s['number'] == 1:
            detail = f'Dataset: {dataset.file_name}'
        elif s['number'] == 2:
            detail = f'Status: {valid.validation_status if valid else "N/A"}'
        elif s['number'] == 3:
            detail = f'Mode: {eda.eda_mode if eda else "N/A"}'
        elif s['number'] == 4:
            detail = f'Mode: {prep.mode if prep else "N/A"}'
        elif s['number'] == 5:
            detail = f'Model: {fc.model_name if fc else "N/A"}'
        elif s['number'] == 6:
            detail = f'Best: {comp.best_model if comp else "N/A"}'
        elif s['number'] == 7:
            detail = 'Report generated'
        steps_summary.append({'number': s['number'], 'name': s['name'], 'detail': detail})

    validation_data = None
    if valid:
        validation_data = {
            'total_rows': valid.total_rows,
            'total_columns': valid.total_columns,
            'missing_values': ast.literal_eval(valid.missing_values) if isinstance(valid.missing_values, str) else valid.missing_values,
            'duplicate_rows': valid.duplicate_rows,
            'empty_columns': ast.literal_eval(valid.empty_columns) if isinstance(valid.empty_columns, str) else valid.empty_columns,
            'duplicate_columns': ast.literal_eval(valid.duplicate_columns) if isinstance(valid.duplicate_columns, str) else valid.duplicate_columns,
            'date_columns': ast.literal_eval(valid.date_columns) if isinstance(valid.date_columns, str) else valid.date_columns,
            'numeric_columns': ast.literal_eval(valid.numeric_columns) if isinstance(valid.numeric_columns, str) else valid.numeric_columns,
            'categorical_columns': ast.literal_eval(valid.categorical_columns) if isinstance(valid.categorical_columns, str) else valid.categorical_columns,
            'validation_status': valid.validation_status,
        }

    eda_data = None
    if eda:
        eda_data = {
            'eda_mode': eda.eda_mode,
            'total_charts': eda.total_charts,
        }
        if valid:
            mv = ast.literal_eval(valid.missing_values) if isinstance(valid.missing_values, str) else valid.missing_values
            eda_data['missing_values'] = sum(m.get('count', 0) for m in (mv or []))
        data_quality_score = 0
        if valid and valid.validation_status == 'completed':
            score = 100
            if valid.duplicate_rows > 0:
                score -= 10
            mv_count = sum(m.get('count', 0) for m in (ast.literal_eval(valid.missing_values) if isinstance(valid.missing_values, str) else valid.missing_values))
            if mv_count > 0:
                score -= min(20, mv_count // 10)
            if ast.literal_eval(valid.empty_columns) if isinstance(valid.empty_columns, str) else valid.empty_columns:
                score -= 15
            data_quality_score = max(0, score)
        eda_data['data_quality_score'] = data_quality_score

    preprocessing_data = None
    if prep:
        steps = ast.literal_eval(prep.steps_applied) if isinstance(prep.steps_applied, str) else prep.steps_applied
        preprocessing_data = {
            'mode': prep.mode,
            'steps_applied': steps,
            'original_shape': prep.original_shape,
            'processed_shape': prep.processed_shape,
            'missing_method': (steps or {}).get('missing_method', 'N/A'),
            'outlier_method': (steps or {}).get('outlier_method', 'N/A'),
            'encoding_method': (steps or {}).get('encoding_method', 'N/A'),
            'scaling_method': (steps or {}).get('scaling_method', 'N/A'),
        }

    forecast_models = {}
    forecasts = []
    target_col = forecast_results.get('target_col', 'Value') if forecast_results else 'Value'
    date_col = forecast_results.get('date_col', 'Date') if forecast_results else 'Date'
    if forecast_results:
        models = forecast_results.get('models', {})
        for mname, mdata in models.items():
            forecast_models[mname] = {
                'status': mdata.get('status', 'unknown'),
                'metrics': mdata.get('metrics', {}),
                'error': mdata.get('error', ''),
            }
        forecast_data_list = forecast_results.get('models', {})
        fc_best = forecast_results.get('best_model')
        lookup_model = best_model_name or fc_best
        if lookup_model and lookup_model in forecast_data_list:
            forecasts = [
                {'period': f'Period {i + 1}', 'value': round(v, 4)}
                for i, v in enumerate(forecast_data_list[lookup_model].get('future_predictions', []))
            ]

    workflow_time = _get_workflow_time(dataset_id)

    context = {
        'dataset_id': dataset_id,
        'dataset_name': dataset.file_name,
        'dataset_rows': dataset.rows_count,
        'dataset_columns': dataset.columns_count,
        'upload_date': dataset.upload_date.strftime('%B %d, %Y') if dataset.upload_date else 'N/A',
        'generated_at': datetime.utcnow().strftime('%B %d, %Y at %I:%M %p'),
        'best_model': best_model_name or 'N/A',
        'best_explanation': best_explanation,
        'best_rmse': best_rmse,
        'best_mae': best_mae,
        'best_mape': best_mape,
        'best_r2': best_r2,
        'steps_summary': steps_summary,
        'steps_completed': len([s for s in steps_summary if s['detail']]),
        'total_steps': len(STEPS),
        'validation': validation_data,
        'eda': eda_data,
        'eda_mode': eda.eda_mode if eda else None,
        'eda_charts': eda.total_charts if eda else None,
        'eda_report_path': eda.report_path if eda else None,
        'preprocessing': preprocessing_data,
        'forecast_model': fc.model_name if fc else (forecast_results.get('best_model') if forecast_results else None),
        'forecast_horizon': fc.forecast_horizon if fc else (forecast_results.get('horizon') if forecast_results else None),
        'target_column': target_col,
        'date_column': date_col,
        'forecast_models': forecast_models,
        'forecasts': forecasts,
        'ranking': ranking,
        'insights': insights,
        'chart_data': chart_data,
        'results_table': results_table,
        'workflow_time': workflow_time,
        'report_id': None,
    }

    _generate_smart_insights(context)

    report = Report(dataset_id=dataset_id)
    db.session.add(report)
    db.session.commit()
    context['report_id'] = report.id

    save_report_results(dataset_id, context)
    return context, None


def get_report_context(dataset_id):
    existing = load_report_results(dataset_id)
    if existing:
        return existing, None
    return generate_full_report(dataset_id)


def get_analysis_history(user_id, limit=10):
    from models.analysis_history_model import AnalysisHistory
    return AnalysisHistory.query.filter_by(user_id=user_id).order_by(
        AnalysisHistory.created_at.desc()
    ).limit(limit).all()


def save_analysis_history(dataset_id, user_id):
    from models.analysis_history_model import AnalysisHistory
    from models.dataset_model import Dataset
    from models.forecast_report_model import ForecastReport
    from models.comparison_report_model import ComparisonReport

    existing = AnalysisHistory.query.filter_by(
        dataset_id=dataset_id, user_id=user_id
    ).first()
    if existing:
        return existing

    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return None
    fc = ForecastReport.query.filter_by(dataset_id=dataset_id).order_by(
        ForecastReport.created_at.desc()).first()
    comp = ComparisonReport.query.filter_by(dataset_id=dataset_id).order_by(
        ComparisonReport.created_at.desc()).first()

    workflow_time = _get_workflow_time(dataset_id) or 0.0

    entry = AnalysisHistory(
        dataset_id=dataset_id,
        user_id=user_id,
        dataset_name=dataset.file_name,
        best_model=comp.best_model if comp else None,
        forecast_model=fc.model_name if fc else None,
        forecast_horizon=fc.forecast_horizon if fc else None,
        total_rows=dataset.rows_count,
        total_columns=dataset.columns_count,
        total_workflow_time=workflow_time,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


def generate_report_csv(dataset_id):
    context = load_report_results(dataset_id)
    if not context:
        return None, 'No report data found.'

    rows = []
    if context.get('results_table'):
        for m in context['results_table']:
            rows.append({
                'Rank': m.get('rank_label', '-'),
                'Model Name': m['name'],
                'Category': m.get('model_category', ''),
                'Status': m.get('status', ''),
                'MAE': m.get('mae', 'N/A'),
                'RMSE': m.get('rmse', 'N/A'),
                'MAPE': m.get('mape', 'N/A'),
                'R2 Score': m.get('r2', 'N/A'),
                'Training Time': m.get('training_time', 'N/A'),
            })
    elif context.get('forecasts'):
        rows = context['forecasts']

    df = pd.DataFrame(rows)
    path = os.path.join(_get_report_dir(dataset_id), 'full_report.csv')
    df.to_csv(path, index=False)
    return path, None


def generate_report_excel(dataset_id):
    context = load_report_results(dataset_id)
    if not context:
        return None, 'No report data found.'

    path = os.path.join(_get_report_dir(dataset_id), 'full_report.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        steps = pd.DataFrame(context.get('steps_summary', []))
        if not steps.empty:
            steps.to_excel(writer, sheet_name='Workflow Summary', index=False)

        ranking = pd.DataFrame(context.get('results_table', []))
        if not ranking.empty:
            ranking.to_excel(writer, sheet_name='Model Rankings', index=False)

        forecasts = pd.DataFrame(context.get('forecasts', []))
        if not forecasts.empty:
            forecasts.to_excel(writer, sheet_name='Forecasts', index=False)

        insights_df = pd.DataFrame(
            [{'Insight': s} for s in context.get('smart_insights', [])]
        )
        if not insights_df.empty:
            insights_df.to_excel(writer, sheet_name='Smart Insights', index=False)

    return path, None


def generate_report_pdf(dataset_id):
    context = load_report_results(dataset_id)
    if not context:
        return None, 'No report data found.'

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

        path = os.path.join(_get_report_dir(dataset_id), 'full_report.pdf')
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph('ForecastIQ Full Report', styles['Title']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            f'<b>Dataset:</b> {context.get("dataset_name", "")} &nbsp;&nbsp;'
            f'<b>Generated:</b> {context.get("generated_at", "")}',
            styles['Normal']
        ))
        elements.append(Spacer(1, 12))

        wt = context.get('workflow_time')
        if wt:
            elements.append(Paragraph(f'<b>Workflow Time:</b> {wt} minutes', styles['Normal']))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph(f'<b>Best Model:</b> {context.get("best_model", "N/A")}', styles['Normal']))
        elements.append(Paragraph(f'<b>Explanation:</b> {context.get("best_explanation", "")}', styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [['Rank', 'Model', 'Category', 'Status', 'MAE', 'RMSE', 'MAPE', 'R2', 'Time']]
        for m in context.get('results_table', []):
            table_data.append([
                str(m.get('rank_label', '-')),
                m['name'],
                m.get('model_category', ''),
                m.get('status', ''),
                str(m.get('mae', 'N/A')),
                str(m.get('rmse', 'N/A')),
                str(m.get('mape', 'N/A')),
                str(m.get('r2', 'N/A')),
                str(m.get('training_time', 'N/A')),
            ])

        if len(table_data) > 1:
            t = Table(table_data, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90D9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F8FC')]),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))

        steps_data = [['Step', 'Name', 'Detail']]
        for s in context.get('steps_summary', []):
            steps_data.append([str(s['number']), s['name'], s.get('detail', '')])
        t2 = Table(steps_data, repeatRows=1)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90D9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F8FC')]),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 12))

        if context.get('smart_insights'):
            elements.append(Paragraph('<b>Smart Insights</b>', styles['Heading2']))
            for ins in context['smart_insights']:
                elements.append(Paragraph(f'&bull; {ins}', styles['Normal']))
            elements.append(Spacer(1, 12))

        if context.get('forecasts'):
            elements.append(Paragraph('<b>Forecast Values</b>', styles['Heading2']))
            fc_data = [['Period', 'Value']]
            for f in context['forecasts']:
                fc_data.append([f.get('period', ''), str(f.get('value', ''))])
            t3 = Table(fc_data, repeatRows=1)
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90D9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(t3)

        doc.build(elements)
        return path, None
    except ImportError:
        return None, 'reportlab not installed. Install with: pip install reportlab'
