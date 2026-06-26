import os
import json
import numpy as np
import pandas as pd
from flask import current_app
from database import db
from models.comparison_report_model import ComparisonReport
from services.forecasting_service import (
    load_forecast_results, ALL_MODELS, _get_forecast_dir
)


def get_comparison_report(dataset_id):
    return ComparisonReport.query.filter_by(dataset_id=dataset_id).order_by(
        ComparisonReport.created_at.desc()
    ).first()


def _get_compare_dir(dataset_id):
    d = _get_forecast_dir(dataset_id)
    compare_dir = os.path.join(d, 'comparison')
    os.makedirs(compare_dir, exist_ok=True)
    return compare_dir


def _get_results_path(dataset_id):
    return os.path.join(_get_compare_dir(dataset_id), 'comparison_results.json')


def load_comparison_results(dataset_id):
    path = _get_results_path(dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_comparison_results(dataset_id, results):
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


def build_comparison(dataset_id):
    forecast_data = load_forecast_results(dataset_id)
    if not forecast_data:
        return None, 'No forecast results found. Run forecasting first.'

    models_data = forecast_data.get('models', {})
    if not models_data:
        return None, 'No models found in forecast results.'

    successful_models = []
    for mname, mdata in models_data.items():
        if mdata.get('status') == 'success' and mdata.get('metrics'):
            metrics = mdata['metrics']
            successful_models.append({
                'name': mname,
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'mape': metrics.get('mape'),
                'r2': metrics.get('r2'),
                'training_time': mdata.get('params', {}).get('training_time'),
                'status': 'success',
                'model_category': _get_model_category(mname),
                'test_predictions': mdata.get('test_predictions', []),
                'future_predictions': mdata.get('future_predictions', [])
            })
        else:
            successful_models.append({
                'name': mname,
                'mae': None,
                'rmse': None,
                'mape': None,
                'r2': None,
                'training_time': None,
                'status': 'failed',
                'error': mdata.get('error', 'Unknown error'),
                'model_category': _get_model_category(mname),
                'test_predictions': [],
                'future_predictions': []
            })

    ranked = _rank_models(successful_models)
    best_model_name = ranked[0]['name'] if ranked and ranked[0].get('status') == 'success' else None
    best_explanation = _generate_best_explanation(ranked)

    chart_data = _build_chart_data(ranked)
    insights = _generate_insights(ranked, forecast_data)
    results_table = _build_results_table(ranked)

    future_dates = forecast_data.get('future_dates', [])
    horizon = forecast_data.get('horizon', len(future_dates))
    target_col = forecast_data.get('target_col', 'Target')
    date_col = forecast_data.get('date_col', 'Date')

    report = ComparisonReport(
        dataset_id=dataset_id,
        best_model=best_model_name or 'N/A',
        ranking=json.dumps(ranked),
        comparison_metrics=json.dumps(chart_data),
    )
    db.session.add(report)
    db.session.commit()

    results = {
        'dataset_id': dataset_id,
        'best_model': best_model_name,
        'best_explanation': best_explanation,
        'ranking': ranked,
        'chart_data': chart_data,
        'insights': insights,
        'results_table': results_table,
        'future_dates': future_dates,
        'horizon': horizon,
        'target_col': target_col,
        'date_col': date_col,
        'report_id': report.id,
    }

    save_comparison_results(dataset_id, results)
    return results, None


def _get_model_category(mname):
    from services.forecasting_service import TRADITIONAL_MODELS, ML_MODELS, DL_MODELS
    if mname in TRADITIONAL_MODELS:
        return 'Traditional'
    if mname in ML_MODELS:
        return 'Machine Learning'
    if mname in DL_MODELS:
        return 'Deep Learning'
    return 'Unknown'


def _rank_models(models):
    sorted_models = sorted(
        [m for m in models if m.get('status') == 'success' and m.get('rmse') is not None],
        key=lambda m: (m['rmse'], m['mae'] if m['mae'] is not None else float('inf'),
                       -(m['r2'] if m['r2'] is not None else float('-inf')))
    )
    failed = [m for m in models if m.get('status') != 'success']
    result = []
    for i, m in enumerate(sorted_models):
        m['rank'] = i + 1
        m['rank_label'] = f'#{i + 1}'
        m['rank_reason'] = _rank_reason(m, i, sorted_models)
        result.append(m)
    for m in failed:
        m['rank'] = None
        m['rank_label'] = '-'
        m['rank_reason'] = m.get('error', 'Model failed to train')
        result.append(m)
    return result


def _rank_reason(model, index, all_models):
    reasons = []
    if index == 0:
        reasons.append('Lowest RMSE among all models')
        if all_models and model.get('mae') is not None:
            if all(m.get('mae') is not None and model['mae'] <= m['mae'] for m in all_models):
                reasons.append('Best MAE score')
        if all_models and model.get('r2') is not None:
            if all(m.get('r2') is not None and model['r2'] >= m['r2'] for m in all_models):
                reasons.append('Highest R\u00b2 score')
    else:
        better = all_models[0]
        if model.get('rmse') is not None and better.get('rmse') is not None and better['rmse'] != 0:
            diff_pct = ((model['rmse'] - better['rmse']) / better['rmse']) * 100
            reasons.append(f'RMSE {diff_pct:.1f}% higher than {better["name"]}')
        if model.get('r2') is not None and better.get('r2') is not None:
            r2_diff = better['r2'] - model['r2']
            reasons.append(f'R\u00b2 {r2_diff:.4f} lower than {better["name"]}')
    return '; '.join(reasons) if reasons else 'Ranked by RMSE'


def _generate_best_explanation(ranked):
    successful = [m for m in ranked if m.get('status') == 'success']
    if not successful:
        return 'No successful models to evaluate.'
    best = successful[0]
    parts = [f'{best["name"]} is the best performing model.']
    if best.get('rmse') is not None:
        parts.append(f'Lowest RMSE: {best["rmse"]:.4f}')
    if best.get('mae') is not None:
        parts.append(f'Lowest MAE: {best["mae"]:.4f}')
    if best.get('r2') is not None:
        parts.append(f'Highest R\u00b2: {best["r2"]:.4f}')
    if len(successful) > 1:
        runner_up = successful[1]
        if runner_up.get('rmse') is not None and best.get('rmse') is not None:
            improvement = ((runner_up['rmse'] - best['rmse']) / runner_up['rmse']) * 100
            parts.append(f'{improvement:.1f}% better RMSE than runner-up ({runner_up["name"]}).')
    return '. '.join(parts)


def _build_chart_data(ranked):
    successful = [m for m in ranked if m.get('status') == 'success']
    return {
        'labels': [m['name'] for m in successful],
        'rmse': [m['rmse'] for m in successful],
        'mae': [m['mae'] for m in successful],
        'mape': [m['mape'] for m in successful],
        'r2': [m['r2'] for m in successful],
        'training_time': [
            m.get('training_time') if m.get('training_time') is not None else 0
            for m in successful
        ],
    }


def _generate_insights(ranked, forecast_data):
    successful = [m for m in ranked if m.get('status') == 'success']
    insights = []
    if not successful:
        insights.append({'type': 'warning', 'text': 'No models completed training successfully.'})
        return insights

    best = successful[0]
    insights.append({
        'type': 'best',
        'text': f'{best["name"]} ranks #1 with RMSE {best["rmse"]:.4f}, '
                f'MAE {best["mae"]:.4f}, R\u00b2 {best["r2"]:.4f}.'
    })

    category_counts = {}
    for m in successful:
        cat = m.get('model_category', 'Unknown')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    best_cat = max(category_counts, key=category_counts.get) if category_counts else ''
    if best_cat:
        count = category_counts[best_cat]
        insights.append({
            'type': 'category',
            'text': f'{best_cat} models dominate with {count} of {len(successful)} successful models.'
        })

    if len(successful) > 1:
        rmse_values = [m['rmse'] for m in successful if m['rmse'] is not None]
        if len(rmse_values) > 1:
            spread = max(rmse_values) - min(rmse_values)
            mean_rmse = np.mean(rmse_values)
            cv = spread / mean_rmse if mean_rmse > 0 else 0
            if cv < 0.1:
                insights.append({
                    'type': 'consistency',
                    'text': f'All models show consistent RMSE (spread: {spread:.4f}), '
                            f'suggesting the dataset is well-behaved.'
                })
            elif cv > 0.5:
                insights.append({
                    'type': 'variance',
                    'text': f'Large RMSE spread ({spread:.4f}) across models — '
                            f'choice of algorithm significantly impacts performance.'
                })

    fc_insights = forecast_data.get('insights', {})
    trend = fc_insights.get('trend_direction', '')
    if trend:
        insights.append({
            'type': 'trend',
            'text': f'Forecast trend is {trend} with '
                    f'{fc_insights.get("total_growth_pct", 0):.1f}% total growth.'
        })

    if len(successful) >= 2:
        fastest = min(successful, key=lambda m: m.get('training_time') or float('inf'))
        if fastest.get('training_time') is not None:
            insights.append({
                'type': 'speed',
                'text': f'{fastest["name"]} is the fastest model '
                        f'({fastest["training_time"]:.2f}s training time).'
            })

    if any(m.get('r2') is not None and m['r2'] < 0 for m in successful):
        insights.append({
            'type': 'warning',
            'text': f'Some models show negative R\u00b2 — they perform worse than a '
                    f'constant mean baseline. Consider feature engineering or data transformation.'
        })

    r2_values = [m['r2'] for m in successful if m['r2'] is not None]
    if r2_values and np.mean(r2_values) > 0.9:
        insights.append({
            'type': 'success',
            'text': f'Average R\u00b2 of {np.mean(r2_values):.3f} indicates excellent '
                    f'model fit across most algorithms.'
        })

    return insights


def _build_results_table(ranked):
    table = []
    for m in ranked:
        table.append({
            'name': m['name'],
            'mae': _fmt_metric(m.get('mae')),
            'rmse': _fmt_metric(m.get('rmse')),
            'mape': _fmt_pct(m.get('mape')),
            'r2': _fmt_metric(m.get('r2')),
            'training_time': _fmt_time(m.get('training_time')),
            'horizon': m.get('horizon'),
            'status': m.get('status', 'unknown'),
            'rank': m.get('rank'),
            'rank_label': m.get('rank_label', '-'),
            'rank_reason': m.get('rank_reason', ''),
            'model_category': m.get('model_category', ''),
            'error': m.get('error', ''),
        })
    return table


def _fmt_metric(val):
    if val is None:
        return 'N/A'
    return round(val, 4)


def _fmt_pct(val):
    if val is None:
        return 'N/A'
    return f'{round(val, 2)}%'


def _fmt_time(val):
    if val is None:
        return 'N/A'
    return f'{val:.2f}s'


def get_comparison_context(dataset_id):
    existing = load_comparison_results(dataset_id)
    if existing:
        return existing, None
    return build_comparison(dataset_id)


def get_model_actual_predicted(dataset_id, model_name):
    forecast_data = load_forecast_results(dataset_id)
    if not forecast_data:
        return None, 'No forecast results found.'
    models_data = forecast_data.get('models', {})
    mdata = models_data.get(model_name)
    if not mdata:
        return None, f'Model "{model_name}" not found.'
    if mdata.get('status') != 'success':
        return None, f'Model "{model_name}" did not complete successfully.'
    target_col = forecast_data.get('target_col', 'Target')
    date_col = forecast_data.get('date_col', 'Date')
    future_dates = forecast_data.get('future_dates', [])
    test_pred = mdata.get('test_predictions', [])
    future_pred = mdata.get('future_predictions', [])
    return {
        'model_name': model_name,
        'target_col': target_col,
        'date_col': date_col,
        'test_predictions': test_pred,
        'future_predictions': future_pred,
        'future_dates': future_dates,
    }, None


def generate_comparison_csv(dataset_id):
    comparison = load_comparison_results(dataset_id)
    if not comparison:
        return None, 'No comparison results found.'

    rows = []
    for m in comparison.get('results_table', []):
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

    df = pd.DataFrame(rows)
    path = os.path.join(_get_compare_dir(dataset_id), 'comparison_results.csv')
    df.to_csv(path, index=False)
    return path, None


def generate_comparison_excel(dataset_id):
    comparison = load_comparison_results(dataset_id)
    if not comparison:
        return None, 'No comparison results found.'

    rows = []
    for m in comparison.get('results_table', []):
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

    df = pd.DataFrame(rows)
    path = os.path.join(_get_compare_dir(dataset_id), 'comparison_results.xlsx')
    df.to_excel(path, index=False, engine='openpyxl')
    return path, None


def generate_comparison_pdf(dataset_id):
    comparison = load_comparison_results(dataset_id)
    if not comparison:
        return None, 'No comparison results found.'

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

        path = os.path.join(_get_compare_dir(dataset_id), 'comparison_results.pdf')
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph('Model Comparison Report', styles['Title']))
        elements.append(Spacer(1, 12))

        best = comparison.get('best_model', 'N/A')
        explanation = comparison.get('best_explanation', '')
        elements.append(Paragraph(f'<b>Best Model:</b> {best}', styles['Normal']))
        elements.append(Paragraph(f'<b>Explanation:</b> {explanation}', styles['Normal']))
        elements.append(Spacer(1, 12))

        table_data = [['Rank', 'Model', 'Category', 'Status', 'MAE', 'RMSE', 'MAPE', 'R2', 'Time']]
        for m in comparison.get('results_table', []):
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

        doc.build(elements)
        return path, None
    except ImportError:
        return None, 'reportlab not installed. Install with: pip install reportlab'
