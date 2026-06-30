import os
import json
import traceback
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from datetime import datetime
from flask import current_app
from database import db
from models.eda_report_model import EDAReport
from models.dataset_model import Dataset
from services.dataset_service import read_dataframe
from services.chart_service import (histogram, density_plot, boxplot,
                                    correlation_heatmap, missing_values_chart,
                                    missing_heatmap, line_chart, bar_chart,
                                    pie_chart, trend_chart, rolling_average_chart)


def _log_error(chart_name, column, error):
    traceback.print_exc()
    print(f"[EDA ERROR] {chart_name} | Column: {column} | {error}")


def _get_output_dir(dataset_id):
    base = current_app.config['EDA_REPORTS_FOLDER']
    out = os.path.join(base, str(dataset_id))
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'charts'), exist_ok=True)
    return out


def _load_df(dataset):
    ext = dataset.file_name.rsplit('.', 1)[1].lower()
    df = read_dataframe(dataset.file_path, ext)
    return df


def compute_statistics(df):
    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        try:
            mode_vals = s.mode()
            stats[col] = {
                'count': int(len(s)),
                'mean': round(float(s.mean()), 4),
                'median': round(float(s.median()), 4),
                'mode': float(mode_vals.iloc[0]) if len(mode_vals) > 0 else None,
                'std': round(float(s.std()), 4),
                'variance': round(float(s.var()), 4),
                'min': round(float(s.min()), 4),
                'max': round(float(s.max()), 4),
                'q1': round(float(s.quantile(0.25)), 4),
                'q2': round(float(s.quantile(0.50)), 4),
                'q3': round(float(s.quantile(0.75)), 4),
                'iqr': round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                'skewness': round(float(s.skew()), 4),
                'kurtosis': round(float(s.kurtosis()), 4)
            }
        except Exception as e:
            _log_error('compute_statistics', col, e)
    return stats


def compute_missing_analysis(df):
    result = []
    for col in df.columns:
        count = int(df[col].isnull().sum())
        pct = round(count / len(df) * 100, 2) if len(df) > 0 else 0
        if count > 0:
            result.append({'column': col, 'count': count, 'percentage': pct})
    total_missing = sum(m['count'] for m in result)
    return {
        'columns': result,
        'total_missing': total_missing,
        'total_cells': len(df) * len(df.columns),
        'overall_pct': round(total_missing / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0
    }


def compute_correlation(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr_matrix = numeric.corr().round(4)
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                'col1': cols[i], 'col2': cols[j],
                'value': round(float(corr_matrix.iloc[i, j]), 4)
            })
    pairs.sort(key=lambda x: abs(x['value']), reverse=True)
    high_corr = [p for p in pairs if abs(p['value']) >= 0.7]
    return {
        'matrix': corr_matrix.to_dict(),
        'pairs': pairs,
        'high_corr': high_corr,
        'columns': cols
    }


def detect_outliers_iqr(df):
    numeric = df.select_dtypes(include=[np.number])
    results = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        results[col] = {
            'lower_bound': round(float(lower), 4),
            'upper_bound': round(float(upper), 4),
            'outlier_count': int(len(outliers)),
            'outlier_pct': round(len(outliers) / len(s) * 100, 2),
            'outlier_min': round(float(outliers.min()), 4) if len(outliers) > 0 else None,
            'outlier_max': round(float(outliers.max()), 4) if len(outliers) > 0 else None
        }
    return results


def detect_outliers_zscore(df, threshold=3):
    numeric = df.select_dtypes(include=[np.number])
    results = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        z = np.abs(scipy_stats.zscore(s, nan_policy='omit'))
        outliers = s[z > threshold]
        results[col] = {
            'threshold': threshold,
            'outlier_count': int(len(outliers)),
            'outlier_pct': round(len(outliers) / len(s) * 100, 2),
            'outlier_min': round(float(outliers.min()), 4) if len(outliers) > 0 else None,
            'outlier_max': round(float(outliers.max()), 4) if len(outliers) > 0 else None
        }
    return results


def analyze_categorical(df):
    cats = df.select_dtypes(include=['object', 'category'])
    results = {}
    for col in cats.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        freq = s.value_counts()
        top_n = freq.head(10)
        results[col] = {
            'unique_count': int(s.nunique()),
            'top_values': {str(k): int(v) for k, v in top_n.items()},
            'freq_dict': {str(k): int(v) for k, v in freq.items()},
            'top_value': str(freq.index[0]) if len(freq) > 0 else None,
            'top_freq': int(freq.iloc[0]) if len(freq) > 0 else 0
        }
    return results


def analyze_timeseries(df, date_col=None):
    """
    Analyze time series information from the dataset.

    - Automatically detects datetime column if not provided.
    - Handles mixed timezone formats safely.
    - Works with weather, stock, sales and generic datasets.
    """

    # Auto detect datetime column
    if date_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break

        if date_col is None:
            for col in df.columns:
                try:
                    sample = df[col].dropna().head(100)

                    parsed = pd.to_datetime(
                        sample,
                        errors="coerce",
                        utc=True
                    )

                    # Accept column only if most values are valid dates
                    if parsed.notna().mean() >= 0.8:
                        date_col = col
                        break

                except Exception:
                    continue

    if date_col is None:
        return None

    # Convert datetime safely
    try:
        ts = pd.to_datetime(
            df[date_col],
            errors="coerce",
            utc=True
        )

        # Remove timezone information
        ts = ts.dt.tz_localize(None)

    except Exception as e:
        _log_error("analyze_timeseries", date_col, e)
        return None

    # Find first numeric column
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric:
        return {
            "date_column": date_col,
            "value_column": None,
            "msg": "No numeric column available for trend analysis"
        }

    value_col = numeric[0]

    # Build cleaned dataframe
    ts_df = pd.DataFrame({
        "date": ts,
        "value": df[value_col]
    })

    ts_df = ts_df.dropna()

    if ts_df.empty:
        return {
            "date_column": date_col,
            "value_column": value_col,
            "msg": "No valid datetime values found"
        }

    ts_df = ts_df.sort_values("date")

    if len(ts_df) < 4:
        return {
            "date_column": date_col,
            "value_column": value_col,
            "msg": "Not enough data points for time series analysis"
        }

    try:
        frequency = pd.infer_freq(ts_df["date"])
    except Exception:
        frequency = None

    return {
        "date_column": date_col,
        "value_column": value_col,
        "date_min": str(ts_df["date"].min()),
        "date_max": str(ts_df["date"].max()),
        "total_points": len(ts_df),
        "frequency": frequency if frequency else "Irregular"
    }

def get_feature_insights(df, stats, correlation, outliers_iqr):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    insights = {
        'potential_targets': [],
        'high_variance_features': [],
        'highly_correlated_pairs': [],
        'many_outliers_features': [],
        'low_variance_features': []
    }
    if stats:
        for col, s in stats.items():
            if s['variance'] > 0:
                insights['high_variance_features'].append({'column': col, 'variance': s['variance']})
        insights['high_variance_features'].sort(key=lambda x: x['variance'], reverse=True)
        insights['high_variance_features'] = insights['high_variance_features'][:5]
    if correlation and correlation.get('high_corr'):
        insights['highly_correlated_pairs'] = correlation['high_corr'][:10]
    if outliers_iqr:
        for col, o in outliers_iqr.items():
            if o['outlier_pct'] > 5:
                insights['many_outliers_features'].append({'column': col, 'pct': o['outlier_pct']})
        insights['many_outliers_features'].sort(key=lambda x: x['pct'], reverse=True)
    if numeric:
        potential = [c for c in numeric if c in stats and stats[c]['variance'] > 0]
        insights['potential_targets'] = potential[:5]
    return insights


def compute_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    boolean = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    return {
        'numeric': numeric,
        'categorical': categorical,
        'date': date,
        'boolean': boolean
    }


def compute_outlier_summary(outliers_iqr):
    if not outliers_iqr:
        return {'total_outliers': 0, 'affected_columns': [], 'overall_pct': 0}
    total = sum(o['outlier_count'] for o in outliers_iqr.values())
    affected = [{'column': col, 'count': o['outlier_count'], 'pct': o['outlier_pct']}
                for col, o in outliers_iqr.items() if o['outlier_count'] > 0]
    return {
        'total_outliers': total,
        'affected_columns': affected,
        'overall_pct': round(total / max(sum(o['outlier_count'] + 1 for o in outliers_iqr.values()) - total, 1), 2)
    }

def _prepare_dataframe(df):
    """
    Prepare dataframe safely for EDA.

    - Detect datetime columns.
    - Handle mixed formats.
    - Handle timezone-aware timestamps.
    - Preserve categorical columns.
    """

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        if df[col].dtype != object:
            continue

        sample = df[col].dropna().head(100)

        if len(sample) == 0:
            continue

        try:

            parsed = pd.to_datetime(
                sample,
                format="mixed",
                errors="coerce",
                utc=True
            )

            if parsed.notna().mean() >= 0.8:

                df[col] = pd.to_datetime(
                    df[col],
                    format="mixed",
                    errors="coerce",
                    utc=True
                )

                df[col] = df[col].dt.tz_localize(None)

        except Exception:
            pass

    return df

def _detect_datetime_columns(df):
    """
    Detect datetime columns including object/string date columns.
    Supports mixed date formats and timezone-aware timestamps.
    """

    date_cols = []

    for col in df.columns:

        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        # Take a sample of non-null values
        sample = df[col].dropna().head(100)

        if len(sample) == 0:
            continue

        try:
            converted = pd.to_datetime(
                sample,
                format="mixed",
                errors="coerce",
                utc=True
            )

            # Accept column if at least 80% are valid dates
            if converted.notna().mean() >= 0.8:
                date_cols.append(col)

        except Exception:
            continue

    return date_cols

def generate_charts(df, output_dir, selected_charts=None):
    charts = {}
    df =_prepare_dataframe(df)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = _detect_datetime_columns(df)
    all_charts = selected_charts or ['histogram', 'boxplot', 'density', 'correlation', 'missing',
                                     'missing_heatmap', 'bar', 'pie', 'trend']
    if 'histogram' in all_charts:
        charts['histograms'] = []
        for col in numeric[:6]:
            try:
                chart_data, img = histogram(df, col, output_dir)
                if chart_data:
                    charts['histograms'].append({'column': col, 'plotly': chart_data, 'image': img})
            except Exception as e:
                _log_error('histogram', col, e)
    if 'boxplot' in all_charts:
        charts['boxplots'] = []
        for col in numeric[:6]:
            try:
                chart_data, img = boxplot(df, col, output_dir)
                if chart_data:
                    charts['boxplots'].append({'column': col, 'plotly': chart_data, 'image': img})
            except Exception as e:
                _log_error('boxplot', col, e)
    if 'density' in all_charts:
        charts['density'] = []
        for col in numeric[:6]:
            try:
                chart_data, img = density_plot(df, col, output_dir)
                if chart_data:
                    charts['density'].append({'column': col, 'plotly': chart_data, 'image': img})
            except Exception as e:
                _log_error('density_plot', col, e)
    if 'correlation' in all_charts:
        try:
            chart_data, img = correlation_heatmap(df, output_dir)
            charts['correlation'] = {'plotly': chart_data, 'image': img} if chart_data else None
        except Exception as e:
            _log_error('correlation_heatmap', 'all', e)
            charts['correlation'] = None
    if 'missing' in all_charts:
        missing = compute_missing_analysis(df)['columns']
        if missing:
            try:
                chart_data, img = missing_values_chart(missing, output_dir)
                charts['missing'] = {'plotly': chart_data, 'image': img} if chart_data else None
            except Exception as e:
                _log_error('missing_values_chart', 'all', e)
                charts['missing'] = None
    if 'missing_heatmap' in all_charts:
        try:
            chart_data, img = missing_heatmap(df, output_dir)
            charts['missing_heatmap'] = {'plotly': chart_data, 'image': img} if chart_data else None
        except Exception as e:
            _log_error('missing_heatmap', 'all', e)
            charts['missing_heatmap'] = None
    if 'bar' in all_charts:
        charts['bar'] = []
        for col in categorical[:3]:
            try:
                freq = df[col].dropna().value_counts().head(10).to_dict()
                freq = {str(k): int(v) for k, v in freq.items()}
                chart_data, img = bar_chart(freq, f'Top Values — {col}', output_dir, f'bar_{col}')
                if chart_data:
                    charts['bar'].append({'column': col, 'plotly': chart_data, 'image': img})
            except Exception as e:
                _log_error('bar_chart', col, e)
    if 'pie' in all_charts:
        charts['pie'] = []
        for col in categorical[:3]:
            try:
                freq = df[col].dropna().value_counts().head(8).to_dict()
                freq = {str(k): int(v) for k, v in freq.items()}
                chart_data, img = pie_chart(freq, f'Distribution — {col}', output_dir, f'pie_{col}')
                if chart_data:
                    charts['pie'].append({'column': col, 'plotly': chart_data, 'image': img})
            except Exception as e:
                _log_error('pie_chart', col, e)
    if 'trend' in all_charts:
        if not date_cols:
            _log_error('trend_chart', '', 'No datetime column detected in dataset')
        elif not numeric:
            _log_error('trend_chart', '', 'No numeric columns available for trend analysis')
        else:
            date_col = date_cols[0]
            value_col = numeric[0]
            try:
                chart_data, img = trend_chart(df, date_col, value_col, output_dir)
                charts['trend'] = {'plotly': chart_data, 'image': img} if chart_data else None
            except Exception as e:
                _log_error('trend_chart', value_col, e)
                charts['trend'] = None
            try:
                ra_data, ra_img = rolling_average_chart(df, date_col, value_col, 7, output_dir)
                charts['rolling_avg'] = {'plotly': ra_data, 'image': ra_img} if ra_data else None
            except Exception as e:
                _log_error('rolling_average_chart', value_col, e)
                charts['rolling_avg'] = None
    total_charts = sum(len(v) for v in charts.values() if isinstance(v, list)) + \
                   sum(1 for v in charts.values() if isinstance(v, dict) and v is not None)
    return charts, total_charts


def run_automatic_eda(dataset_id, user_id):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, 'Dataset not found'
    df = _load_df(dataset)
    if df is None:
        return None, 'Unable to read dataset'
    output_dir = _get_output_dir(dataset_id)

    stats = compute_statistics(df)
    missing = compute_missing_analysis(df)
    correlation = compute_correlation(df)
    outliers_iqr = detect_outliers_iqr(df)
    outliers_zscore = detect_outliers_zscore(df)
    categorical = analyze_categorical(df)
    timeseries = analyze_timeseries(df)
    feature_insights = get_feature_insights(df, stats, correlation, outliers_iqr)
    column_types = compute_column_types(df)

    charts, total_charts = generate_charts(df, output_dir)

    report = EDAReport(
        dataset_id=dataset_id, eda_mode='automatic',
        report_path=output_dir, total_charts=total_charts,
        generated_at=datetime.utcnow()
    )
    db.session.add(report)
    db.session.commit()

    results = {
        'report_id': report.id,
        'dataset_overview': {
            'rows': len(df), 'columns': len(df.columns),
            'memory_kb': round(df.memory_usage(deep=True).sum() / 1024, 2),
            'numeric_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_count': len(df.select_dtypes(include=['object', 'category']).columns),
           'date_count': len(_detect_datetime_columns(df)),
            'column_types': column_types
        },
        'statistics': stats,
        'missing_analysis': missing,
        'correlation': correlation,
        'outliers_iqr': outliers_iqr,
        'outliers_zscore': outliers_zscore,
        'categorical_analysis': categorical,
        'timeseries': timeseries,
        'feature_insights': feature_insights,
        'charts': charts,
        'total_charts': total_charts,
        'output_dir': output_dir
    }
    return results, None


def run_manual_eda(dataset_id, user_id, selections):
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not dataset:
        return None, 'Dataset not found'
    df = _load_df(dataset)
    if df is None:
        return None, 'Unable to read dataset'
    output_dir = _get_output_dir(dataset_id)

    selected_stats = selections.get('statistics', [])
    selected_analysis = selections.get('analysis', [])
    selected_charts = selections.get('charts', [])

    results = {
        'dataset_overview': {
            'rows': len(df), 'columns': len(df.columns),
            'memory_kb': round(df.memory_usage(deep=True).sum() / 1024, 2),
            'numeric_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_count': len(df.select_dtypes(include=['object', 'category']).columns),
           'date_count': len(_detect_datetime_columns(df)),
            'column_types': compute_column_types(df)
        }
    }
    charts = {}
    total_charts = 0

    if selected_stats:
        full_stats = compute_statistics(df)
        filtered = {}
        for col, s in full_stats.items():
            filtered[col] = {k: v for k, v in s.items() if k in selected_stats}
        results['statistics'] = filtered

    if 'missing' in selected_analysis:
        results['missing_analysis'] = compute_missing_analysis(df)

    if 'correlation' in selected_analysis:
        results['correlation'] = compute_correlation(df)

    if 'outlier' in selected_analysis:
        results['outliers_iqr'] = detect_outliers_iqr(df)
        results['outliers_zscore'] = detect_outliers_zscore(df)

    if 'trend' in selected_analysis:
        results['timeseries'] = analyze_timeseries(df)

    if 'seasonality' in selected_analysis:
        if results.get('timeseries') and results['timeseries'].get('value_column'):
            results['seasonality'] = {'note': 'Seasonality decomposition available in preprocessing phase'}
        else:
            results['seasonality'] = {'note': 'Requires date and numeric columns'}

    chart_map = {
        'histogram': 'histogram', 'boxplot': 'boxplot',
        'correlation_heatmap': 'correlation', 'missing_chart': 'missing',
        'line_chart': 'line', 'bar_chart': 'bar', 'pie_chart': 'pie'
    }
    mapped_charts = []
    for c in selected_charts:
        if c in chart_map and chart_map[c] not in mapped_charts:
            mapped_charts.append(chart_map[c])

    if mapped_charts:
        charts_result, total_charts = generate_charts(df, output_dir, selected_charts=mapped_charts)
        charts.update(charts_result)

    results['charts'] = charts
    results['total_charts'] = total_charts
    results['output_dir'] = output_dir

    report = EDAReport(
        dataset_id=dataset_id, eda_mode='manual',
        report_path=output_dir, total_charts=total_charts,
        generated_at=datetime.utcnow()
    )
    db.session.add(report)
    db.session.commit()
    results['report_id'] = report.id
    return results, None


def generate_html_report(dataset, results, output_dir):
    stats = results.get('statistics', {})
    missing = results.get('missing_analysis', {})
    correlation = results.get('correlation', {})
    outliers = results.get('outliers_iqr', {})
    categorical = results.get('categorical_analysis', {})
    timeseries = results.get('timeseries', {})
    insights = results.get('feature_insights', {})
    overview = results.get('dataset_overview', {})

    html_parts = [f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>EDA Report — {dataset.file_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 10px; }}
  h2 {{ color: #0d6efd; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #0d6efd; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fc; }}
  .section {{ margin: 20px 0; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; color: white; }}
  .badge-green {{ background: #198754; }}
  .badge-red {{ background: #dc3545; }}
  .badge-orange {{ background: #fd7e14; }}
  img {{ max-width: 100%; height: auto; margin: 10px 0; }}
  .footer {{ margin-top: 40px; text-align: center; color: #888; font-size: 12px; }}
</style></head><body>
<h1>EDA Report — {dataset.file_name}</h1>
<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
<p>Rows: {overview.get('rows', 'N/A')} | Columns: {overview.get('columns', 'N/A')} | Size: {overview.get('memory_kb', 0)} KB</p>
''']

    if stats:
        html_parts.append('<div class="section"><h2>Statistical Summary</h2><table><tr><th>Column</th>')
        metrics = ['count', 'mean', 'median', 'std', 'min', 'max', 'q1', 'q3', 'iqr']
        for m in metrics:
            html_parts.append(f'<th>{m.title()}</th>')
        html_parts.append('</tr>')
        for col, s in list(stats.items())[:10]:
            html_parts.append(f'<tr><td><strong>{col}</strong></td>')
            for m in metrics:
                v = s.get(m, '')
                html_parts.append(f'<td>{v}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table></div>')

    if missing and missing.get('columns'):
        html_parts.append(f'<div class="section"><h2>Missing Values</h2>')
        html_parts.append(f'<p>Total missing: {missing["total_missing"]} / {missing["total_cells"]} ({missing["overall_pct"]}%)</p>')
        html_parts.append('<table><tr><th>Column</th><th>Missing</th><th>Percentage</th></tr>')
        for m in missing['columns']:
            badge = 'badge-red' if m['percentage'] > 20 else 'badge-orange' if m['percentage'] > 5 else ''
            html_parts.append(f'<tr><td>{m["column"]}</td><td>{m["count"]}</td><td><span class="badge {badge}">{m["percentage"]}%</span></td></tr>')
        html_parts.append('</table></div>')

    if outliers:
        html_parts.append('<div class="section"><h2>Outlier Detection (IQR)</h2><table><tr><th>Column</th><th>Outliers</th><th>%</th><th>Lower Bound</th><th>Upper Bound</th></tr>')
        for col, o in outliers.items():
            badge = 'badge-red' if o['outlier_pct'] > 5 else ''
            html_parts.append(f'<tr><td>{col}</td><td>{o["outlier_count"]}</td><td><span class="badge {badge}">{o["outlier_pct"]}%</span></td><td>{o["lower_bound"]}</td><td>{o["upper_bound"]}</td></tr>')
        html_parts.append('</table></div>')

    if correlation and correlation.get('high_corr'):
        html_parts.append('<div class="section"><h2>Highly Correlated Features</h2><table><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr>')
        for p in correlation['high_corr'][:10]:
            html_parts.append(f'<tr><td>{p["col1"]}</td><td>{p["col2"]}</td><td><strong>{p["value"]}</strong></td></tr>')
        html_parts.append('</table></div>')

    corr_img = os.path.join(output_dir, 'images', 'correlation_heatmap.png')
    if os.path.exists(corr_img):
        import base64
        with open(corr_img, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        html_parts.append(f'<div class="section"><h2>Correlation Heatmap</h2><img src="data:image/png;base64,{b64}"></div>')

    for img_type in ['missing_values']:
        img_path = os.path.join(output_dir, 'images', f'{img_type}.png')
        if os.path.exists(img_path):
            import base64
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            html_parts.append(f'<div class="section"><h2>{img_type.replace("_", " ").title()}</h2><img src="data:image/png;base64,{b64}"></div>')

    if categorical:
        html_parts.append('<div class="section"><h2>Categorical Analysis</h2>')
        for col, cat in list(categorical.items())[:5]:
            html_parts.append(f'<p><strong>{col}:</strong> {cat["unique_count"]} unique values, top: {cat["top_value"]} ({cat["top_freq"]})</p>')
        html_parts.append('</div>')

    if insights:
        html_parts.append('<div class="section"><h2>Feature Insights</h2>')
        if insights.get('potential_targets'):
            html_parts.append(f'<p><strong>Potential Target Columns:</strong> {", ".join(insights["potential_targets"])}</p>')
        if insights.get('highly_correlated_pairs'):
            html_parts.append(f'<p><strong>Highly Correlated Pairs:</strong> {len(insights["highly_correlated_pairs"])} found</p>')
        if insights.get('many_outliers_features'):
            html_parts.append(f'<p><strong>Features with Many Outliers:</strong> {", ".join(f["column"] for f in insights["many_outliers_features"][:5])}</p>')
        html_parts.append('</div>')

    html_parts.append(f'<div class="footer">Generated by ForecastIQ EDA Engine &mdash; {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>')
    html_parts.append('</body></html>')

    report_path = os.path.join(output_dir, 'eda_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    return report_path


def _get_results_path(dataset_id):
    base = current_app.config['EDA_REPORTS_FOLDER']
    return os.path.join(base, str(dataset_id), 'results.json')


def save_eda_results(dataset_id, results):
    path = _get_results_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_data = {}
    for k, v in results.items():
        if k == 'output_dir':
            continue
        try:
            json.dumps(v)
            save_data[k] = v
        except (TypeError, ValueError):
            save_data[k] = str(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)


def load_eda_results(dataset_id):
    path = _get_results_path(dataset_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_eda_report(report_id):
    return EDAReport.query.get(report_id)


def compute_data_quality_score(results):
    overview = results.get('dataset_overview', {})
    missing = results.get('missing_analysis', {})
    outliers = results.get('outliers_iqr', {})

    rows = overview.get('rows', 0)
    cols = overview.get('columns', 0)

    total_cells = rows * cols if rows and cols else 0
    total_missing = missing.get('total_missing', 0) if missing else 0
    completeness = 100.0
    if total_cells > 0:
        completeness = round(100 - (total_missing / total_cells * 100), 1)

    outlier_cols = len(outliers) if outliers else 0
    outlier_score = 100.0
    if cols > 0:
        outlier_score = round(max(0, 100 - (outlier_cols / cols * 100)), 1)

    quality = round(completeness * 0.6 + outlier_score * 0.4, 1)
    grade = 'A' if quality >= 90 else 'B' if quality >= 75 else 'C' if quality >= 60 else 'D'

    return {
        'score': quality,
        'completeness': completeness,
        'outlier_score': outlier_score,
        'grade': grade,
        'total_cells': total_cells,
        'total_missing': total_missing,
        'missing_pct': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0
    }


def generate_smart_insights(results):
    overview = results.get('dataset_overview', {})
    missing = results.get('missing_analysis', {})
    correlation = results.get('correlation', {})
    outliers = results.get('outliers_iqr', {})
    stats = results.get('statistics', {})
    cat_analysis = results.get('categorical_analysis', {})
    ts = results.get('timeseries', {})

    insights = []
    rows = overview.get('rows', 0)
    cols = overview.get('columns', 0)
    numeric_count = overview.get('numeric_count', 0)
    cat_count = overview.get('categorical_count', 0)
    date_count = overview.get('date_count', 0)

    if rows > 0:
        insights.append({
            'icon': 'fa-database', 'color': 'primary',
            'title': 'Dataset Overview',
            'text': f'{rows} rows × {cols} columns ({numeric_count} numeric, {cat_count} categorical, {date_count} date)'
        })

    if missing and missing.get('columns'):
        cols_list = missing['columns']
        most_missing = max(cols_list, key=lambda c: c['percentage'])
        least_missing = min(cols_list, key=lambda c: c['percentage'])
        pct = missing['overall_pct']
        severity = 'danger' if pct > 20 else 'warning' if pct > 5 else 'info'
        insights.append({
            'icon': 'fa-exclamation-triangle', 'color': severity,
            'title': 'Missing Values',
            'text': f'{missing["total_missing"]} total ({pct}%) across {len(cols_list)} columns. Most missing: {most_missing["column"]} ({most_missing["percentage"]}%)'
        })

    if outliers:
        cols_with = [(col, o) for col, o in outliers.items() if o['outlier_count'] > 0]
        if cols_with:
            worst = max(cols_with, key=lambda x: x[1]['outlier_pct'])
            insights.append({
                'icon': 'fa-chart-box', 'color': 'warning',
                'title': 'Outlier Detection',
                'text': f'{len(cols_with)} columns have outliers. Highest: {worst[0]} ({worst[1]["outlier_pct"]}%) — consider capping or transformation'
            })

    if correlation and correlation.get('high_corr'):
        n = len(correlation['high_corr'])
        insights.append({
            'icon': 'fa-link', 'color': 'info',
            'title': 'Correlation Analysis',
            'text': f'{n} highly correlated feature pairs found. Consider dimensionality reduction or feature selection'
        })

    if stats:
        skewed = [(col, s) for col, s in stats.items() if abs(s.get('skewness', 0)) > 1]
        if skewed:
            worst_skew = max(skewed, key=lambda x: abs(x[1]['skewness']))
            insights.append({
                'icon': 'fa-chart-bar', 'color': 'secondary',
                'title': 'Distribution Insights',
                'text': f'{len(skewed)} skewed features detected. Highest skew: {worst_skew[0]} ({worst_skew[1]["skewness"]}) — log transform recommended'
            })

    if cat_analysis:
        high_card = [(col, c) for col, c in cat_analysis.items() if c.get('unique_count', 0) > 50]
        if high_card:
            insights.append({
                'icon': 'fa-tag', 'color': 'success',
                'title': 'Categorical Data',
                'text': f'{len(high_card)} high-cardinality categorical columns found. Consider target encoding or feature hashing'
            })

    if ts and ts.get('value_column'):
        insights.append({
            'icon': 'fa-chart-line', 'color': 'info',
            'title': 'Time Series Detected',
            'text': f'Date column "{ts["date_column"]}" with value "{ts["value_column"]}". Range: {ts.get("date_min")} to {ts.get("date_max")}'
        })

    if not insights:
        insights.append({
            'icon': 'fa-check-circle', 'color': 'success',
            'title': 'Analysis Complete',
            'text': 'EDA analysis completed successfully. Explore the tabs for detailed insights.'
        })

    return insights
