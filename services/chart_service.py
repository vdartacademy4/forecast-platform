import os
import json
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats

plt.style.use('seaborn-v0_8-whitegrid')


def _log_error(chart_name, column, error):
    traceback.print_exc()
    print(f"[CHART ERROR] {chart_name} | Column: {column} | {error}")


def _prepare_numeric(series, column):
    """Validate, clean, and convert a series to clean float values."""
    if series is None or len(series) == 0:
        _log_error('prepare_numeric', column, 'Empty series')
        return None

    cleaned = pd.to_numeric(series, errors='coerce')
    cleaned = cleaned.dropna()
    if len(cleaned) == 0:
        _log_error('prepare_numeric', column, 'No valid numeric values after conversion')
        return None

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cleaned) == 0:
        _log_error('prepare_numeric', column, 'No finite values after removing infinity')
        return None

    if cleaned.nunique() <= 1:
        _log_error('prepare_numeric', column, f'Constant column (single unique value: {cleaned.iloc[0]})')
        return None

    if len(cleaned) < 2:
        _log_error('prepare_numeric', column, f'Only {len(cleaned)} non-null value(s), need at least 2')
        return None

    return cleaned.astype(float)


def _validate_output(fig, chart_name, column):
    """Verify Plotly figure has at least one trace with data."""
    if fig is None:
        _log_error('validate_output', column, f'{chart_name}: Figure is None')
        return None
    try:
        data = fig.get('data') if isinstance(fig, dict) else fig.data
        if data is None or len(data) == 0:
            _log_error('validate_output', column, f'{chart_name}: No traces in figure')
            return None
        return fig
    except Exception as e:
        _log_error('validate_output', column, f'{chart_name}: Validation exception: {e}')
        return None


def _save_matplotlib(fig, filepath, dpi=100):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath


def _plotly_to_json(fig):
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


def histogram(df, column, output_dir):
    data = _prepare_numeric(df[column], column)
    if data is None:
        return None, None
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30, marker_color='#0d6efd')])
    fig.update_layout(title=f'Histogram — {column}', xaxis_title=column, yaxis_title='Frequency',
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))
    fig = _validate_output(fig, 'histogram', column)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, color='#0d6efd', edgecolor='white', alpha=0.8)
    ax.set_title(f'Histogram — {column}'); ax.set_xlabel(column); ax.set_ylabel('Frequency')
    img_path = os.path.join(output_dir, 'images', f'histogram_{column}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def density_plot(df, column, output_dir):
    data = _prepare_numeric(df[column], column)
    if data is None or len(data) < 3:
        _log_error('density_plot', column, f'Need at least 3 points, got {len(data) if data is not None else 0}')
        return None, None
    try:
        kde = scipy_stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        y_kde = kde(x_range)
        fig = go.Figure(data=[go.Scatter(
            x=x_range, y=y_kde, mode='lines',
            fill='tozeroy', line=dict(color='#0d6efd', width=2),
            name='Density'
        )])
        fig.update_layout(title=f'Density Plot — {column}',
                          xaxis_title=column, yaxis_title='Density',
                          template='plotly_white', height=400,
                          margin=dict(l=50, r=20, t=50, b=50))
        fig = _validate_output(fig, 'density_plot', column)
        if fig is None:
            return None, None

        mpl_fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(x_range, y_kde, alpha=0.5, color='#0d6efd')
        ax.plot(x_range, y_kde, color='#0d6efd', linewidth=2)
        ax.set_title(f'Density Plot — {column}'); ax.set_xlabel(column); ax.set_ylabel('Density')
        img_path = os.path.join(output_dir, 'images', f'density_{column}.png')
        _save_matplotlib(mpl_fig, img_path)
        return _plotly_to_json(fig), img_path
    except Exception as e:
        _log_error('density_plot', column, e)
        return None, None


def boxplot(df, column, output_dir):
    data = _prepare_numeric(df[column], column)
    if data is None:
        return None, None
    fig = go.Figure(data=[go.Box(y=data, name=column, marker_color='#0d6efd')])
    fig.update_layout(title=f'Box Plot — {column}', template='plotly_white', height=400,
                      margin=dict(l=50, r=20, t=50, b=50))
    fig = _validate_output(fig, 'boxplot', column)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=True, patch_artist=True, boxprops=dict(facecolor='#0d6efd', alpha=0.7))
    ax.set_title(f'Box Plot — {column}'); ax.set_ylabel(column)
    img_path = os.path.join(output_dir, 'images', f'boxplot_{column}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    numeric_df = numeric_df.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)
    if numeric_df.shape[1] < 2:
        _log_error('correlation_heatmap', '', 'Need at least 2 numeric columns with data')
        return None, None
    corr = numeric_df.corr().round(4)
    cols = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=cols, colorscale='RdBu_r', zmin=-1, zmax=1,
                                     text=np.round(z, 2), texttemplate='%{text}', textfont=dict(size=9)))
    fig.update_layout(title='Correlation Heatmap', template='plotly_white', height=500, width=600,
                      margin=dict(l=80, r=20, t=50, b=80))
    fig = _validate_output(fig, 'correlation_heatmap', 'all')
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(z, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{z[i, j]:.2f}', ha='center', va='center', fontsize=7, color='black')
    mpl_fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Correlation Heatmap')
    img_path = os.path.join(output_dir, 'images', 'correlation_heatmap.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def missing_values_chart(missing_data, output_dir):
    if not missing_data:
        _log_error('missing_values_chart', '', 'No missing data provided')
        return None, None
    cols = [m['column'] for m in missing_data]
    vals = []
    for v in missing_data:
        try:
            vals.append(float(v['count']))
        except (ValueError, TypeError):
            vals.append(0)
    if len(cols) == 0:
        _log_error('missing_values_chart', '', 'Empty columns list')
        return None, None

    fig = go.Figure(data=[go.Bar(x=cols, y=vals, marker_color='#dc3545', text=vals, textposition='outside')])
    fig.update_layout(title='Missing Values by Column', xaxis_title='Column', yaxis_title='Missing Count',
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=80))
    fig = _validate_output(fig, 'missing_values_chart', 'all')
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cols, vals, color='#dc3545', alpha=0.8)
    ax.bar_label(bars, fontsize=9)
    ax.set_title('Missing Values by Column'); ax.set_xlabel('Column'); ax.set_ylabel('Missing Count')
    ax.tick_params(axis='x', rotation=45)
    img_path = os.path.join(output_dir, 'images', 'missing_values.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def missing_heatmap(df, output_dir):
    missing_binary = df.isnull().astype(int)
    total_missing = int(missing_binary.sum().sum())
    if total_missing == 0:
        _log_error('missing_heatmap', '', 'No missing values to plot')
        return None, None

    max_cols = 50
    cols = missing_binary.columns.tolist()
    if len(cols) > max_cols:
        cols = cols[:max_cols]
    plot_data = missing_binary[cols]

    max_rows = 2000
    if len(plot_data) > max_rows:
        indices = np.linspace(0, len(plot_data) - 1, max_rows, dtype=int)
        plot_data = plot_data.iloc[indices]

    fig = go.Figure(data=go.Heatmap(
        z=plot_data.T.values,
        x=list(range(len(plot_data))),
        y=cols,
        colorscale=[[0, '#e8f5e9'], [1, '#dc3545']],
        showscale=False
    ))
    fig.update_layout(title='Missing Value Heatmap', template='plotly_white',
                      height=max(300, len(cols) * 20), width=700,
                      margin=dict(l=100, r=20, t=50, b=50),
                      xaxis_visible=False)
    fig = _validate_output(fig, 'missing_heatmap', 'all')
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.4)))
    ax.imshow(plot_data.T.values, cmap='Reds', aspect='auto', interpolation='none')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_title('Missing Value Heatmap')
    img_path = os.path.join(output_dir, 'images', 'missing_heatmap.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def line_chart(df, x_col, y_col, output_dir, title=None):
    y_data = _prepare_numeric(df[y_col], y_col)
    if y_data is None:
        return None, None
    x_raw = df[x_col].dropna()
    valid_idx = y_data.index.intersection(x_raw.index)
    if len(valid_idx) < 2:
        _log_error('line_chart', y_col, f'Need at least 2 paired points, got {len(valid_idx)}')
        return None, None
    x_vals = x_raw.loc[valid_idx].astype(str).tolist()
    y_vals = y_data.loc[valid_idx].tolist()
    fig = go.Figure(data=[go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=y_col,
                                      line=dict(color='#0d6efd', width=1.5))])
    fig.update_layout(title=title or f'{y_col} over {x_col}', template='plotly_white', height=400,
                      margin=dict(l=50, r=20, t=50, b=50))
    fig = _validate_output(fig, 'line_chart', y_col)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(y_vals)), y_vals, color='#0d6efd', linewidth=1.5)
    ax.set_title(title or f'{y_col} over {x_col}')
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    img_path = os.path.join(output_dir, 'images', f'line_{y_col}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def bar_chart(freq_dict, title, output_dir, filename='bar_chart'):
    if not freq_dict:
        _log_error('bar_chart', filename, 'Empty frequency dictionary')
        return None, None
    keys = [str(k) for k in freq_dict.keys()]
    vals = []
    for v in freq_dict.values():
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(0)
    if len(keys) == 0:
        _log_error('bar_chart', filename, 'No keys after conversion')
        return None, None

    fig = go.Figure(data=[go.Bar(x=keys, y=vals,
                                  marker_color='#198754', text=vals, textposition='outside')])
    fig.update_layout(title=title, template='plotly_white', height=400,
                      margin=dict(l=50, r=20, t=50, b=80))
    fig = _validate_output(fig, 'bar_chart', filename)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(keys, vals, color='#198754', alpha=0.8)
    ax.bar_label(bars, fontsize=9)
    ax.set_title(title); ax.tick_params(axis='x', rotation=45)
    img_path = os.path.join(output_dir, 'images', f'{filename}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def pie_chart(freq_dict, title, output_dir, filename='pie_chart'):
    if not freq_dict:
        _log_error('pie_chart', filename, 'Empty frequency dictionary')
        return None, None
    labels = [str(k) for k in freq_dict.keys()]
    values = []
    for v in freq_dict.values():
        try:
            values.append(float(v))
        except (ValueError, TypeError):
            values.append(0)
    if len(labels) == 0:
        _log_error('pie_chart', filename, 'No labels after conversion')
        return None, None

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title=title, template='plotly_white', height=400)
    fig = _validate_output(fig, 'pie_chart', filename)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    img_path = os.path.join(output_dir, 'images', f'{filename}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def _prepare_datetime(series, column_name=""):
    """
    Safely convert a column to datetime.

    Supports:
    - Mixed datetime formats
    - Timezone-aware timestamps
    - Timezone-naive timestamps
    - Weather datasets
    """

    try:

        converted = pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
            utc=True
        )

        converted = converted.dt.tz_localize(None)

        converted = converted.dropna()

        if converted.empty:
            _log_error(
                "prepare_datetime",
                column_name,
                "No valid datetime values"
            )
            return None

        return converted

    except Exception as e:

        _log_error(
            "prepare_datetime",
            column_name,
            f"Conversion failed: {e}"
        )

        return None


def trend_chart(df, date_col, value_col, output_dir):
    y_data = _prepare_numeric(df[value_col], value_col)
    if y_data is None:
        return None, None

    dt_index = _prepare_datetime(df[date_col], date_col)
    if dt_index is None:
        _log_error('trend_chart', value_col, f'Could not parse date column: {date_col}')
        return None, None

    common = y_data.index.intersection(dt_index.index)
    if len(common) < 2:
        _log_error('trend_chart', value_col, f'Need at least 2 paired date-value points, got {len(common)}')
        return None, None

    x_vals = dt_index.loc[common].sort_values()
    y_vals = y_data.loc[x_vals.index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                              name=value_col, line=dict(color='#0d6efd', width=2)))
    fig.update_layout(title=f'Trend — {value_col}', xaxis_title='Date', yaxis_title=value_col,
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))
    fig = _validate_output(fig, 'trend_chart', value_col)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals.values, y_vals.values, color='#0d6efd', linewidth=1.5, marker='o', markersize=3)
    ax.set_title(f'Trend — {value_col}'); ax.set_xlabel('Date'); ax.set_ylabel(value_col)
    img_path = os.path.join(output_dir, 'images', f'trend_{value_col}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def rolling_average_chart(df, date_col, value_col, window, output_dir):
    y_data = _prepare_numeric(df[value_col], value_col)
    if y_data is None:
        return None, None

    dt_index = _prepare_datetime(df[date_col], date_col)
    if dt_index is None:
        _log_error('rolling_average_chart', value_col, f'Could not parse date column: {date_col}')
        return None, None

    common = y_data.index.intersection(dt_index.index)
    if len(common) < window:
        _log_error('rolling_average_chart', value_col,
                   f'Need at least {window} paired points, got {len(common)}')
        return None, None

    x_vals = dt_index.loc[common].sort_values()
    y_vals = y_data.loc[x_vals.index]
    rolling_vals = y_vals.rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals,
                              mode='lines', name='Original',
                              line=dict(color='#adb5bd', width=1)))
    fig.add_trace(go.Scatter(x=x_vals, y=rolling_vals,
                              mode='lines', name=f'{window}-Period Rolling Avg',
                              line=dict(color='#dc3545', width=2)))
    fig.update_layout(title=f'Trend with {window}-Period Rolling Average',
                      xaxis_title='Date', yaxis_title=value_col,
                      template='plotly_white', height=400,
                      margin=dict(l=50, r=20, t=50, b=50))
    fig = _validate_output(fig, 'rolling_average_chart', value_col)
    if fig is None:
        return None, None

    mpl_fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals.values, y_vals.values, color='#adb5bd', linewidth=1, alpha=0.7, label='Original')
    ax.plot(x_vals.values, rolling_vals.values, color='#dc3545', linewidth=2, label=f'{window}-Period Avg')
    ax.set_title(f'Trend with {window}-Period Rolling Average')
    ax.set_xlabel('Date'); ax.set_ylabel(value_col)
    ax.legend()
    img_path = os.path.join(output_dir, 'images', f'rolling_{value_col}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path
