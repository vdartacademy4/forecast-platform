import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8-whitegrid')


def _save_matplotlib(fig, filepath, dpi=100):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath


def _plotly_to_json(fig):
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


def histogram(df, column, output_dir):
    data = df[column].dropna()
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=30, marker_color='#0d6efd')])
    fig.update_layout(title=f'Histogram — {column}', xaxis_title=column, yaxis_title='Frequency',
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, color='#0d6efd', edgecolor='white', alpha=0.8)
    ax.set_title(f'Histogram — {column}'); ax.set_xlabel(column); ax.set_ylabel('Frequency')
    img_path = os.path.join(output_dir, 'images', f'histogram_{column}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def boxplot(df, column, output_dir):
    data = df[column].dropna()
    fig = go.Figure(data=[go.Box(y=data, name=column, marker_color='#0d6efd')])
    fig.update_layout(title=f'Box Plot — {column}', template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=True, patch_artist=True, boxprops=dict(facecolor='#0d6efd', alpha=0.7))
    ax.set_title(f'Box Plot — {column}'); ax.set_ylabel(column)
    img_path = os.path.join(output_dir, 'images', f'boxplot_{column}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None, None
    corr = numeric_df.corr().round(4)
    cols = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=cols, colorscale='RdBu_r', zmin=-1, zmax=1,
                                     text=np.round(z, 2), texttemplate='%{text}', textfont=dict(size=9)))
    fig.update_layout(title='Correlation Heatmap', template='plotly_white', height=500, width=600,
                      margin=dict(l=80, r=20, t=50, b=80))

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
        return None, None
    cols = [m['column'] for m in missing_data]
    vals = [m['count'] for m in missing_data]

    fig = go.Figure(data=[go.Bar(x=cols, y=vals, marker_color='#dc3545', text=vals, textposition='outside')])
    fig.update_layout(title='Missing Values by Column', xaxis_title='Column', yaxis_title='Missing Count',
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=80))

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cols, vals, color='#dc3545', alpha=0.8)
    ax.bar_label(bars, fontsize=9)
    ax.set_title('Missing Values by Column'); ax.set_xlabel('Column'); ax.set_ylabel('Missing Count')
    ax.tick_params(axis='x', rotation=45)
    img_path = os.path.join(output_dir, 'images', 'missing_values.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def line_chart(df, x_col, y_col, output_dir, title=None):
    data = df[[x_col, y_col]].dropna()
    fig = px.line(data, x=x_col, y=y_col, title=title or f'{y_col} over {x_col}')
    fig.update_layout(template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data[x_col].values, data[y_col].values, color='#0d6efd', linewidth=1.5)
    ax.set_title(title or f'{y_col} over {x_col}')
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    img_path = os.path.join(output_dir, 'images', f'line_{y_col}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def bar_chart(freq_dict, title, output_dir, filename='bar_chart'):
    fig = go.Figure(data=[go.Bar(x=list(freq_dict.keys()), y=list(freq_dict.values()),
                                  marker_color='#198754', text=list(freq_dict.values()), textposition='outside')])
    fig.update_layout(title=title, template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=80))

    mpl_fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(freq_dict.keys()), list(freq_dict.values()), color='#198754', alpha=0.8)
    ax.bar_label(bars, fontsize=9)
    ax.set_title(title); ax.tick_params(axis='x', rotation=45)
    img_path = os.path.join(output_dir, 'images', f'{filename}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def pie_chart(freq_dict, title, output_dir, filename='pie_chart'):
    labels = list(freq_dict.keys())
    values = list(freq_dict.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title=title, template='plotly_white', height=400)

    mpl_fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    img_path = os.path.join(output_dir, 'images', f'{filename}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path


def trend_chart(df, date_col, value_col, output_dir):
    data = df[[date_col, value_col]].dropna().sort_values(date_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=data[value_col], mode='lines+markers',
                              name=value_col, line=dict(color='#0d6efd', width=2)))
    fig.update_layout(title=f'Trend — {value_col}', xaxis_title='Date', yaxis_title=value_col,
                      template='plotly_white', height=400, margin=dict(l=50, r=20, t=50, b=50))

    mpl_fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data[date_col].values, data[value_col].values, color='#0d6efd', linewidth=1.5, marker='o', markersize=3)
    ax.set_title(f'Trend — {value_col}'); ax.set_xlabel('Date'); ax.set_ylabel(value_col)
    img_path = os.path.join(output_dir, 'images', f'trend_{value_col}.png')
    _save_matplotlib(mpl_fig, img_path)
    return _plotly_to_json(fig), img_path
