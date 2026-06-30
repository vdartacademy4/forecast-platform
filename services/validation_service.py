import json
import pandas as pd
import numpy as np
from database import db
from models.validation_report_model import ValidationReport
from services.dataset_service import read_dataframe


def run_validation(dataset):
    extension = dataset.file_name.rsplit('.', 1)[1].lower()
    df = read_dataframe(dataset.file_path, extension)

    if df is None:
        report = ValidationReport(
            dataset_id=dataset.id,
            total_rows=0,
            total_columns=0,
            validation_status='failed'
        )
        db.session.add(report)
        db.session.commit()
        return report

    missing = check_missing_values(df)
    dup_rows = check_duplicate_rows(df)
    empty_cols = check_empty_columns(df)
    dup_cols = check_duplicate_columns(df)
    date_cols = detect_date_columns(df)
    numeric_cols, categorical_cols, col_types = detect_data_types(df)
    size_mb = round(dataset.file_size / (1024 * 1024), 4)

    report = ValidationReport(
        dataset_id=dataset.id,
        total_rows=len(df),
        total_columns=len(df.columns),
        missing_values=json.dumps(missing),
        duplicate_rows=dup_rows,
        empty_columns=json.dumps(empty_cols),
        duplicate_columns=json.dumps(dup_cols),
        date_columns=json.dumps(date_cols),
        numeric_columns=json.dumps(numeric_cols),
        categorical_columns=json.dumps(categorical_cols),
        column_types=json.dumps(col_types),
        dataset_size_mb=size_mb,
        validation_status='completed'
    )
    db.session.add(report)
    db.session.commit()
    return report


def check_missing_values(df):
    result = []
    for col in df.columns:
        count = int(df[col].isnull().sum())
        if count > 0:
            pct = round(float(count / len(df) * 100), 2) if len(df) > 0 else 0
            result.append({'column': col, 'count': count, 'percentage': pct})
    return result


def check_duplicate_rows(df):
    if len(df) == 0:
        return 0
    return int(df.duplicated().sum())


def check_empty_columns(df):
    return [str(col) for col in df.columns if df[col].isnull().all() or len(df[col].dropna()) == 0]


def check_duplicate_columns(df):
    groups = []
    seen = set()
    cols = list(df.columns)
    for i in range(len(cols)):
        if cols[i] in seen:
            continue
        group = [cols[i]]
        for j in range(i + 1, len(cols)):
            if cols[j] in seen:
                continue
            if df[cols[i]].dtype == df[cols[j]].dtype and df[cols[i]].equals(df[cols[j]]):
                group.append(cols[j])
                seen.add(cols[j])
        if len(group) > 1:
            groups.append(group)
        seen.add(cols[i])
    return groups


def detect_date_columns(df):
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(str(col))
            continue
        sample = df[col].dropna().astype(str).head(100)
        if len(sample) == 0:
            continue
        
        try:
            parsed = pd.to_datetime(
                sample,
                format="mixed",
                errors="coerce",
                utc=True
            )
        except TypeError:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                utc=True
            )

            if parsed.notna().mean() >= 0.8:
                date_cols.append(str(col))
                date_cols.append(str(col))
        except (ValueError, TypeError):
            pass
    return date_cols


def detect_data_types(df):
    numeric = []
    categorical = []
    col_types = {}

    for col in df.columns:
        col_name = str(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col_name)
            col_types[col_name] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            categorical.append(col_name)
            col_types[col_name] = 'date'
        elif pd.api.types.is_object_dtype(df[col]):
            unique_ratio = df[col].nunique() / max(len(df[col].dropna()), 1)
            if unique_ratio < 0.5 or df[col].nunique() < 20:
                categorical.append(col_name)
                col_types[col_name] = 'categorical'
            else:
                col_types[col_name] = 'text'
        else:
            col_types[col_name] = str(df[col].dtype)

    return numeric, categorical, col_types


def get_report(dataset_id):
    return ValidationReport.query.filter_by(dataset_id=dataset_id).order_by(
        ValidationReport.created_at.desc()
    ).first()
