import os
import pandas as pd
from flask import current_app
from database import db
from models.dataset_model import Dataset
from models.user_model import User
from utils.file_utils import allowed_file, save_uploaded_file


def get_upload_folder():
    return current_app.config.get('UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploads'))


def read_dataframe(file_path, extension, nrows=None):
    try:
        if extension == 'csv':
            return pd.read_csv(file_path, nrows=nrows)
        elif extension in ('xls', 'xlsx'):
            return pd.read_excel(file_path, nrows=nrows)
        return None
    except Exception:
        return None


def upload_dataset(file, user_id):
    if not file or not allowed_file(file.filename):
        return None, 'Invalid file type. Allowed: CSV, XLS, XLSX.'

    upload_folder = get_upload_folder()
    os.makedirs(upload_folder, exist_ok=True)

    file_info = save_uploaded_file(file, upload_folder)
    extension = file_info['extension']

    df = read_dataframe(file_info['file_path'], extension)
    if df is None:
        os.remove(file_info['file_path'])
        return None, 'Unable to read file. It may be corrupted.'

    if df.empty:
        os.remove(file_info['file_path'])
        return None, 'Uploaded file is empty.'

    dataset = Dataset(
        user_id=user_id,
        file_name=file_info['original_name'],
        file_path=file_info['file_path'],
        file_size=file_info['file_size'],
        rows_count=len(df),
        columns_count=len(df.columns)
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset, None


def get_user_datasets(user_id, page=1, per_page=20):
    query = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.upload_date.desc())
    total = query.count()
    datasets = query.offset((page - 1) * per_page).limit(per_page).all()
    return datasets, total


def get_dataset(dataset_id, user_id):
    return Dataset.query.filter_by(id=dataset_id, user_id=user_id).first()


def delete_dataset(dataset):
    file_path = dataset.file_path
    db.session.delete(dataset)
    db.session.commit()
    if os.path.exists(file_path):
        os.remove(file_path)
    return True


def get_preview_data(file_path, extension, rows=20):
    df = read_dataframe(file_path, extension, nrows=rows)
    if df is None:
        return None, None, None
    columns = list(df.columns)
    dtypes = {str(col): str(df[col].dtype) for col in df.columns}
    data = df.fillna('').values.tolist()
    return columns, data, dtypes


def get_column_info(file_path, extension):
    df = read_dataframe(file_path, extension)
    if df is None:
        return None, None
    return len(df), len(df.columns)
