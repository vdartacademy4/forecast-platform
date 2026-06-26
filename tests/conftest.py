import os
import sys
import pytest
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app
from database import db as _db


@pytest.fixture(scope='session')
def app():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    app.config['EDA_REPORTS_FOLDER'] = tempfile.mkdtemp()
    app.config['PROCESSED_FOLDER'] = tempfile.mkdtemp()
    return app


@pytest.fixture(scope='function')
def db(app):
    with app.app_context():
        _db.create_all()
        yield _db
        _db.drop_all()


@pytest.fixture(scope='function')
def client(app, db):
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture(scope='function')
def auth_client(client, db):
    from models.user_model import User
    with client.application.app_context():
        user = User(username='testuser', email='test@example.com')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()
    client.post('/login', data={
        'username': 'testuser',
        'password': 'password123'
    })
    return client


@pytest.fixture(scope='function')
def sample_dataset(db):
    from models.dataset_model import Dataset
    ds = Dataset(
        user_id=1,
        file_name='test_data.csv',
        file_path='/tmp/test_data.csv',
        file_size=1024,
        rows_count=100,
        columns_count=5
    )
    db.session.add(ds)
    db.session.commit()
    return ds


@pytest.fixture(scope='function')
def sample_time_series_csv():
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    trend = np.linspace(0, 10, 200)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(200) / 30)
    noise = np.random.normal(0, 0.5, 200)
    values = 50 + trend + seasonal + noise

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], 200),
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(5, 2, 200)
    })
    path = os.path.join(tempfile.mkdtemp(), 'timeseries.csv')
    df.to_csv(path, index=False)
    return path
