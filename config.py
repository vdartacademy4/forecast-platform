import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get(
        'SECRET_KEY',
        'forecastiq-secret-key-change-in-production'
    )
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(basedir, 'instance', 'forecastiq.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = 86400
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = 'Lax'
    UPLOAD_FOLDER = os.path.join(basedir, 'data', 'uploads')
    EDA_REPORTS_FOLDER = os.path.join(basedir, 'reports', 'eda_reports')
    PROCESSED_FOLDER = os.path.join(basedir, 'data', 'processed')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
