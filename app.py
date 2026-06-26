from flask import Flask
from config import Config
from database import db
from routes.auth import auth_bp
from routes.dataset_routes import dataset_bp
from routes.eda import eda_bp
from routes.preprocessing import preprocessing_bp
from routes.forecasting import forecasting_bp
from routes.comparison import comparison_bp
from routes.reports import reports_bp
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['EDA_REPORTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

    db.init_app(app)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(forecasting_bp)
    app.register_blueprint(comparison_bp)
    app.register_blueprint(reports_bp)

    with app.app_context():
        from models.user_model import User
        from models.dataset_model import Dataset
        from models.validation_report_model import ValidationReport
        from models.eda_report_model import EDAReport
        from models.activity_log_model import ActivityLog
        from models.preprocessing_report_model import PreprocessingReport
        from models.forecast_report_model import ForecastReport
        from models.comparison_report_model import ComparisonReport
        from models.report_model import Report
        from models.analysis_history_model import AnalysisHistory
        db.create_all()

    return app
