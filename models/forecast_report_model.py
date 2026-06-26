from database import db
from datetime import datetime


class ForecastReport(db.Model):
    __tablename__ = 'forecast_reports'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    forecast_horizon = db.Column(db.Integer, nullable=False)
    target_column = db.Column(db.String(128))
    date_column = db.Column(db.String(128))
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    forecast_file = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ForecastReport dataset={self.dataset_id} model={self.model_name}>'
