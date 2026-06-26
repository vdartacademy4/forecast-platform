from database import db
from datetime import datetime


class AnalysisHistory(db.Model):
    __tablename__ = 'analysis_history'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dataset_name = db.Column(db.String(255), nullable=False)
    best_model = db.Column(db.String(50))
    forecast_model = db.Column(db.String(50))
    forecast_horizon = db.Column(db.Integer)
    total_rows = db.Column(db.Integer, default=0)
    total_columns = db.Column(db.Integer, default=0)
    total_workflow_time = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'best_model': self.best_model,
            'forecast_model': self.forecast_model,
            'forecast_horizon': self.forecast_horizon,
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'total_workflow_time': self.total_workflow_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f'<AnalysisHistory dataset={self.dataset_id} best={self.best_model}>'
