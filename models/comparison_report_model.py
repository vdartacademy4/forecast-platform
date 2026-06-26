from database import db
from datetime import datetime


class ComparisonReport(db.Model):
    __tablename__ = 'comparison_reports'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    best_model = db.Column(db.String(50), nullable=False)
    ranking = db.Column(db.Text, nullable=False)
    comparison_metrics = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ComparisonReport dataset={self.dataset_id} best={self.best_model}>'
