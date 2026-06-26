from database import db
from datetime import datetime


class EDAReport(db.Model):
    __tablename__ = 'eda_reports'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    eda_mode = db.Column(db.String(20))
    report_path = db.Column(db.String(512))
    total_charts = db.Column(db.Integer, default=0)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<EDAReport dataset={self.dataset_id} mode={self.eda_mode}>'
