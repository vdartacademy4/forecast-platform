from database import db
from datetime import datetime


class PreprocessingReport(db.Model):
    __tablename__ = 'preprocessing_reports'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    mode = db.Column(db.String(20), nullable=False)
    steps_applied = db.Column(db.Text, default='{}')
    original_shape = db.Column(db.String(50))
    processed_shape = db.Column(db.String(50))
    output_file = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PreprocessingReport dataset={self.dataset_id} mode={self.mode}>'
