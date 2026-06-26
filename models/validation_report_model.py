from database import db
from datetime import datetime


class ValidationReport(db.Model):
    __tablename__ = 'validation_reports'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    total_rows = db.Column(db.Integer, default=0)
    total_columns = db.Column(db.Integer, default=0)
    missing_values = db.Column(db.Text, default='[]')
    duplicate_rows = db.Column(db.Integer, default=0)
    empty_columns = db.Column(db.Text, default='[]')
    duplicate_columns = db.Column(db.Text, default='[]')
    date_columns = db.Column(db.Text, default='[]')
    numeric_columns = db.Column(db.Text, default='[]')
    categorical_columns = db.Column(db.Text, default='[]')
    column_types = db.Column(db.Text, default='{}')
    dataset_size_mb = db.Column(db.Float, default=0.0)
    validation_status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'missing_values': self.missing_values,
            'duplicate_rows': self.duplicate_rows,
            'empty_columns': self.empty_columns,
            'duplicate_columns': self.duplicate_columns,
            'date_columns': self.date_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'column_types': self.column_types,
            'dataset_size_mb': self.dataset_size_mb,
            'validation_status': self.validation_status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<ValidationReport dataset={self.dataset_id}>'
