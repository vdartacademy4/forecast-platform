from database import db
from datetime import datetime


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, default=0)
    rows_count = db.Column(db.Integer, default=0)
    columns_count = db.Column(db.Integer, default=0)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

    validation_reports = db.relationship(
        'ValidationReport', backref='dataset',
        lazy='dynamic', cascade='all, delete-orphan'
    )
    user = db.relationship('User', backref=db.backref('datasets', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'rows_count': self.rows_count,
            'columns_count': self.columns_count,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None
        }

    def __repr__(self):
        return f'<Dataset {self.file_name}>'
