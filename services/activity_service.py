from database import db
from models.activity_log_model import ActivityLog
from datetime import datetime


def log_activity(user_id, action, entity_type=None, dataset_id=None, details=None):
    log = ActivityLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        dataset_id=dataset_id,
        details=details,
        timestamp=datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()
    return log


def get_recent_activities(user_id, limit=10):
    return ActivityLog.query.filter_by(user_id=user_id).order_by(
        ActivityLog.timestamp.desc()
    ).limit(limit).all()
