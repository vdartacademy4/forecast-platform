from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db():
    from models.user_model import User
    db.create_all()
