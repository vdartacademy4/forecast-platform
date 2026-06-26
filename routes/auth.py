import os
from flask import (Blueprint, render_template, request, make_response,
                   redirect, url_for, flash, session)
from database import db
from models.user_model import User
from models.dataset_model import Dataset
from models.validation_report_model import ValidationReport
from models.eda_report_model import EDAReport
from services.activity_service import get_recent_activities
from functools import wraps

auth_bp = Blueprint('auth', __name__)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def _no_cache(resp):
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@auth_bp.after_request
def add_security_headers(resp):
    if 'user_id' in session:
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    return resp


@auth_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('auth.dashboard'))
    return redirect(url_for('auth.login'))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('auth.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username/email and password.', 'danger')
            return render_template('login.html')

        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.check_password(password):
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            resp = make_response(redirect(url_for('auth.dashboard')))
            return _no_cache(resp)

        flash('Invalid username/email or password.', 'danger')

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('auth.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []

        if not username:
            errors.append('Username is required.')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        elif not username.isalnum():
            errors.append('Username must contain only letters and numbers.')

        if not email:
            errors.append('Email is required.')
        elif '@' not in email:
            errors.append('Please enter a valid email address.')

        if not password:
            errors.append('Password is required.')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters.')

        if password != confirm_password:
            errors.append('Passwords do not match.')

        if not errors:
            if User.query.filter_by(username=username).first():
                errors.append('Username already exists.')
            if User.query.filter_by(email=email).first():
                errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        user = User(username=username, email=email)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')

    return render_template('register.html')


@auth_bp.route('/logout')
def logout():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    session.clear()
    session.permanent = False
    flash('You have been logged out successfully.', 'info')
    resp = make_response(redirect(url_for('auth.login')))
    return _no_cache(resp)


@auth_bp.route('/dashboard')
@login_required
def dashboard():
    uid = session['user_id']
    total_datasets = Dataset.query.filter_by(user_id=uid).count()
    validated = db.session.query(ValidationReport).join(
        Dataset, ValidationReport.dataset_id == Dataset.id
    ).filter(
        Dataset.user_id == uid,
        ValidationReport.validation_status == 'completed'
    ).count()
    eda_count = EDAReport.query.join(
        Dataset, EDAReport.dataset_id == Dataset.id
    ).filter(Dataset.user_id == uid).count()
    activities = get_recent_activities(uid, limit=10)

    from models.analysis_history_model import AnalysisHistory
    analyses = AnalysisHistory.query.filter_by(user_id=uid).order_by(
        AnalysisHistory.created_at.desc()
    ).limit(5).all()
    analysis_count = AnalysisHistory.query.filter_by(user_id=uid).count()

    from models.forecast_report_model import ForecastReport
    forecast_count = db.session.query(ForecastReport).join(
        Dataset, ForecastReport.dataset_id == Dataset.id
    ).filter(Dataset.user_id == uid).count()

    from sqlalchemy import func
    best_models = db.session.query(
        AnalysisHistory.best_model,
        func.count(AnalysisHistory.id).label('count')
    ).filter(
        AnalysisHistory.user_id == uid,
        AnalysisHistory.best_model.isnot(None)
    ).group_by(AnalysisHistory.best_model).order_by(
        func.count(AnalysisHistory.id).desc()
    ).all()

    return render_template('dashboard.html',
        username=session.get('username'),
        total_datasets=total_datasets,
        validated_count=validated,
        eda_count=eda_count,
        activities=activities,
        analyses=analyses,
        analysis_count=analysis_count,
        forecast_count=forecast_count,
        best_models=best_models)


@auth_bp.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        session.permanent = False
        flash('User not found. Please log in again.', 'warning')
        return redirect(url_for('auth.login'))
    return render_template('profile.html', user=user)
