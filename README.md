# ForecastIQ — Universal Forecasting Platform

A web-based AI/ML forecasting platform where users can upload datasets, perform automated data validation, exploratory data analysis (EDA), data preprocessing, forecasting, model comparison, and report generation.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5, Plotly.js |
| Backend | Python Flask, REST API |
| Database | SQLite (SQLAlchemy ORM) |
| ML/AI | Pandas, NumPy, Scikit-learn, Statsmodels, Prophet, XGBoost, TensorFlow/Keras |
| Visualization | Plotly, Matplotlib |

---

## Current Features (Phase 4)

### Phase 1 — System Architecture
- Complete architecture document (`PHASE1_ARCHITECTURE.md`)
- Folder structure, database schema, API blueprint, development roadmap

### Phase 2 — Authentication
- User registration with validation
- Login/logout with session management
- Werkzeug password hashing (pbkdf2:sha256)
- Protected routes with `@login_required` decorator
- User profile page

### Phase 3 — Dataset Upload & Validation
- Upload CSV, XLS, XLSX files (drag-and-drop, max 50MB)
- Dataset preview with data table
- Upload history with status badges
- 9 validation checks:
  - Empty dataset detection
  - Missing value analysis (count + percentage)
  - Duplicate row detection
  - Empty column detection
  - Duplicate column detection
  - Data type classification (numeric, categorical, date, text)
  - Date column auto-detection
  - Dataset size analysis

### Phase 4 — EDA Engine (Dual Mode)

**Automatic EDA:**
- Dataset overview (rows, columns, memory usage)
- Statistical summary (mean, median, mode, std, variance, quartiles, skewness, kurtosis)
- Missing value analysis
- Correlation matrix + heatmap
- Outlier detection (IQR + Z-Score)
- Distribution analysis (histograms, boxplots)
- Categorical analysis (frequency, bar charts, pie charts)
- Time series analysis (trend, date range, frequency)
- Feature insights (target suggestions, high correlations, outlier-prone columns)
- 7+ interactive Plotly chart types
- HTML report generation with embedded images

**Manual EDA:**
- Select specific statistics, analyses, and visualizations via checkboxes
- Generate only what you need

---

## Project Structure

```
forecast_platform/
├── app.py                  # Flask app factory
├── config.py               # Configuration
├── database.py             # SQLAlchemy init
├── run.py                  # Entry point
├── requirements.txt
│
├── routes/
│   ├── auth.py             # Authentication routes
│   ├── dataset_routes.py   # Upload, validate, preview routes
│   └── eda.py              # EDA routes (auto/manual/report)
│
├── models/
│   ├── user_model.py
│   ├── dataset_model.py
│   ├── validation_report_model.py
│   └── eda_report_model.py
│
├── services/
│   ├── dataset_service.py  # File upload, preview
│   ├── validation_service.py  # Data quality checks
│   ├── eda_service.py      # EDA computation & report generation
│   └── chart_service.py    # Plotly + Matplotlib chart generation
│
├── utils/
│   └── file_utils.py       # File validation, secure naming
│
├── templates/
│   ├── base.html           # Bootstrap 5 base template
│   ├── login.html, register.html
│   ├── dashboard.html, profile.html
│   ├── upload_dataset.html, upload_history.html
│   ├── dataset_preview.html, validation_report.html
│   ├── eda_mode.html, eda_dashboard.html
│   ├── manual_eda.html, eda_summary.html, eda_charts.html
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── data/uploads/           # Uploaded dataset files
├── reports/eda_reports/    # Generated EDA reports & charts
└── instance/               # SQLite database
```

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `users` | User accounts (id, username, email, password_hash, created_at) |
| `datasets` | Uploaded file metadata (file_name, rows, columns, size, upload_date) |
| `validation_reports` | Validation results (missing values, duplicates, column types, outliers) |
| `eda_reports` | EDA analysis records (mode, chart count, report path) |

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone or navigate to the project
cd forecast_platform

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## User Workflow

```
Register → Login → Upload Dataset → Validate
    → EDA Mode Selection
        → Automatic EDA (one-click full analysis)
        → Manual EDA (select specific analyses)
    → View Dashboard / Summary / Charts
    → Generate Report (HTML download)
```

---

## API Routes

### Authentication
| Route | Methods | Description |
|-------|---------|-------------|
| `/` | GET | Landing / redirect |
| `/login` | GET, POST | User login |
| `/register` | GET, POST | User registration |
| `/logout` | GET | Logout |
| `/dashboard` | GET | User dashboard |
| `/profile` | GET | User profile |

### Datasets
| Route | Methods | Description |
|-------|---------|-------------|
| `/upload` | GET, POST | Upload dataset |
| `/datasets` | GET | Upload history |
| `/dataset/<id>` | GET | Dataset detail + preview |
| `/dataset/<id>/preview` | GET | Full preview |
| `/dataset/<id>/validate` | POST | Run validation |
| `/validation-report/<id>` | GET | View validation report |
| `/dataset/<id>/delete` | POST | Delete dataset |

### EDA
| Route | Methods | Description |
|-------|---------|-------------|
| `/eda-mode/<id>` | GET | EDA mode selection |
| `/eda-auto/<id>` | POST | Run automatic EDA |
| `/eda-manual/<id>` | GET, POST | Run manual EDA |
| `/eda-dashboard/<id>` | GET | EDA results dashboard |
| `/eda-summary/<id>` | GET | EDA summary |
| `/eda-charts/<id>` | GET | Chart gallery |
| `/generate-eda-report/<id>` | POST | Generate HTML report |
| `/download-eda-report/<id>` | GET | Download report |

---

## Development Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1 | ✅ | System architecture, folder structure, database design |
| Phase 2 | ✅ | User authentication, session management |
| Phase 3 | ✅ | Dataset upload, validation engine |
| Phase 4 | ✅ | Dual-mode EDA (auto + manual), chart generation, reports |
| Phase 5 | ⏳ | Data preprocessing (missing values, scaling, encoding, train/test split) |
| Phase 6 | ⏳ | Forecasting engine (ARIMA, Prophet, XGBoost, LSTM) |
| Phase 7 | ⏳ | Model comparison & report download |
| Phase 8 | ⏳ | Deployment preparation (Docker, cloud) |

---

## Security

- Password hashing via Werkzeug (pbkdf2:sha256)
- Server-side session management (no large data in client cookies)
- Secure file upload with UUID renaming
- Input validation on all forms
- SQLAlchemy ORM (parameterized queries, no raw SQL)
- File type and size restrictions (CSV/XLS/XLSX, 50MB max)

---

## License

MIT
