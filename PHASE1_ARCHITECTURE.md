# ForecastIQ — Universal Forecasting Platform
## Phase 1 — System Architecture Document

---

# 1. Complete Project Architecture

## 1.1 Frontend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT BROWSER                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Bootstrap 5 + Custom CSS (UI Framework)                   │
│   ├── Responsive Layout (Mobile-first)                      │
│   ├── Dark/Light Theme Support                              │
│   └── Custom Components (charts, tables, forms)             │
│                                                             │
│   JavaScript (Vanilla + Fetch API)                          │
│   ├── REST Client Layer (api.js)                            │
│   ├── Chart Rendering (Plotly.js CDN)                       │
│   ├── Session Management (sessionStorage)                   │
│   └── Form Validation & UX Logic                            │
│                                                             │
│   Page Components (12 Pages)                                │
│   ├── Landing Page (index.html)                             │
│   ├── Authentication (login.html, register.html)            │
│   ├── Dashboard (dashboard.html)                            │
│   ├── Dataset Upload (upload.html)                          │
│   ├── Validation Report (validation.html)                   │
│   ├── EDA Dashboard (eda.html)                              │
│   ├── Preprocessing (preprocessing.html)                    │
│   ├── Forecasting Module (forecast.html)                    │
│   ├── Model Comparison (comparison.html)                    │
│   ├── Report Download (reports.html)                        │
│   └── User Profile (profile.html)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 Backend Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FLASK APPLICATION                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Application Layer                                          │
│   ├── app.py                     (App Factory)               │
│   ├── config.py                  (Configuration)             │
│   ├── extensions.py              (DB, Session init)          │
│   └── wsgi.py                    (Entry Point)               │
│                                                               │
│   Blueprint Modules (REST APIs)                              │
│   ├── auth_routes.py             (Authentication)            │
│   ├── dataset_routes.py          (Dataset Management)        │
│   ├── validation_routes.py       (Data Validation)           │
│   ├── eda_routes.py              (Exploratory Data Analysis) │
│   ├── preprocessing_routes.py    (Data Preprocessing)        │
│   ├── forecast_routes.py         (Forecasting Engine)        │
│   ├── comparison_routes.py       (Model Comparison)          │
│   └── report_routes.py           (Report Generation)         │
│                                                               │
│   Service Layer (Business Logic)                             │
│   ├── auth_service.py            (Auth Logic)                │
│   ├── dataset_service.py         (Dataset Ops)               │
│   ├── validation_service.py      (Validation Logic)          │
│   ├── eda_service.py             (EDA Computation)           │
│   ├── preprocessing_service.py   (Clean/Transform)           │
│   ├── forecast_service.py        (Forecast Execution)        │
│   ├── comparison_service.py      (Model Compare)             │
│   └── report_service.py          (PDF/HTML Generation)       │
│                                                               │
│   Forecasting Engine Layer                                   │
│   ├── engine/                                                 │
│   │   ├── __init__.py                                        │
│   │   ├── base_model.py          (Abstract Base Model)       │
│   │   ├── arima_model.py         (Statsmodels ARIMA)         │
│   │   ├── sarima_model.py        (Seasonal ARIMA)            │
│   │   ├── prophet_model.py       (Facebook Prophet)          │
│   │   ├── xgboost_model.py       (XGBoost Regressor)         │
│   │   ├── lstm_model.py          (TensorFlow/Keras LSTM)     │
│   │   ├── linear_model.py        (Linear Regression)         │
│   │   ├── ensemble_model.py      (Ensemble Averaging)        │
│   │   └── model_factory.py       (Factory Pattern)           │
│                                                               │
│   Utility Layer                                              │
│   ├── utils/                                                 │
│   │   ├── __init__.py                                        │
│   │   ├── file_utils.py          (File Handling)             │
│   │   ├── date_utils.py          (Date Operations)           │
│   │   ├── metric_utils.py        (Error Metrics)             │
│   │   ├── plot_utils.py          (Plot Generation)           │
│   │   └── decorators.py          (Decorators)                │
│                                                               │
│   Data Layer                                                 │
│   ├── models/                                                │
│   │   ├── __init__.py                                        │
│   │   ├── user.py                                           │
│   │   ├── dataset.py                                        │
│   │   ├── validation_report.py                               │
│   │   ├── eda_report.py                                     │
│   │   ├── preprocessing_report.py                            │
│   │   ├── forecasting_model.py                               │
│   │   ├── forecasting_result.py                              │
│   │   ├── generated_report.py                                │
│   │   └── activity_log.py                                   │
│   └── database.py               (DB Init/Migration)         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 1.3 Database Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│    users     │     │    datasets      │     │ validation_reports  │
│──────────────│     │──────────────────│     │─────────────────────│
│ id (PK)      │────→│ id (PK)          │────→│ id (PK)             │
│ username     │     │ user_id (FK)     │     │ dataset_id (FK)     │
│ email        │     │ filename         │     │ status              │
│ password_hash│     │ original_name    │     │ missing_values      │
│ created_at   │     │ file_size        │     │ outliers            │
│ updated_at   │     │ file_path        │     │ duplicates          │
│ is_active    │     │ file_type        │     │ data_types          │
│ last_login   │     │ row_count        │     │ summary_stats       │
└──────────────┘     │ column_count     │     │ created_at          │
                     │ columns_json     │     └─────────────────────┘
                     │ description      │
                     │ uploaded_at      │     ┌─────────────────────┐
                     │ status           │     │   eda_reports       │
                     └──────────────────┘     │─────────────────────│
                            │                 │ id (PK)             │
                            │                 │ dataset_id (FK)     │
                            ▼                 │ correlation_matrix  │
                     ┌──────────────────┐     │ descriptive_stats   │
                     │preprocessing_re- │     │ distributions       │
                     │ports             │     │ time_series_plots   │
                     │──────────────────│     │ seasonal_decomp     │
                     │ id (PK)          │     │ stationarity_test   │
                     │ dataset_id (FK)  │     │ created_at          │
                     │ missing_handled  │     └─────────────────────┘
                     │ outlier_handled  │
                     │ scaling_applied  │     ┌──────────────────────┐
                     │ encoding_applied │     │ forecasting_models   │
                     │ feature_eng      │     │──────────────────────│
                     │ steps_log        │     │ id (PK)              │
                     │ created_at       │     │ user_id (FK)         │
                     └──────────────────┘     │ dataset_id (FK)      │
                            │                 │ model_name           │
                            ▼                 │ model_type           │
                     ┌──────────────────┐     │ parameters           │
                     │forecasting_results│    │ metrics              │
                     │──────────────────│     │ trained_on           │
                     │ id (PK)          │     │ created_at           │
                     │ model_id (FK)    │     │ is_active            │
                     │ forecast_horizon │     └──────────────────────┘
                     │ predictions_json │              │
                     │ confidence_upper │              ▼
                     │ confidence_lower │     ┌──────────────────────┐
                     │ metrics          │     │ generated_reports    │
                     │ created_at       │     │──────────────────────│
                     └──────────────────┘     │ id (PK)              │
                                              │ user_id (FK)         │
                     ┌──────────────────┐     │ type (eda/preprocess/│
                     │  activity_logs   │     │      forecast)       │
                     │──────────────────│     │ format (pdf/html)    │
                     │ id (PK)          │     │ file_path            │
                     │ user_id (FK)     │     │ created_at           │
                     │ action           │     └──────────────────────┘
                     │ entity_type      │
                     │ entity_id        │
                     │ details          │
                     │ ip_address       │
                     │ timestamp        │
                     └──────────────────┘
```

## 1.4 Forecasting Engine Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  FORECASTING ENGINE                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Input: Preprocessed DataFrame + Parameters                 │
│        │                                                     │
│        ▼                                                     │
│   ┌──────────────────────────────────────────────────┐       │
│   │            Model Factory (Factory Pattern)       │       │
│   │  model_factory.get_model(model_type, params)     │       │
│   └────────┬─────────┬─────────┬─────────┬──────────┘       │
│            │         │         │         │                    │
│            ▼         ▼         ▼         ▼                    │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│   │  ARIMA   │ │ Prophet  │ │ XGBoost  │ │   LSTM   │       │
│   │ SARIMA   │ │          │ │          │ │ (Keras)  │       │
│   └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘       │
│         │            │            │            │              │
│         ▼            ▼            ▼            ▼              │
│   ┌──────────────────────────────────────────────────┐       │
│   │           Train/Test Split                       │       │
│   │           TimeSeriesSplit CV                     │       │
│   └──────────────────────────────────────────────────┘       │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │           Model Training & Tuning                │       │
│   │           Hyperparameter Optimization            │       │
│   └──────────────────────────────────────────────────┘       │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │           Evaluation Metrics                     │       │
│   │  MAE, MSE, RMSE, MAPE, SMAPE, R²                │       │
│   └──────────────────────────────────────────────────┘       │
│         │                                                    │
│         ▼                                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │           Future Predictions                     │       │
│   │           Confidence Intervals                   │       │
│   └──────────────────────────────────────────────────┘       │
│         │                                                    │
│         ▼                                                    │
│   Output: Predictions + Metrics + Plots                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

# 2. Detailed Folder Structure

```
forecastiq/
│
├── app/
│   ├── __init__.py                    # App factory (create_app)
│   ├── config.py                      # Configuration classes
│   ├── extensions.py                  # Flask extensions init
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth_routes.py             # /api/auth/* endpoints
│   │   ├── dataset_routes.py          # /api/datasets/* endpoints
│   │   ├── validation_routes.py       # /api/validation/* endpoints
│   │   ├── eda_routes.py              # /api/eda/* endpoints
│   │   ├── preprocessing_routes.py    # /api/preprocessing/* endpoints
│   │   ├── forecast_routes.py         # /api/forecast/* endpoints
│   │   ├── comparison_routes.py       # /api/comparison/* endpoints
│   │   └── report_routes.py           # /api/reports/* endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py            # Register, login, logout
│   │   ├── dataset_service.py         # Upload, list, delete datasets
│   │   ├── validation_service.py      # Data quality checks
│   │   ├── eda_service.py             # Statistical analysis
│   │   ├── preprocessing_service.py   # Clean, scale, encode
│   │   ├── forecast_service.py        # Run forecast models
│   │   ├── comparison_service.py      # Compare multiple models
│   │   └── report_service.py          # Generate PDF/HTML reports
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── base_model.py              # Abstract base class
│   │   ├── arima_model.py             # ARIMA implementation
│   │   ├── sarima_model.py            # Seasonal ARIMA
│   │   ├── prophet_model.py           # Prophet implementation
│   │   ├── xgboost_model.py           # XGBoost regressor
│   │   ├── lstm_model.py              # TensorFlow LSTM
│   │   ├── linear_model.py            # LinearRegression
│   │   ├── ensemble_model.py          # Ensemble average
│   │   ├── model_factory.py           # Factory pattern selector
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py                    # User model (SQLAlchemy)
│   │   ├── dataset.py                 # Dataset model
│   │   ├── validation_report.py       # Validation report model
│   │   ├── eda_report.py              # EDA report model
│   │   ├── preprocessing_report.py    # Preprocessing report model
│   │   ├── forecasting_model.py       # Forecasting model metadata
│   │   ├── forecasting_result.py      # Forecasting results
│   │   ├── generated_report.py        # Generated reports
│   │   ├── activity_log.py            # Activity logs
│   │   └── enums.py                   # Enum types
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_utils.py              # File validation, storage
│   │   ├── date_utils.py              # Date parsing helpers
│   │   ├── metric_utils.py            # Metric computation
│   │   ├── plot_utils.py              # Plotly/Matplotlib helpers
│   │   ├── decorators.py              # @login_required, etc.
│   │   └── validators.py              # Input sanitization
│   │
│   └── templates/
│       └── reports/                   # Report HTML templates
│           ├── eda_report_template.html
│           ├── forecast_report_template.html
│           └── comparison_report_template.html
│
├── frontend/
│   ├── index.html                     # Landing page
│   ├── pages/
│   │   ├── login.html                 # User login
│   │   ├── register.html              # User registration
│   │   ├── dashboard.html             # Main dashboard
│   │   ├── upload.html                # Dataset upload
│   │   ├── validation.html            # Validation results
│   │   ├── eda.html                   # EDA dashboard
│   │   ├── preprocessing.html         # Preprocessing controls
│   │   ├── forecast.html              # Forecasting module
│   │   ├── comparison.html            # Model comparison
│   │   ├── reports.html               # Report download center
│   │   └── profile.html               # User profile
│   │
│   ├── assets/
│   │   ├── css/
│   │   │   ├── style.css              # Global styles
│   │   │   ├── landing.css            # Landing page styles
│   │   │   └── dark-theme.css         # Dark theme override
│   │   ├── js/
│   │   │   ├── api.js                 # Central API client (Fetch)
│   │   │   ├── auth.js                # Auth-related JS
│   │   │   ├── dashboard.js           # Dashboard JS
│   │   │   ├── upload.js              # Upload logic
│   │   │   ├── validation.js          # Validation display
│   │   │   ├── eda.js                 # EDA chart rendering
│   │   │   ├── preprocessing.js       # Preprocessing controls
│   │   │   ├── forecast.js            # Forecast interaction
│   │   │   ├── comparison.js          # Comparison charts
│   │   │   ├── reports.js             # Report download logic
│   │   │   ├── profile.js             # Profile management
│   │   │   ├── utils.js               # Utility functions
│   │   │   └── charts.js              # Reusable chart helpers
│   │   └── images/
│   │       ├── logo.svg               # App logo
│   │       ├── favicon.ico            # Favicon
│   │       └── illustrations/         # Page illustrations
│   │
│   └── components/                    # Reusable HTML components
│       ├── navbar.html
│       ├── sidebar.html
│       ├── footer.html
│       └── modals.html
│
├── data/                              # Data storage
│   ├── uploads/                       # Uploaded CSV files
│   ├── reports/                       # Generated reports
│   ├── models/                        # Saved model pickles
│   └── plots/                         # Generated plot images
│
├── migrations/                        # Database migrations
│   ├── __init__.py
│   ├── 001_initial_schema.sql
│   └── migrate.py                     # Migration script
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_auth.py
│   ├── test_datasets.py
│   ├── test_validation.py
│   ├── test_eda.py
│   ├── test_preprocessing.py
│   ├── test_forecast.py
│   ├── test_engine/
│   │   ├── __init__.py
│   │   ├── test_arima.py
│   │   ├── test_prophet.py
│   │   ├── test_xgboost.py
│   │   ├── test_lstm.py
│   │   └── test_model_factory.py
│   └── test_utils.py
│
├── docs/
│   ├── API.md                         # API documentation
│   └── setup.md                       # Setup instructions
│
├── requirements.txt                   # Python dependencies
├── run.py                             # Application entry point
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Multi-container setup
├── .env.example                       # Environment variables template
├── .gitignore
├── README.md
└── LICENSE
```

---

# 3. Database Design

## Table: `users`

Stores registered user accounts and authentication data.

| Field          | Type         | Constraints              | Description                  |
|----------------|--------------|--------------------------|------------------------------|
| id             | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique user ID               |
| username       | VARCHAR(80)  | NOT NULL UNIQUE           | Login username               |
| email          | VARCHAR(120) | NOT NULL UNIQUE           | Email address                |
| password_hash  | VARCHAR(256) | NOT NULL                  | bcrypt hashed password       |
| full_name      | VARCHAR(120) | NULLABLE                  | Display name                 |
| bio            | TEXT         | NULLABLE                  | Short user bio               |
| avatar_path    | VARCHAR(256) | NULLABLE                  | Profile picture path         |
| is_active      | BOOLEAN      | DEFAULT TRUE              | Account active status        |
| last_login     | DATETIME     | NULLABLE                  | Last login timestamp         |
| created_at     | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Account creation timestamp   |
| updated_at     | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Last update timestamp        |

## Table: `datasets`

Records metadata for every uploaded dataset.

| Field          | Type         | Constraints              | Description                  |
|----------------|--------------|--------------------------|------------------------------|
| id             | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique dataset ID            |
| user_id        | INTEGER      | FOREIGN KEY (users.id)   | Owner of dataset             |
| filename       | VARCHAR(255) | NOT NULL                  | Stored filename (UUID)       |
| original_name  | VARCHAR(255) | NOT NULL                  | Original filename            |
| file_size      | INTEGER      | NOT NULL                  | File size in bytes           |
| file_path      | VARCHAR(512) | NOT NULL                  | Full storage path            |
| file_type      | VARCHAR(50)  | DEFAULT 'csv'             | File extension               |
| row_count      | INTEGER      | NULLABLE                  | Number of rows               |
| column_count   | INTEGER      | NULLABLE                  | Number of columns            |
| columns_json   | TEXT         | NULLABLE                  | Column names & types (JSON)  |
| description    | TEXT         | NULLABLE                  | User-provided description    |
| forecast_type  | VARCHAR(50)  | NULLABLE                  | stock/gold/weather/land/sales/custom |
| date_column    | VARCHAR(255) | NULLABLE                  | Identified date column name  |
| target_column  | VARCHAR(255) | NULLABLE                  | Identified target column     |
| frequency      | VARCHAR(20)  | NULLABLE                  | D/W/M/Q/Y inferred frequency |
| status         | VARCHAR(20)  | DEFAULT 'uploaded'        | uploaded/validating/validated/processing/processed/forecasting/completed |
| uploaded_at    | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Upload timestamp             |
| updated_at     | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Last update timestamp        |

## Table: `validation_reports`

Stores data quality validation results for datasets.

| Field              | Type         | Constraints              | Description                  |
|--------------------|--------------|--------------------------|------------------------------|
| id                 | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique report ID             |
| dataset_id         | INTEGER      | FOREIGN KEY (datasets.id) | Associated dataset           |
| status             | VARCHAR(20)  | DEFAULT 'pending'         | pending/completed/failed     |
| overall_score      | FLOAT        | NULLABLE                  | Overall data quality score   |
| missing_values     | TEXT         | NULLABLE                  | Missing value summary (JSON) |
| outliers           | TEXT         | NULLABLE                  | Outlier detection (JSON)     |
| duplicates         | TEXT         | NULLABLE                  | Duplicate rows info (JSON)   |
| data_types         | TEXT         | NULLABLE                  | Column data types (JSON)     |
| unique_values      | TEXT         | NULLABLE                  | Unique value counts (JSON)   |
| basic_stats        | TEXT         | NULLABLE                  | Descriptive stats (JSON)     |
| issues_found       | TEXT         | NULLABLE                  | List of issues (JSON)        |
| warnings           | TEXT         | NULLABLE                  | List of warnings (JSON)      |
| recommendations    | TEXT         | NULLABLE                  | Recommended actions (JSON)   |
| created_at         | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Report generation timestamp  |

## Table: `eda_reports`

Stores exploratory data analysis results including statistical summaries and visualizations.

| Field              | Type         | Constraints              | Description                  |
|--------------------|--------------|--------------------------|------------------------------|
| id                 | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique EDA report ID         |
| dataset_id         | INTEGER      | FOREIGN KEY (datasets.id) | Associated dataset           |
| status             | VARCHAR(20)  | DEFAULT 'pending'         | pending/completed/failed     |
| descriptive_stats  | TEXT         | NULLABLE                  | Summary statistics (JSON)    |
| correlation_matrix | TEXT         | NULLABLE                  | Correlation data (JSON)      |
| distributions      | TEXT         | NULLABLE                  | Distribution data (JSON)     |
| time_series_plots  | TEXT         | NULLABLE                  | Plot paths/JSON              |
| seasonal_decomp    | TEXT         | NULLABLE                  | Seasonal decomposition (JSON)|
| stationarity_test  | TEXT         | NULLABLE                  | ADF test results (JSON)      |
| trend_analysis     | TEXT         | NULLABLE                  | Trend analysis (JSON)        |
| heatmap_path       | VARCHAR(512) | NULLABLE                  | Correlation heatmap path     |
| pairplot_path      | VARCHAR(512) | NULLABLE                  | Pairplot image path          |
| created_at         | DATETIME     | DEFAULT CURRENT_TIMESTAMP | EDA completion timestamp     |

## Table: `preprocessing_reports`

Tracks all preprocessing steps applied to a dataset.

| Field               | Type         | Constraints              | Description                  |
|---------------------|--------------|--------------------------|------------------------------|
| id                  | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique report ID             |
| dataset_id          | INTEGER      | FOREIGN KEY (datasets.id) | Associated dataset           |
| status              | VARCHAR(20)  | DEFAULT 'pending'         | pending/completed/failed     |
| missing_handled     | BOOLEAN      | DEFAULT FALSE             | Missing values handled       |
| missing_method      | VARCHAR(50)  | NULLABLE                  | drop/mean/median/mode/interpolate/ffill/bfill |
| outlier_handled     | BOOLEAN      | DEFAULT FALSE             | Outliers handled             |
| outlier_method      | VARCHAR(50)  | NULLABLE                  | iqr/zscore/isolation_forest  |
| scaling_applied     | BOOLEAN      | DEFAULT FALSE             | Scaling applied              |
| scaling_method      | VARCHAR(50)  | NULLABLE                  | standard/minmax/robust       |
| encoding_applied    | BOOLEAN      | DEFAULT FALSE             | Encoding applied             |
| encoding_method     | VARCHAR(50)  | NULLABLE                  | onehot/label                 |
| feature_engineering | BOOLEAN      | DEFAULT FALSE             | Features engineered          |
| lag_features        | TEXT         | NULLABLE                  | Lag column config (JSON)     |
| rolling_features    | TEXT         | NULLABLE                  | Rolling window config (JSON) |
| date_features       | TEXT         | NULLABLE                  | Date features extracted (JSON)|
| train_start         | DATETIME     | NULLABLE                  | Training data start          |
| train_end           | DATETIME     | NULLABLE                  | Training data end            |
| test_start          | DATETIME     | NULLABLE                  | Test data start              |
| test_end            | DATETIME     | NULLABLE                  | Test data end                |
| split_ratio         | FLOAT        | DEFAULT 0.2               | Train/test split ratio       |
| steps_log           | TEXT         | NULLABLE                  | All steps applied (JSON)     |
| preprocessed_path   | VARCHAR(512) | NULLABLE                  | Cleaned data file path       |
| created_at          | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Completion timestamp         |

## Table: `forecasting_models`

Stores metadata for trained forecasting model instances.

| Field         | Type         | Constraints              | Description                  |
|---------------|--------------|--------------------------|------------------------------|
| id            | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique model instance ID     |
| user_id       | INTEGER      | FOREIGN KEY (users.id)   | Model owner                  |
| dataset_id    | INTEGER      | FOREIGN KEY (datasets.id) | Source dataset               |
| model_name    | VARCHAR(100) | NOT NULL                  | Human-readable name          |
| model_type    | VARCHAR(50)  | NOT NULL                  | arima/sarima/prophet/xgboost/lstm/linear/ensemble |
| parameters    | TEXT         | NULLABLE                  | Model hyperparameters (JSON) |
| metrics       | TEXT         | NULLABLE                  | Evaluation metrics (JSON)    |
| model_path    | VARCHAR(512) | NULLABLE                  | Saved model file path        |
| training_time | FLOAT        | NULLABLE                  | Training duration (seconds)  |
| is_active     | BOOLEAN      | DEFAULT TRUE              | Soft delete flag             |
| version       | INTEGER      | DEFAULT 1                 | Model version number         |
| created_at    | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Training timestamp           |
| updated_at    | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Last update timestamp        |

## Table: `forecasting_results`

Stores prediction results and confidence intervals from trained models.

| Field             | Type         | Constraints                    | Description                    |
|-------------------|--------------|--------------------------------|--------------------------------|
| id                | INTEGER      | PRIMARY KEY AUTOINCREMENT      | Unique result ID               |
| model_id          | INTEGER      | FOREIGN KEY (forecasting_models.id) | Parent model instance         |
| forecast_horizon  | INTEGER      | NOT NULL                       | Number of future periods       |
| frequency         | VARCHAR(10)  | DEFAULT 'D'                    | D/W/M/Q/Y                      |
| predictions_json  | TEXT         | NOT NULL                       | Predicted values (JSON array)  |
| dates_json        | TEXT         | NOT NULL                       | Corresponding dates (JSON)     |
| confidence_lower  | TEXT         | NULLABLE                       | Lower bound (JSON array)       |
| confidence_upper  | TEXT         | NULLABLE                       | Upper bound (JSON array)       |
| metrics           | TEXT         | NULLABLE                       | Test set metrics (JSON)        |
| plot_path         | VARCHAR(512) | NULLABLE                       | Forecast plot image path       |
| created_at        | DATETIME     | DEFAULT CURRENT_TIMESTAMP      | Prediction timestamp           |

## Table: `generated_reports`

Tracks generated downloadable reports in various formats.

| Field       | Type         | Constraints              | Description                    |
|-------------|--------------|--------------------------|--------------------------------|
| id          | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique report ID               |
| user_id     | INTEGER      | FOREIGN KEY (users.id)   | Report owner                   |
| dataset_id  | INTEGER      | FOREIGN KEY (datasets.id) | Associated dataset             |
| model_id    | INTEGER      | FOREIGN KEY (forecasting_models.id) | Associated model (nullable)|
| type        | VARCHAR(50)  | NOT NULL                  | validation/eda/preprocessing/forecast/comparison |
| format      | VARCHAR(10)  | DEFAULT 'pdf'             | pdf/html/csv/json              |
| title       | VARCHAR(255) | NULLABLE                  | Report title                   |
| file_path   | VARCHAR(512) | NOT NULL                  | Report file path               |
| file_size   | INTEGER      | NULLABLE                  | Report file size               |
| created_at  | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Generation timestamp           |

## Table: `activity_logs`

Audit trail for all user actions within the platform.

| Field       | Type         | Constraints              | Description                    |
|-------------|--------------|--------------------------|--------------------------------|
| id          | INTEGER      | PRIMARY KEY AUTOINCREMENT | Unique log ID                  |
| user_id     | INTEGER      | FOREIGN KEY (users.id)   | User who performed action      |
| action      | VARCHAR(100) | NOT NULL                  | Action performed               |
| entity_type | VARCHAR(50)  | NULLABLE                  | users/datasets/models/reports  |
| entity_id   | INTEGER      | NULLABLE                  | ID of affected entity          |
| details     | TEXT         | NULLABLE                  | Additional context (JSON)      |
| ip_address  | VARCHAR(45)  | NULLABLE                  | Request IP address             |
| timestamp   | DATETIME     | DEFAULT CURRENT_TIMESTAMP | Action timestamp               |

---

# 4. Frontend Page Flow

## Page Navigation Map

```
                    ┌─────────────────┐
                    │   Landing Page  │
                    │   (index.html)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Get Started   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐
      │   Login    │ │  Register  │ │  (Guest)   │
      └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
             │              │              │
             └──────┬───────┘              │
                    ▼                      │
          ┌─────────────────┐             │
          │   Dashboard     │◄────────────┘
          │ (dashboard.html)│
          └────────┬────────┘
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Upload   │ │ Profile  │ │ View Previous│
│ Dataset  │ │(profile) │ │ Datasets     │
└────┬─────┘ └──────────┘ └──────┬───────┘
     │                            │
     ▼                            │
┌──────────┐                     │
│Validation│                     │
│(validation)                    │
└────┬─────┘                     │
     │                           │
     ▼                           │
┌──────────┐                     │
│   EDA    │                     │
│  (eda)   │                     │
└────┬─────┘                     │
     │                           │
     ▼                           │
┌──────────────┐                 │
│Preprocessing │                 │
│(preprocessing)                 │
└──────┬───────┘                 │
       │                         │
       ▼                         │
┌──────────────┐                 │
│ Forecasting  │                 │
│ (forecast)   │                 │
└──────┬───────┘                 │
       │                         │
       ▼                         │
┌──────────────┐                 │
│   Model      │                 │
│  Comparison  │                 │
│ (comparison) │                 │
└──────┬───────┘                 │
       │                         │
       ▼                         ▼
┌──────────────┐          ┌──────────┐
│   Reports    │          │ Dashboard│
│  (reports)   │          │ (back)   │
└──────────────┘
```

## Detailed Page Descriptions

| Page             | URL              | Purpose                                              |
|------------------|------------------|------------------------------------------------------|
| Landing          | `/`              | Hero section, features overview, tech stack, CTA     |
| Login            | `/login`         | Email/password authentication form                   |
| Register         | `/register`      | New user registration form                           |
| Dashboard        | `/dashboard`     | Stats cards, recent datasets, quick actions, charts  |
| Upload Dataset   | `/upload`        | Drag-and-drop file upload, preview table, metadata   |
| Validation       | `/validate/<id>` | Data quality score, missing values, outliers, issues |
| EDA Dashboard    | `/eda/<id>`      | Statistical tables, correlation heatmap, plots       |
| Preprocessing    | `/preprocess/<id>` | Missing/outlier/scaling controls, train/test split |
| Forecasting      | `/forecast/<id>` | Model selection, params, run forecast, view results  |
| Model Comparison | `/compare/<id>`  | Side-by-side metrics, overlaid forecast plots        |
| Reports          | `/reports`       | List of generated reports, download buttons          |
| User Profile     | `/profile`       | Edit name, email, password, avatar                   |

---

# 5. Backend API Design

## 5.1 Authentication APIs

| Endpoint              | Method | Request Body                                          | Response Body                                                  |
|-----------------------|--------|------------------------------------------------------|----------------------------------------------------------------|
| `/api/auth/register`  | POST   | `{ "username", "email", "password", "full_name" }`   | `{ "success": true, "message": "...", "user": { id, username, email } }` |
| `/api/auth/login`     | POST   | `{ "email", "password" }`                            | `{ "success": true, "message": "...", "user": { id, username, email, full_name }, "session_token": "..." }` |
| `/api/auth/logout`    | POST   | (session cookie)                                     | `{ "success": true, "message": "Logged out" }`                 |
| `/api/auth/profile`   | GET    | (session cookie)                                     | `{ "success": true, "user": { id, username, email, full_name, bio, created_at } }` |
| `/api/auth/profile`   | PUT    | `{ "full_name", "bio", "email" }`                   | `{ "success": true, "message": "Profile updated", "user": {...} }` |
| `/api/auth/password`  | PUT    | `{ "current_password", "new_password" }`             | `{ "success": true, "message": "Password changed" }`           |

## 5.2 Dataset APIs

| Endpoint                     | Method | Request Body                                         | Response Body                                                  |
|------------------------------|--------|------------------------------------------------------|----------------------------------------------------------------|
| `/api/datasets/upload`       | POST   | `multipart/form-data: file + description + forecast_type` | `{ "success": true, "dataset": { id, filename, row_count, column_count, ... } }` |
| `/api/datasets`              | GET    | Query: `?page=1&per_page=10`                        | `{ "success": true, "datasets": [...], "total": N, "page": 1 }` |
| `/api/datasets/<id>`         | GET    | -                                                    | `{ "success": true, "dataset": { ... } }`                      |
| `/api/datasets/<id>/preview` | GET    | Query: `?rows=10`                                   | `{ "success": true, "columns": [...], "data": [[...], ...], "dtypes": {...} }` |
| `/api/datasets/<id>`         | DELETE | (session cookie)                                    | `{ "success": true, "message": "Dataset deleted" }`            |

## 5.3 Validation APIs

| Endpoint                          | Method | Request Body | Response Body                                                |
|-----------------------------------|--------|-------------|--------------------------------------------------------------|
| `/api/validation/<dataset_id>`    | POST   | -           | `{ "success": true, "report": { id, status: "processing" } }` |
| `/api/validation/<dataset_id>/status` | GET | -         | `{ "success": true, "status": "completed", "progress": 100 }` |
| `/api/validation/<dataset_id>/report` | GET | -         | `{ "success": true, "report": { overall_score, missing_values, outliers, duplicates, issues, recommendations } }` |

## 5.4 EDA APIs

| Endpoint                     | Method | Request Body | Response Body                                                |
|------------------------------|--------|-------------|--------------------------------------------------------------|
| `/api/eda/<dataset_id>`      | POST   | -           | `{ "success": true, "report": { id, status: "processing" } }` |
| `/api/eda/<dataset_id>/status` | GET  | -           | `{ "success": true, "status": "completed" }`                 |
| `/api/eda/<dataset_id>/report` | GET  | -           | `{ "success": true, "report": { descriptive_stats, correlation_matrix, distributions, stationarity_test, seasonal_decomp } }` |
| `/api/eda/<dataset_id>/plots` | GET   | -           | `{ "success": true, "plots": { heatmap, pairplot, distributions: [...], time_series: [...] } }` |

## 5.5 Preprocessing APIs

| Endpoint                             | Method | Request Body                                                 | Response Body                                                |
|--------------------------------------|--------|-------------------------------------------------------------|--------------------------------------------------------------|
| `/api/preprocessing/<dataset_id>`    | POST   | `{ "missing": { "method": "mean" }, "outlier": { "method": "iqr" }, "scaling": { "method": "standard" }, "encoding": { "method": "onehot" }, "split": { "ratio": 0.2 }, "features": { "lags": [1,7,30], "rolling": [7,30], "date_features": true } }` | `{ "success": true, "report": { id, status: "processing" } }` |
| `/api/preprocessing/<dataset_id>/status` | GET | - | `{ "success": true, "status": "completed" }`                 |
| `/api/preprocessing/<dataset_id>/report` | GET | - | `{ "success": true, "report": { steps_log, preprocessed_path, train_size, test_size } }` |
| `/api/preprocessing/<dataset_id>/data` | GET | - | `{ "success": true, "train": {...}, "test": {...} }`         |

## 5.6 Forecast APIs

| Endpoint                                | Method | Request Body                                                 | Response Body                                                |
|-----------------------------------------|--------|-------------------------------------------------------------|--------------------------------------------------------------|
| `/api/forecast/models`                  | GET    | -                                                           | `{ "success": true, "models": ["arima", "sarima", "prophet", "xgboost", "lstm", "linear", "ensemble"] }` |
| `/api/forecast/<dataset_id>/train`      | POST   | `{ "model_type": "prophet", "parameters": {...}, "horizon": 30 }` | `{ "success": true, "model_id": N, "status": "training" }`   |
| `/api/forecast/<dataset_id>/status/<model_id>` | GET | -                                                | `{ "success": true, "status": "completed", "progress": 100 }` |
| `/api/forecast/<dataset_id>/results/<model_id>` | GET | - | `{ "success": true, "result": { predictions, dates, confidence_lower, confidence_upper, metrics }, "plot_url": "..." }` |
| `/api/forecast/<dataset_id>/models`     | GET    | -                                                           | `{ "success": true, "models": [{ id, model_type, metrics, created_at }, ...] }` |
| `/api/forecast/<dataset_id>/hyperparameters/<model_type>` | GET | - | `{ "success": true, "default_params": {...}, "search_space": {...} }` |

## 5.7 Comparison APIs

| Endpoint                            | Method | Request Body                                                 | Response Body                                                |
|-------------------------------------|--------|-------------------------------------------------------------|--------------------------------------------------------------|
| `/api/comparison/<dataset_id>`      | POST   | `{ "model_ids": [1, 2, 3] }`                                | `{ "success": true, "comparison": { metrics_table: {...}, best_model: "...", rankings: [...] } }` |
| `/api/comparison/<dataset_id>/plot` | POST   | `{ "model_ids": [1, 2, 3] }`                                | `{ "success": true, "plot_url": "..." }`                     |

## 5.8 Report APIs

| Endpoint                           | Method | Request Body                        | Response Body                                                |
|------------------------------------|--------|------------------------------------|--------------------------------------------------------------|
| `/api/reports/generate`            | POST   | `{ "dataset_id": N, "type": "forecast", "model_id": N, "format": "pdf" }` | `{ "success": true, "report": { id, file_path, download_url } }` |
| `/api/reports`                     | GET    | Query: `?page=1&per_page=10`      | `{ "success": true, "reports": [...], "total": N }`          |
| `/api/reports/<id>`                | GET    | -                                  | `{ "success": true, "report": { ... } }`                     |
| `/api/reports/<id>/download`       | GET    | -                                  | (Binary file download with Content-Disposition header)       |
| `/api/reports/<id>`                | DELETE | -                                  | `{ "success": true, "message": "Report deleted" }`           |

---

# 6. Complete User Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE USER WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: REGISTRATION                                                   │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /register → Fill Form → Submit                       │         │
│  │ Backend: Hash password → Create user → Set session          │         │
│  │ Response: Redirect to /dashboard                            │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 2: LOGIN                                                         │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /login → Enter credentials → Submit                  │         │
│  │ Backend: Verify password → Create session → Log activity    │         │
│  │ Response: Redirect to /dashboard                            │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 3: DASHBOARD OVERVIEW                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /dashboard → View stats: datasets count, models     │         │
│  │            trained, reports generated, recent activity     │         │
│  │ Action: Click "Upload New Dataset"                         │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 4: UPLOAD DATASET                                                 │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /upload → Drag-and-drop CSV → Set forecast type      │         │
│  │         → Add description → Submit                         │         │
│  │ Backend: Validate file → Save to /data/uploads →           │         │
│  │          Parse columns → Detect date/target → Create       │         │
│  │          dataset record → Redirect to /validate/<id>       │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 5: DATA VALIDATION                                                │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /validate/<id> → View auto-running validation       │         │
│  │ Backend: Check missing values, outliers, duplicates,       │         │
│  │          data types, basic statistics → Generate report    │         │
│  │ User: Review quality score, issues, recommendations        │         │
│  │ Action: Click "Proceed to EDA"                             │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 6: EXPLORATORY DATA ANALYSIS                                      │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /eda/<id> → View comprehensive analysis              │         │
│  │ Backend: Descriptive stats, correlation matrix,            │         │
│  │          distribution plots, time series plots,            │         │
│  │          seasonal decomposition, stationarity test         │         │
│  │ User: Explore interactive Plotly charts                    │         │
│  │ Action: Click "Proceed to Preprocessing"                   │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 7: DATA PREPROCESSING                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /preprocess/<id> → Configure steps:                 │         │
│  │   □ Handle Missing: [drop / mean / median / mode /         │         │
│  │                      interpolate / ffill / bfill]          │         │
│  │   □ Handle Outliers: [IQR / Z-Score / Isolation Forest]     │         │
│  │   □ Scaling: [Standard / MinMax / Robust / None]           │         │
│  │   □ Encoding: [One-Hot / Label / None]                     │         │
│  │   □ Feature Engineering: [Lags, Rolling Windows,           │         │
│  │                           Date Features]                   │         │
│  │   □ Train/Test Split: [Slider 0.1-0.4]                    │         │
│  │ User: Click "Apply Preprocessing" → View summary log       │         │
│  │ Action: Click "Proceed to Forecasting"                     │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 8: FORECASTING                                                    │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /forecast/<id> → Select Model(s):                   │         │
│  │   □ ARIMA/SARIMA □ Prophet □ XGBoost                      │         │
│  │   □ LSTM □ Linear Regression □ Ensemble                   │         │
│  │ → Configure Hyperparameters → Set Forecast Horizon         │         │
│  │ → Click "Run Forecast"                                     │         │
│  │ Backend: Train model(s) → Evaluate → Predict future        │         │
│  │ User: View predictions table + interactive forecast plot    │         │
│  │   with confidence intervals                                │         │
│  │ Action: Click "Compare Models" (if multiple) OR            │         │
│  │          Click "Generate Report"                           │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 9: MODEL COMPARISON                                               │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /compare/<id> → Select 2+ trained models            │         │
│  │ View: Side-by-side metrics table (MAE, RMSE, MAPE, R²)     │         │
│  │ View: Overlaid forecast plot                                │         │
│  │ View: Model ranking by selected metric                     │         │
│  │ Action: Click "Generate Comparison Report"                 │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                          │
│  STEP 10: REPORT GENERATION & DOWNLOAD                                  │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │ User → /reports → View all generated reports               │         │
│  │ Backend: Generate PDF/HTML report with:                    │         │
│  │   - Dataset info                                           │         │
│  │   - Validation summary                                     │         │
│  │   - EDA highlights                                         │         │
│  │   - Preprocessing steps                                    │         │
│  │   - Model details & metrics                                │         │
│  │   - Forecast plots & predictions                           │         │
│  │   - Comparison results                                     │         │
│  │ User: Click "Download" → Get PDF/HTML/CSV report           │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 7. Forecasting Workflow Design

## End-to-End Forecasting Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FORECASTING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RAW DATASET                                                             │
│  (CSV from user)                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 1: VALIDATION                          │    │
│  │  • Check file format & encoding                                 │    │
│  │  • Identify date column (auto-detect)                           │    │
│  │  • Identify target column (auto-detect)                         │    │
│  │  • Check for missing values                                     │    │
│  │  • Check for outliers (IQR/Z-score)                             │    │
│  │  • Check for duplicates                                         │    │
│  │  • Validate data types                                          │    │
│  │  • Generate quality score & recommendations                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 2: EDA                                  │    │
│  │  • Descriptive statistics (mean, median, std, min, max)         │    │
│  │  • Time series plot (full range)                                │    │
│  │  • Distribution analysis (histogram, box plot)                  │    │
│  │  • Correlation analysis (heatmap)                               │    │
│  │  • Seasonal decomposition (trend, seasonal, residual)           │    │
│  │  • Stationarity test (ADF test)                                 │    │
│  │  • Autocorrelation (ACF/PACF plots)                             │    │
│  │  • Trend analysis & pattern detection                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 3: PREPROCESSING                       │    │
│  │  • Handle missing values (configurable method)                  │    │
│  │  • Handle outliers (configurable method)                        │    │
│  │  • Remove duplicates                                            │    │
│  │  • Scale/normalize features                                     │    │
│  │  • Encode categorical variables                                 │    │
│  │  • Sort by date, set date as index                              │    │
│  │  • Resample to consistent frequency                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 4: FEATURE ENGINEERING                 │    │
│  │  • Lag features (t-1, t-7, t-30, etc.)                         │    │
│  │  • Rolling window statistics (mean, std over 7/30 days)        │    │
│  │  • Date/time features (day of week, month, quarter, year)       │    │
│  │  • Holiday features (weekend flag, holiday flag)                │    │
│  │  • Differencing (first order, seasonal)                         │    │
│  │  • Fourier features (seasonal patterns)                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 5: TRAIN/TEST SPLIT                    │    │
│  │  • Time-series aware split (no shuffling)                      │    │
│  │  • Configurable ratio (default: 80/20)                         │    │
│  │  • Walk-forward validation for time series CV                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 6: MODEL SELECTION                     │    │
│  │                                                                    │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐  │    │
│  │  │  ARIMA/  │ │ Prophet  │ │ XGBoost  │ │   LSTM   │ │Linear │  │    │
│  │  │  SARIMA  │ │          │ │          │ │ (Keras)  │ │  Reg   │  │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬───┘  │    │
│  │       │            │            │            │            │       │    │
│  │       └────────────┴────────────┴────────────┴────────────┘       │    │
│  │                         │                                          │    │
│  │                   ┌─────▼──────┐                                   │    │
│  │                   │  Ensemble  │  (Weighted average of above)      │    │
│  │                   └────────────┘                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              STEP 7: HYPERPARAMETER TUNING                     │    │
│  │  • ARIMA: Auto (p,d,q) search via AIC/BIC                      │    │
│  │  • Prophet: changepoint_prior, seasonality_prior, etc.          │    │
│  │  • XGBoost: n_estimators, max_depth, learning_rate, subsample  │    │
│  │  • LSTM: units, layers, dropout, epochs, batch_size            │    │
│  │  • Grid search / Random search / Bayesian optimization         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              STEP 8: MODEL TRAINING                             │    │
│  │  • Fit model on training data                                   │    │
│  │  • Time-series cross-validation (expanding window)              │    │
│  │  • Track training time & loss curves                            │    │
│  │  • Save trained model to /data/models/                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              STEP 9: EVALUATION                                 │    │
│  │  • Predict on test set                                           │    │
│  │  • Calculate metrics:                                            │    │
│  │    - MAE (Mean Absolute Error)                                   │    │
│  │    - MSE (Mean Squared Error)                                    │    │
│  │    - RMSE (Root Mean Squared Error)                              │    │
│  │    - MAPE (Mean Absolute Percentage Error)                       │    │
│  │    - SMAPE (Symmetric MAPE)                                     │    │
│  │    - R² (Coefficient of Determination)                           │    │
│  │  • Plot actual vs predicted                                      │    │
│  │  • Residual analysis                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              STEP 10: FUTURE PREDICTION                        │    │
│  │  • Generate predictions for forecast_horizon periods ahead      │    │
│  │  • Calculate confidence intervals (where supported)             │    │
│  │  • Generate forecast plot (historical + future)                 │    │
│  │  • Return predictions as JSON + plot image                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              STEP 11: REPORT GENERATION                        │    │
│  │  • Compile all results into structured report                   │    │
│  │  • Generate PDF via WeasyPrint or ReportLab                     │    │
│  │  • Include: dataset info, validation, EDA, preprocessing,       │    │
│  │    model details, metrics, plots, predictions                   │    │
│  │  • Save to /data/reports/                                       │    │
│  │  • Provide download URL                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 8. Security Architecture

## 8.1 Password Hashing

| Component          | Implementation                                |
|--------------------|-----------------------------------------------|
| Algorithm          | Werkzeug `generate_password_hash` (pbkdf2:sha256) |
| Salt               | Automatic per-password salt                   |
| Verification       | `check_password_hash`                         |
| Storage            | `password_hash` column in `users` table       |
| Min Length         | 8 characters enforced at registration         |

## 8.2 Session Management

| Component          | Implementation                                |
|--------------------|-----------------------------------------------|
| Storage            | Flask session (signed cookies)                |
| Secret Key         | 32-byte random key from environment variable  |
| Session Type       | Server-side (filesystem) recommended          |
| Session TTL        | 24 hours, sliding expiration                  |
| Decorator          | `@login_required` checks session user_id      |
| Logout             | Clears session and removes cookie             |

## 8.3 Secure File Upload

| Area               | Implementation                                |
|--------------------|-----------------------------------------------|
| Allowed Extensions | `.csv`, `.xlsx`, `.json`, `.parquet`          |
| Max File Size      | 50 MB (configurable in config.py)             |
| File Renaming      | UUID-based filename prevents path traversal   |
| Storage Path       | `/data/uploads/<uuid>_<original_name>`        |
| MIME Validation    | Check `Content-Type` header + magic bytes     |
| CSV Injection      | Sanitize cells starting with `=`, `+`, `-`, `@` |

## 8.4 Input Validation

| Area               | Implementation                                |
|--------------------|-----------------------------------------------|
| Backend            | Marshmallow or custom schema validation       |
| SQL Injection      | SQLAlchemy ORM (parameterized queries)        |
| XSS Prevention     | Jinja2 auto-escaping in templates             |
| CSRF Protection    | Flask-WTF CSRF tokens on forms                |
| API Validation     | JSON schema validation for all endpoints      |
| Sanitization       | Strip HTML tags from string inputs            |

## 8.5 SQL Injection Prevention

| Measure            | Implementation                                |
|--------------------|-----------------------------------------------|
| ORM Usage          | SQLAlchemy exclusively (no raw SQL)           |
| Parameterized Q    | All queries use bound parameters              |
| Migrations         | Alembic for schema changes (no raw SQL)       |
| Input Escaping     | ORM handles escaping automatically            |

## 8.6 Additional Security Measures

| Measure                 | Implementation                                    |
|-------------------------|---------------------------------------------------|
| CORS                    | Flask-CORS with restricted origins                |
| Rate Limiting           | Flask-Limiter on auth endpoints (5/min)           |
| HTTPS Enforcement       | Talisman for production                           |
| Error Handling          | No stack traces in production                     |
| Logging                 | All auth events logged to activity_logs           |
| Environment Variables   | `.env` file for secrets (not committed)           |
| Dependency Scanning     | Regular `pip-audit` checks                        |

---

# 9. Scalability Planning

## 9.1 SQLite → PostgreSQL Migration

| Aspect              | SQLite (Phase 1)              | PostgreSQL (Phase 2+)          |
|---------------------|-------------------------------|--------------------------------|
| Concurrency         | Single-writer                 | Multi-writer with MVCC         |
| Storage             | Single file                   | Managed server                 |
| Performance         | Good for small datasets       | Handles millions of rows       |
| JSON Support        | Limited                       | Full JSONB                     |
| Full-Text Search    | No                            | Built-in                       |
| Migration Path      | -                             | Use `pgloader` or Alembic      |
| Connection Pool     | Not needed                    | SQLAlchemy pool_size=10        |
| Type Changes        | -                             | Use Alembic migration scripts  |

**Migration strategy**: Use SQLAlchemy's abstraction layer — code changes are minimal. Only the `SQLALCHEMY_DATABASE_URI` config changes.

## 9.2 Flask → FastAPI Migration

| Aspect              | Flask (Phase 1)               | FastAPI (Phase 2+)             |
|---------------------|-------------------------------|--------------------------------|
| Performance         | Synchronous (WSGI)            | Async (ASGI)                   |
| Validation          | Manual / Marshmallow          | Pydantic (auto)                |
| Documentation       | Manual                        | Auto-generated OpenAPI/Swagger |
| Type Hints          | Optional                      | Required (built-in)            |
| WebSocket           | Limited                       | Native support                 |
| Background Tasks    | Thread pool                   | Asyncio tasks                  |
| Dependency Injection| Manual                        | Built-in                       |
| Migration Path      | -                             | Wrap route logic in services   |

**Migration strategy**: Service layer is already decoupled from routes. Create new FastAPI route files referencing the same services.

## 9.3 Local → Cloud Deployment

| Aspect              | Local (Phase 1)               | Cloud (Phase 2+)               |
|---------------------|-------------------------------|--------------------------------|
| Hosting             | `python run.py`               | AWS EC2 / GCP Compute / Azure  |
| Database            | SQLite file                   | AWS RDS / Cloud SQL            |
| File Storage        | Local filesystem              | AWS S3 / GCP Cloud Storage     |
| ML Inference        | Synchronous                   | Async + Queue (Celery + Redis) |
| Containerization    | -                             | Docker + ECS/GKE/AKS           |
| CI/CD               | -                             | GitHub Actions + Docker Hub    |
| Monitoring          | -                             | CloudWatch / Stackdriver       |
| Domain & SSL        | localhost                     | Route53 + Cert Manager         |
| CDN                 | -                             | CloudFront / Cloud CDN         |

### Cloud Architecture (Phase 2+)

```
                         ┌─────────────┐
                         │  CloudFront  │
                         │    (CDN)     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Load       │
                         │  Balancer   │
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
        ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
        │  Flask/    │   │   Flask/    │   │   Flask/    │
        │  FastAPI   │   │   FastAPI   │   │   FastAPI   │
        │  Instance 1│   │  Instance 2│   │  Instance 3│
        └─────┬─────┘   └──────┬──────┘   └──────┬──────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Redis Queue      │
                    │  (Celery workers)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     PostgreSQL        │
                    │     (RDS/Aurora)      │
                    └───────────────────────┘
```

---

# 10. Deliverables

## 10.1 System Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FORECASTIQ SYSTEM ARCHITECTURE                 │
│                     Universal Forecasting Platform                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    PRESENTATION LAYER                           │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │ │
│  │  │  HTML5    │ │  CSS3    │ │  Bootstrap│ │  JavaScript      │  │ │
│  │  │  Pages    │ │  Custom  │ │    5      │ │  + Fetch API     │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │           Plotly.js (Interactive Charts)                 │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │  REST API (JSON)                     │
│                                ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    APPLICATION LAYER (Flask)                    │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │ │
│  │  │Auth Routes │ │Dataset     │ │Validation  │ │EDA Routes  │  │ │
│  │  │            │ │Routes      │ │Routes      │ │            │  │ │
│  │  ├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤  │ │
│  │  │Preprocess  │ │Forecast    │ │Comparison  │ │Report      │  │ │
│  │  │Routes      │ │Routes      │ │Routes      │ │Routes      │  │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │              SERVICE LAYER (Business Logic)              │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │                                      │
│                                ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              FORECASTING ENGINE LAYER                          │ │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌─────┐  │ │
│  │  │ARIMA  │ │SARIMA │ │Prophet│ │XGBoost│ │ LSTM   │ │Linear│  │ │
│  │  └───────┘ └───────┘ └───────┘ └───────┘ └────────┘ └─────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │              Model Factory + Metrics                     │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │                                      │
│                                ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    DATA LAYER                                   │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │              SQLAlchemy ORM + SQLite                     │  │ │
│  │  │  users │ datasets │ validation_reports │ eda_reports     │  │ │
│  │  │  preprocessing_reports │ forecasting_models              │  │ │
│  │  │  forecasting_results │ generated_reports │ activity_logs │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │ │
│  │  │ /data/uploads│ │/data/reports │ │ /data/models         │  │ │
│  │  │ (CSV files)  │ │ (PDF files)  │ │ (Pickle/H5 files)    │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 10.2 Development Roadmap

### Phase 1 (MVP) — 6-8 Weeks

| Week | Milestone                          | Deliverables                                        |
|------|------------------------------------|-----------------------------------------------------|
| 1    | Project Setup & Architecture       | Flask app factory, config, models, database, folder structure |
| 2    | Authentication System              | Register, login, logout, session management, profile |
| 3    | Dataset Management                 | Upload, preview, list, delete datasets              |
| 4    | Validation & EDA Modules           | Data quality checks, descriptive stats, plots, decomposition |
| 5    | Preprocessing Module               | Missing values, outliers, scaling, encoding, train/test split |
| 6    | Forecasting Engine                 | ARIMA, Prophet, XGBoost — training, evaluation, prediction |
| 7    | Model Comparison & Reports         | Side-by-side comparison, PDF report generation      |
| 8    | Frontend Integration & Testing     | Connect all pages, end-to-end tests, bug fixes      |

### Phase 2 (Enhanced) — 4-6 Weeks

| Week | Milestone                          | Deliverables                                        |
|------|------------------------------------|-----------------------------------------------------|
| 9    | LSTM & Ensemble Models             | Deep learning models, ensemble averaging            |
| 10   | Hyperparameter Tuning              | Grid search, Auto-ARIMA, Bayesian optimization      |
| 11   | Advanced Feature Engineering       | Fourier features, holiday effects, external regressors |
| 12   | Performance Optimization           | Caching, async tasks, database indexing             |

### Phase 3 (Production) — 4-6 Weeks

| Week | Milestone                          | Deliverables                                        |
|------|------------------------------------|-----------------------------------------------------|
| 13   | PostgreSQL Migration               | Alembic migrations, connection pooling              |
| 14   | Docker Containerization            | Dockerfile, docker-compose, multi-stage builds      |
| 15   | Cloud Deployment                   | AWS/GCP setup, S3 storage, RDS database             |
| 16   | CI/CD Pipeline & Monitoring        | GitHub Actions, automated tests, CloudWatch         |

## 10.3 Technology Stack Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FRONTEND                     BACKEND                                │
│  ├── HTML5                    ├── Python 3.11+                      │
│  ├── CSS3                     ├── Flask 3.x                         │
│  ├── JavaScript (ES6+)        ├── Flask-SQLAlchemy                  │
│  ├── Bootstrap 5.3            ├── Flask-Login                       │
│  ├── Plotly.js                ├── Flask-WTF / Flask-CORS            │
│  ├── Font Awesome 6           ├── Gunicorn (production)             │
│  └── AOS (scroll animations)  └── Waitress (Windows production)     │
│                                                                      │
│  ML & DATA SCIENCE            DATABASE                               │
│  ├── Pandas / NumPy           ├── SQLite (Phase 1)                  │
│  ├── Scikit-learn             └── PostgreSQL (Phase 2+)             │
│  ├── Statsmodels                                                    │
│  ├── Prophet                   DEVOPS                                │
│  ├── XGBoost                   ├── Docker + Docker Compose          │
│  ├── TensorFlow / Keras        ├── Git + GitHub                     │
│  ├── Matplotlib / Seaborn      ├── GitHub Actions (CI/CD)           │
│  └── Plotly (Python)           └── AWS/GCP (Phase 3)                │
│                                                                      │
│  REPORTING                                                           │
│  ├── WeasyPrint (PDF generation)                                    │
│  ├── Jinja2 (HTML templating)                                       │
│  └── JSON/CSV export                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

*End of Phase 1 Architecture Document — ForecastIQ v1.0*
