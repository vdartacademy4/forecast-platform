from app import create_app

app = create_app()

if __name__ == '__main__':
    print('=' * 60)
    print('  ForecastIQ - Universal Forecasting Platform')
    print('  Server starting at http://127.0.0.1:5000')
    print('=' * 60)
    app.run(debug=True, host='127.0.0.1', port=5000)
