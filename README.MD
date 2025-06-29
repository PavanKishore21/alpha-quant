# ALPHA-QUANT

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ALPHA-QUANT** is a sophisticated quantitative trading strategy backtesting system. It leverages machine learning to rank securities and employs an advanced risk management framework to construct optimized portfolios. The system is delivered through a modern web interface built with Flask, providing dynamic visualizations and performance analytics.

---

### Key Features

* **ML-Powered Stock Ranking:** Utilizes a combination of Ridge and MLP Regressor models to predict future returns based on technical indicators like RSI, MACD, and Momentum.
* **Advanced Risk Management:** Implements a dynamic weighting strategy based on:
    * **Inverse Volatility:** Assigns higher weights to more stable stocks.
    * **Model Confidence:** Allocates more capital to stocks with higher prediction scores.
    * **Correlation Penalty:** Diversifies the portfolio by reducing weights of highly correlated assets.
* **Comprehensive Backtesting:** Run historical simulations with customizable parameters like date ranges, stock universe, and transaction costs.
* **In-Depth Performance Analytics:** Visualize portfolio performance against a benchmark (NIFTY 50) and analyze key metrics, including:
    * Sharpe, Sortino, and Calmar Ratios
    * Alpha and Beta
    * Max Drawdown and Annualized Volatility
* **Interactive Web Interface:** A clean, professional dashboard built with Flask, Plotly.js, and Chart.js to configure backtests and analyze results.

### Tech Stack

* **Backend:** Python, Flask
* **Data Analysis & ML:** Pandas, NumPy, Scikit-learn, yfinance
* **Frontend:** HTML, CSS, JavaScript, Bootstrap, Plotly.js, Chart.js
* **Production Server:** Gunicorn

---

### Local Setup Instructions

Follow these steps to run the application on your local machine.

1.  **Prerequisites:**
    * Python 3.10 or newer.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ALPHA-QUANT
    ```

3.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it (on Mac/Linux)
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

---

### Deployment

This application is designed for cloud deployment on platforms like Render or PythonAnywhere. For detailed instructions, refer to the official platform documentation. The key deployment requirements are:

* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn main:app`

---

### License

This project is licensed under the MIT License.