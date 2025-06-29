# main.py - Hybrid Factor & ML Quantitative Trading System
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import traceback
import yfinance as yf
from functools import lru_cache
import gc
import os

warnings.filterwarnings('ignore')

# ML/Stats imports
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Web framework
from flask import Flask, render_template, request, jsonify

# --- FIX: Re-introducing Technical Indicator Functions ---
@lru_cache(maxsize=512)
def calculate_rsi(prices_tuple, period=14):
    prices = pd.Series(prices_tuple, dtype=float)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-6)
    return 100 - (100 / (1 + rs))

@lru_cache(maxsize=512)
def calculate_macd(prices_tuple, fast=12, slow=26):
    prices = pd.Series(prices_tuple, dtype=float)
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2
# --- END FIX ---


# --- Factor Ranking System ---
class FactorRankingSystem:
    def __init__(self, fundamental_data):
        self.fundamental_data = fundamental_data
        self.factor_weights = {'momentum': 0.4, 'value': 0.2, 'quality': 0.3, 'low_volatility': 0.1}

    def _calculate_momentum_scores(self, price_data):
        returns_3m = price_data.pct_change(63).iloc[-1]
        returns_6m = price_data.pct_change(126).iloc[-1]
        returns_12m = price_data.pct_change(252).iloc[-1]
        return (returns_3m.rank(pct=True) + returns_6m.rank(pct=True) + returns_12m.rank(pct=True)) / 3

    def _calculate_value_scores(self):
        pe_ratios = pd.Series({ticker: data.get('pe_ratio') for ticker, data in self.fundamental_data.items()})
        pb_ratios = pd.Series({ticker: data.get('pb_ratio') for ticker, data in self.fundamental_data.items()})
        pe_rank = (1 / pe_ratios.fillna(1e6)).rank(pct=True)
        pb_rank = (1 / pb_ratios.fillna(1e6)).rank(pct=True)
        return (pe_rank + pb_rank) / 2

    def _calculate_quality_scores(self):
        roe = pd.Series({ticker: data.get('roe') for ticker, data in self.fundamental_data.items()})
        return roe.fillna(-1e6).rank(pct=True)

    def _calculate_low_vol_scores(self, price_data):
        returns = price_data.pct_change()
        volatility = returns.rolling(252).std().iloc[-1] * np.sqrt(252)
        return (1 / volatility.fillna(1e6)).rank(pct=True)

    def get_composite_scores(self, price_data):
        momentum_scores = self._calculate_momentum_scores(price_data)
        value_scores = self._calculate_value_scores()
        quality_scores = self._calculate_quality_scores()
        low_vol_scores = self._calculate_low_vol_scores(price_data)
        
        all_stocks = price_data.columns
        composite_scores = (
            momentum_scores.reindex(all_stocks).fillna(0) * self.factor_weights['momentum'] +
            value_scores.reindex(all_stocks).fillna(0) * self.factor_weights['value'] +
            quality_scores.reindex(all_stocks).fillna(0) * self.factor_weights['quality'] +
            low_vol_scores.reindex(all_stocks).fillna(0) * self.factor_weights['low_volatility']
        )
        return composite_scores

# --- Lightweight ML Ranking System ---
class OptimizedMLRankingSystem:
    def __init__(self):
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_regression, k=4)
        self.is_trained = False
        self.feature_names = ['momentum_1m', 'volatility_1m', 'rsi', 'macd']
        self.feature_importances_ = {}

    def get_features_for_stock(self, prices):
        if len(prices) < 65: return pd.DataFrame()
        prices_tuple = tuple(prices)
        returns = prices.pct_change()
        features = pd.DataFrame(index=prices.index)
        for fname in self.feature_names:
            if fname == 'momentum_1m': features[fname] = returns.rolling(21).sum()
            elif fname == 'volatility_1m': features[fname] = returns.rolling(21).std()
            elif fname == 'rsi': features[fname] = calculate_rsi(prices_tuple, period=14)
            elif fname == 'macd': features[fname] = calculate_macd(prices_tuple, fast=12, slow=26)
        return features.fillna(method='ffill').fillna(0)

    def train_model(self, price_data, stock_universe, cutoff_date):
        training_start_date = cutoff_date - timedelta(days=730)
        train_price_data = price_data.loc[training_start_date:cutoff_date]
        if len(train_price_data) < 252:
            self.is_trained = False; return False

        X_list, y_list = [], []
        targets = train_price_data.pct_change(21).shift(-21)
        
        for stock in stock_universe:
            if stock not in train_price_data.columns or stock not in targets.columns: continue
            features = self.get_features_for_stock(train_price_data[stock])
            if features.empty: continue
            full_data = features.join(targets[stock].rename('target')).dropna()
            if len(full_data) > 500: full_data = full_data.sample(n=500, random_state=42)
            if not full_data.empty:
                X_list.append(full_data.drop('target', axis=1))
                y_list.append(full_data['target'])
            del features, full_data
            gc.collect()
        
        if not y_list or not X_list:
            self.is_trained = False; return False

        X, y = pd.concat(X_list), pd.concat(y_list)
        if len(X) > 20000:
            sample_indices = np.random.choice(X.index, 20000, replace=False)
            X, y = X.loc[sample_indices], y.loc[sample_indices]
        
        print(f"Training ML model with {len(X)} samples...")
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.selector.fit_transform(X_scaled, y)
        self.ridge.fit(X_selected, y)
        
        selected_indices = self.selector.get_support(indices=True)
        self.feature_importances_ = dict(zip([self.feature_names[i] for i in selected_indices], self.ridge.coef_))
        
        self.is_trained = True
        del X, y, X_list, y_list, X_scaled, X_selected
        gc.collect()
        return True

    def predict_returns(self, price_data, stocks, as_of_date):
        if not self.is_trained: return {}
        predictions = {}
        for stock in stocks:
            try:
                recent_prices = price_data.loc[as_of_date - timedelta(days=90):as_of_date, stock]
                features = self.get_features_for_stock(recent_prices)
                if features.empty or as_of_date not in features.index: continue
                feature_vector = features.loc[as_of_date].values.reshape(1, -1)
                if np.isnan(feature_vector).any(): continue
                feature_vector_scaled = self.scaler.transform(feature_vector)
                feature_vector_selected = self.selector.transform(feature_vector_scaled)
                predictions[stock] = self.ridge.predict(feature_vector_selected)[0]
            except Exception: continue
        return predictions

# --- Advanced Risk Manager ---
class AdvancedRiskManager:
    def __init__(self, lookback_period=63):
        self.lookback_period = lookback_period

    def calculate_optimal_weights(self, selected_stocks, final_scores, price_data):
        if not selected_stocks: return {}
        lookback_start = price_data.index.max() - pd.DateOffset(days=self.lookback_period * 2)
        recent_prices = price_data.loc[lookback_start:, selected_stocks].tail(self.lookback_period)
        if len(recent_prices) < self.lookback_period - 5 or recent_prices.isnull().values.any():
            num_stocks = len(selected_stocks); return {s: 1/num_stocks for s in selected_stocks} if num_stocks > 0 else {}
        daily_returns = recent_prices.pct_change().dropna()
        if daily_returns.empty:
            num_stocks = len(selected_stocks); return {s: 1/num_stocks for s in selected_stocks} if num_stocks > 0 else {}
        
        volatility = daily_returns.std()
        metrics = pd.DataFrame(index=selected_stocks)
        metrics['score'] = pd.Series(final_scores)
        metrics['volatility'] = volatility
        metrics = metrics.dropna()
        if metrics.empty: return {}
        
        metrics['base_weight'] = metrics['score'] / (metrics['volatility'] + 1e-6)
        metrics['base_weight'] = metrics['base_weight'].clip(lower=0)
        
        if len(metrics.index) < 2: return {metrics.index[0]: 1.0} if not metrics.empty else {}
        
        corr_matrix = daily_returns[metrics.index].corr()
        avg_correlation = (corr_matrix.sum() - 1) / (len(corr_matrix) - 1)
        metrics['correlation_penalty'] = (1 - avg_correlation).clip(lower=0)
        metrics['final_weight'] = metrics['base_weight'] * metrics['correlation_penalty']
        
        total_weight = metrics['final_weight'].sum()
        if total_weight == 0: return {s: 1/len(metrics.index) for s in metrics.index}
        
        return (metrics['final_weight'] / total_weight).to_dict()

# --- Main Trading Strategy Class ---
class HybridTradingStrategy:
    def __init__(self, rebalance_frequency=30, max_stocks=15):
        self.rebalance_frequency = rebalance_frequency
        self.max_stocks = max_stocks
        self.risk_manager = AdvancedRiskManager()
        self.ml_ranker = OptimizedMLRankingSystem()
        self.factor_ranker = None
        self.fundamental_data = {}

    def _fetch_fundamental_data(self, stock_universe):
        print("Fetching fundamental data...")
        for ticker in stock_universe:
            if ticker not in self.fundamental_data:
                try:
                    info = yf.Ticker(ticker).info
                    self.fundamental_data[ticker] = {'pe_ratio': info.get('trailingPE'), 'pb_ratio': info.get('priceToBook'), 'roe': info.get('returnOnEquity')}
                except Exception: self.fundamental_data[ticker] = {}
        self.factor_ranker = FactorRankingSystem(self.fundamental_data)

    def run_backtest(self, start_date, end_date, selected_stocks=None, transaction_cost_bps=0):
        stock_universe = selected_stocks or []
        if not stock_universe: return {'error': 'No stocks selected.'}

        backtest_start_dt = pd.to_datetime(start_date)
        fetch_start_date = backtest_start_dt - pd.DateOffset(days=750) # Need enough data for 12m momentum + training
        
        try:
            print("Fetching historical price data...")
            price_data = yf.download(tickers=stock_universe, start=fetch_start_date, end=end_date, progress=False, threads=False)['Close']
            price_data = price_data.dropna(axis=1, how='all').ffill().bfill()
            if price_data.empty: return {'error': 'Failed to fetch sufficient stock data.'}
            self._fetch_fundamental_data(price_data.columns)
            benchmark_prices = yf.download('^NSEI', start=fetch_start_date, end=end_date, progress=False)['Close'].ffill()
        except Exception as e: return {'error': f'Data fetching failed: {str(e)}'}

        price_data_backtest = price_data.loc[start_date:end_date]
        bench_daily_returns = benchmark_prices.reindex(price_data_backtest.index).ffill().pct_change().fillna(0)
        rebalance_dates = pd.date_range(start=price_data_backtest.index.min(), end=price_data_backtest.index.max(), freq=f'{self.rebalance_frequency}B')
        valid_rebalance_dates = price_data.index.unique()[price_data.index.unique().searchsorted(rebalance_dates)]
        
        if len(valid_rebalance_dates) < 2: return {'error': 'Backtest period too short for rebalancing.'}

        all_daily_returns = pd.Series(0.0, index=price_data_backtest.index)
        stock_returns, last_weights, portfolio_compositions, trade_log = price_data_backtest.pct_change(), {}, [], []
        
        for i in range(len(valid_rebalance_dates) - 1):
            rebalance_date, next_rebalance_date = valid_rebalance_dates[i], valid_rebalance_dates[i+1]
            print(f"Processing period: {rebalance_date.date()} to {next_rebalance_date.date()}")
            
            ranking_data = price_data.loc[:rebalance_date]
            factor_scores = self.factor_ranker.get_composite_scores(ranking_data)
            
            self.ml_ranker.train_model(price_data, list(price_data.columns), rebalance_date)
            ml_scores = self.ml_ranker.predict_returns(price_data, list(price_data.columns), rebalance_date)

            # Combine scores
            final_scores = factor_scores.copy()
            for stock, ml_pred in ml_scores.items():
                if stock in final_scores and pd.notna(ml_pred):
                    final_scores[stock] = (0.7 * final_scores.get(stock, 0)) + (0.3 * ml_pred)
            
            ranked_stocks = final_scores.sort_values(ascending=False)
            selected = list(ranked_stocks.head(self.max_stocks).index)
            weights_dict = self.risk_manager.calculate_optimal_weights(selected, final_scores, ranking_data)

            final_stocks = list(weights_dict.keys())
            current_weights = dict(zip(final_stocks, weights_dict.values()))
            portfolio_compositions.append({'date': rebalance_date.strftime('%Y-%m-%d'), 'stocks': list(current_weights.items())})
            
            turnover = sum(abs(current_weights.get(s, 0) - last_weights.get(s, 0)) for s in set(current_weights) | set(last_weights))
            costs = turnover * (transaction_cost_bps / 10000.0)
            
            period_mask = (price_data_backtest.index > rebalance_date) & (price_data_backtest.index <= next_rebalance_date)
            if period_mask.any() and final_stocks:
                period_returns = stock_returns.loc[period_mask, final_stocks].fillna(0)
                daily_portfolio_returns = period_returns.dot(list(current_weights.values()))
                if not daily_portfolio_returns.empty:
                    daily_portfolio_returns.iloc[0] -= costs
                    all_daily_returns.loc[period_mask] = daily_portfolio_returns
                
                period_return_pct = (price_data.loc[next_rebalance_date] / price_data.loc[rebalance_date] - 1)
                for stock in final_stocks:
                    trade_log.append({'stock': stock.replace('.NS', ''), 'entry_date': rebalance_date.strftime('%Y-%m-%d'), 'exit_date': next_rebalance_date.strftime('%Y-%m-%d'), 'return_pct': period_return_pct.get(stock, 0) * 100})
            
            last_weights = current_weights; gc.collect()

        portfolio_returns = (1 + all_daily_returns).cumprod()
        benchmark_returns_obj = (1 + bench_daily_returns).cumprod()
        benchmark_returns = benchmark_returns_obj.iloc[:,0] if isinstance(benchmark_returns_obj, pd.DataFrame) else benchmark_returns_obj

        total_return, benchmark_total_return = (portfolio_returns.iloc[-1] - 1) if not portfolio_returns.empty else 0, (benchmark_returns.iloc[-1] - 1) if not benchmark_returns.empty else 0
        daily_returns, daily_bench_returns = portfolio_returns.pct_change().fillna(0), benchmark_returns.pct_change().fillna(0)
        
        if len(daily_bench_returns) > 1 and len(daily_returns) > 1:
            X, y = daily_bench_returns.values.reshape(-1, 1), daily_returns.values
            lin_reg = LinearRegression().fit(X, y); beta, alpha = lin_reg.coef_[0], (y.mean() - lin_reg.coef_[0] * X.mean()) * 252
        else: alpha, beta = 0.0, 0.0

        annual_return = daily_returns.mean() * 252
        downside_std = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0.0
        max_drawdown = (portfolio_returns / portfolio_returns.expanding().max() - 1).min() if not portfolio_returns.empty else 0.0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        return {'trade_log': trade_log, 'feature_importances': self.ml_ranker.feature_importances_, 'portfolio_values': portfolio_returns.tolist(), 'benchmark_values': benchmark_returns.tolist(), 'dates': portfolio_returns.index.strftime('%Y-%m-%d').tolist(), 'portfolio_compositions': portfolio_compositions, 'performance_metrics': {'total_return': float(total_return), 'benchmark_total_return': float(benchmark_total_return), 'alpha': float(alpha), 'beta': float(beta), 'sortino_ratio': float(sortino_ratio), 'calmar_ratio': float(calmar_ratio), 'sharpe_ratio': float(annual_return / (daily_returns.std() * np.sqrt(252))) if daily_returns.std() > 0 else 0.0, 'max_drawdown': float(max_drawdown), 'annualized_volatility': float(daily_returns.std() * np.sqrt(252))}}

# --- Flask App and API Endpoints ---
app = Flask(__name__)
strategy = HybridTradingStrategy()

@app.route('/')
def index():
    return render_template('index.html')

NIFTY_50_STOCKS = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

@lru_cache(maxsize=1)
def get_stock_info_cached(tickers):
    infos = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info and info.get('marketCap'):
                infos.append({'ticker': ticker, 'market_cap': info.get('marketCap'), 'pe_ratio': info.get('trailingPE'), 'pb_ratio': info.get('priceToBook'), 'dividend_yield': info.get('dividendYield'), 'roe': info.get('returnOnEquity')})
        except Exception: continue
    return infos

@app.route('/api/nifty50_stocks')
def get_nifty50_stocks():
    sort_by = request.args.get('sort_by', 'market_cap')
    stock_infos = get_stock_info_cached(tuple(NIFTY_50_STOCKS))
    reverse_sort = sort_by in ['market_cap', 'dividend_yield', 'roe']
    stock_infos.sort(key=lambda x: x.get(sort_by, float('-inf') if reverse_sort else float('inf')), reverse=reverse_sort)
    return jsonify(stock_infos)

@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    try:
        data = request.json
        strategy.rebalance_frequency = int(data.get('rebalance_frequency', 30))
        strategy.max_stocks = int(data.get('max_stocks', 15))
        
        results = strategy.run_backtest(data.get('start_date'), data.get('end_date'), data.get('selected_stocks'), int(data.get('transaction_cost_bps', 0)))
        
        if 'error' in results: return jsonify(results), 400
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
