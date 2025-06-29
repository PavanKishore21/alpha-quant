# main.py - Optimized & Risk-Aware Trading System
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import traceback
import yfinance as yf
from joblib import Parallel, delayed
from functools import lru_cache
import gc
import os

warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Web framework
from flask import Flask, render_template, request, jsonify

# --- Technical Indicator Calculation with Caching ---
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

@lru_cache(maxsize=512)
def calculate_bb_width(prices_tuple, period=20):
    prices = pd.Series(prices_tuple, dtype=float)
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return ((sma + 2 * std) - (sma - 2 * std)) / sma.replace(0, 1)

# --- Advanced Risk Manager ---
class AdvancedRiskManager:
    def __init__(self, lookback_period=63):
        self.lookback_period = lookback_period

    def calculate_optimal_weights(self, selected_stocks, ml_scores, price_data):
        if not selected_stocks:
            return {}

        lookback_start = price_data.index.max() - pd.DateOffset(days=self.lookback_period * 2)
        recent_prices = price_data.loc[lookback_start:, selected_stocks].tail(self.lookback_period)
        
        if len(recent_prices) < self.lookback_period - 5 or recent_prices.isnull().values.any():
            num_stocks = len(selected_stocks)
            if num_stocks == 0: return {}
            equal_weights = [1/num_stocks] * num_stocks
            return dict(zip(selected_stocks, equal_weights))
            
        daily_returns = recent_prices.pct_change().dropna()
        if daily_returns.empty:
            num_stocks = len(selected_stocks)
            if num_stocks == 0: return {}
            equal_weights = [1/num_stocks] * num_stocks
            return dict(zip(selected_stocks, equal_weights))

        volatility = daily_returns.std()
        
        metrics = pd.DataFrame(index=selected_stocks)
        metrics['ml_score'] = pd.Series(ml_scores)
        metrics['volatility'] = volatility
        metrics = metrics.dropna()

        metrics['volatility'].replace(0, np.nan, inplace=True)
        metrics.dropna(inplace=True)
        if metrics.empty: return {}

        metrics['base_weight'] = metrics['ml_score'] / metrics['volatility']
        metrics['base_weight'] = metrics['base_weight'].clip(lower=0)
        
        if len(metrics.index) < 2:
            if not metrics.empty:
                return {metrics.index[0]: 1.0}
            else:
                return {}
        
        corr_matrix = daily_returns[metrics.index].corr()
        avg_correlation = (corr_matrix.sum() - 1) / (len(corr_matrix) - 1)
        
        metrics['correlation_penalty'] = (1 - avg_correlation).clip(lower=0)
        metrics['final_weight'] = metrics['base_weight'] * metrics['correlation_penalty']
        
        total_weight = metrics['final_weight'].sum()
        if total_weight == 0:
            num_stocks = len(metrics.index)
            if num_stocks == 0: return {}
            equal_weights = [1/num_stocks] * num_stocks
            return dict(zip(metrics.index, equal_weights))

        final_weights = metrics['final_weight'] / total_weight
        
        del metrics, daily_returns, recent_prices
        gc.collect()
        
        return final_weights.to_dict()

class OptimizedMLRankingSystem:
    def __init__(self):
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_regression, k=6)
        self.is_trained = False
        self.feature_names = ['momentum_1m', 'momentum_3m', 'volatility_1m', 'rsi', 'macd', 'bb_width']
        self.feature_importances_ = {}

    def get_features_for_stock(self, prices):
        if len(prices) < 65: return pd.DataFrame()
        prices_tuple = tuple(prices)
        returns = prices.pct_change()
        features = pd.DataFrame(index=prices.index)
        for fname in self.feature_names:
            if fname == 'momentum_1m': features[fname] = returns.rolling(21).sum()
            elif fname == 'momentum_3m': features[fname] = returns.rolling(63).sum()
            elif fname == 'volatility_1m': features[fname] = returns.rolling(21).std()
            elif fname == 'rsi': features[fname] = calculate_rsi(prices_tuple)
            elif fname == 'macd': features[fname] = calculate_macd(prices_tuple)
            elif fname == 'bb_width': features[fname] = calculate_bb_width(prices_tuple)
        return features.fillna(method='ffill').fillna(0)

    def train_model(self, price_data, stock_universe, cutoff_date):
        training_start_date = cutoff_date - timedelta(days=730)
        train_price_data = price_data.loc[training_start_date:cutoff_date]
        if len(train_price_data) < 252:
            self.is_trained = False
            return False

        feature_list = [
            self.get_features_for_stock(train_price_data[stock]) for stock in stock_universe
        ]

        X_list, y_list = [], []
        targets = train_price_data.pct_change(21).shift(-21)

        for i, stock in enumerate(stock_universe):
            if i >= len(feature_list) or feature_list[i].empty or stock not in targets.columns: continue
            full_data = feature_list[i].join(targets[stock].rename('target')).dropna()
            if not full_data.empty:
                X_list.append(full_data.drop('target', axis=1))
                y_list.append(full_data['target'])
        
        if not y_list or not X_list:
            self.is_trained = False
            return False

        X, y = pd.concat(X_list), pd.concat(y_list)
        
        if len(X) > 15000:
            sample_indices = np.random.choice(X.index, 15000, replace=False)
            X = X.loc[sample_indices]
            y = y.loc[sample_indices]
        
        print(f"Training with {len(X)} samples for cutoff {cutoff_date.date()}...")
        
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.selector.fit_transform(X_scaled, y)
        
        self.ridge.fit(X_selected, y)
        
        selected_feature_indices = self.selector.get_support(indices=True)
        selected_feature_names = [self.feature_names[i] for i in selected_feature_indices]
        self.feature_importances_ = dict(zip(selected_feature_names, self.ridge.coef_))
        
        self.is_trained = True
        del X, y, X_list, y_list, X_scaled, X_selected
        gc.collect()
        return True

    def predict_returns(self, price_data, stocks, as_of_date):
        if not self.is_trained: 
            return {}
        
        predictions = {}
        for stock in stocks:
            try:
                recent_prices = price_data[stock].loc[as_of_date - timedelta(days=90):as_of_date]
                if recent_prices.empty: continue
                
                features = self.get_features_for_stock(recent_prices)
                if features.empty or as_of_date not in features.index: continue
                
                feature_vector = features.loc[as_of_date].values.reshape(1, -1)
                if np.isnan(feature_vector).any(): continue

                feature_vector_scaled = self.scaler.transform(feature_vector)
                feature_vector_selected = self.selector.transform(feature_vector_scaled)
                
                predictions[stock] = self.ridge.predict(feature_vector_selected)[0]
                
            except Exception:
                continue
        return predictions

class AdvancedTradingStrategy:
    def __init__(self, sharpe_lookback=30, rebalance_frequency=30, max_stocks=15):
        self.sharpe_lookback = sharpe_lookback
        self.rebalance_frequency = rebalance_frequency
        self.max_stocks = max_stocks
        self.risk_manager = AdvancedRiskManager()
        self.ml_ranker = OptimizedMLRankingSystem()
        self.default_stock_universe = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    def run_backtest(self, start_date, end_date, selected_stocks=None, transaction_cost_bps=0):
        stock_universe = selected_stocks if (selected_stocks and len(selected_stocks) > 0) else self.default_stock_universe
        
        print("Fetching market data...")
        backtest_start_dt = pd.to_datetime(start_date)
        fetch_start_date = backtest_start_dt - pd.DateOffset(days=750)
        
        try:
            price_data = yf.download(tickers=stock_universe, start=fetch_start_date, end=end_date, progress=False, threads=False)['Close']
            price_data = price_data.dropna(axis=1, how='all').ffill().bfill()
            if price_data.empty: return {'error': 'Failed to fetch sufficient stock data.'}

            benchmark_prices = yf.download('^NSEI', start=fetch_start_date, end=end_date, progress=False)['Close'].ffill()
        except Exception as e:
            return {'error': f'Failed to fetch market data: {str(e)}'}
        
        price_data_backtest = price_data.loc[start_date:end_date]
        bench_daily_returns = benchmark_prices.reindex(price_data_backtest.index).ffill().pct_change().fillna(0)
        
        rebalance_dates = pd.date_range(start=price_data_backtest.index.min(), end=price_data_backtest.index.max(), freq=f'{self.rebalance_frequency}B')
        valid_rebalance_dates = price_data.index.unique()[price_data.index.unique().searchsorted(rebalance_dates)]
        
        if len(valid_rebalance_dates) < 2: 
            return {'error': 'Backtest period too short for rebalancing.'}

        all_daily_returns = pd.Series(0.0, index=price_data_backtest.index)
        stock_returns = price_data_backtest.pct_change()
        last_weights = {}
        portfolio_compositions = []
        trade_log = []
        
        for i in range(len(valid_rebalance_dates) - 1):
            rebalance_date = valid_rebalance_dates[i]
            next_rebalance_date = valid_rebalance_dates[i+1]
            print(f"Processing period: {rebalance_date.date()} to {next_rebalance_date.date()}")
            
            self.ml_ranker.train_model(price_data, stock_universe, rebalance_date)
            
            ranking_data = price_data.loc[:rebalance_date]
            rolling_returns = ranking_data.pct_change().rolling(self.sharpe_lookback)
            sharpe_scores = ((rolling_returns.mean() * np.sqrt(252)) / (rolling_returns.std() * np.sqrt(252) + 1e-6)).iloc[-1]
            
            # --- FIX: More robust two-step ranking logic ---
            # 1. First, select candidates based on positive Sharpe ratio and ATH
            ath_filter = ranking_data.iloc[-252:].max() <= ranking_data.iloc[-1]
            ath_stocks = ath_filter[ath_filter].index.tolist()
            
            positive_sharpe_stocks = [
                stock for stock, score in sharpe_scores.items() 
                if score > 0 and stock in ath_stocks
            ]
            
            final_scores = sharpe_scores.copy().to_dict()

            # 2. Then, refine the scores of these good candidates with the ML model
            if self.ml_ranker.is_trained:
                ml_scores = self.ml_ranker.predict_returns(price_data, positive_sharpe_stocks, rebalance_date)
                for stock, ml_score in ml_scores.items():
                    if stock in final_scores and pd.notna(final_scores.get(stock)) and pd.notna(ml_score):
                        final_scores[stock] = (0.6 * final_scores[stock]) + (0.4 * ml_score)
            
            # Rank the final candidates
            ranked_stocks = sorted(
                positive_sharpe_stocks, 
                key=lambda s: final_scores.get(s, -np.inf), 
                reverse=True
            )

            selected = ranked_stocks[:self.max_stocks]
            weights_dict = self.risk_manager.calculate_optimal_weights(selected, final_scores, price_data.loc[:rebalance_date])

            if not weights_dict:
                final_stocks, weights_arr = [], []
            else:
                final_stocks = list(weights_dict.keys())
                weights_arr = list(weights_dict.values())
            
            current_weights = dict(zip(final_stocks, weights_arr))
            formatted_weights = [(stock, round(weight, 4)) for stock, weight in current_weights.items()]
            portfolio_compositions.append({'date': rebalance_date.strftime('%Y-%m-%d'), 'stocks': formatted_weights})
            
            turnover = sum(abs(current_weights.get(s, 0) - last_weights.get(s, 0)) for s in set(current_weights) | set(last_weights))
            costs = turnover * (transaction_cost_bps / 10000.0)
            
            period_mask = (price_data_backtest.index > rebalance_date) & (price_data_backtest.index <= next_rebalance_date)
            if period_mask.any() and final_stocks:
                period_returns = stock_returns.loc[period_mask, final_stocks].fillna(0)
                daily_portfolio_returns = period_returns.dot(weights_arr)
                if not daily_portfolio_returns.empty:
                    daily_portfolio_returns.iloc[0] -= costs 
                    all_daily_returns.loc[period_mask] = daily_portfolio_returns
                
                period_return_pct = (price_data.loc[next_rebalance_date] / price_data.loc[rebalance_date] - 1)
                for stock in final_stocks:
                    trade_log.append({'stock': stock.replace('.NS', ''), 'entry_date': rebalance_date.strftime('%Y-%m-%d'), 'exit_date': next_rebalance_date.strftime('%Y-%m-%d'), 'return_pct': period_return_pct.get(stock, 0) * 100})
            
            last_weights = current_weights
            gc.collect()

        portfolio_returns = (1 + all_daily_returns).cumprod()
        benchmark_returns_obj = (1 + bench_daily_returns).cumprod()
        
        if isinstance(benchmark_returns_obj, pd.DataFrame):
            benchmark_returns = benchmark_returns_obj.iloc[:, 0] if not benchmark_returns_obj.empty else pd.Series(1, index=portfolio_returns.index)
        else:
             benchmark_returns = benchmark_returns_obj if not benchmark_returns_obj.empty else pd.Series(1, index=portfolio_returns.index)
        
        total_return = (portfolio_returns.iloc[-1] - 1) if not portfolio_returns.empty else 0
        benchmark_total_return = (benchmark_returns.iloc[-1] - 1) if not benchmark_returns.empty else 0
        daily_returns = portfolio_returns.pct_change().fillna(0)
        daily_bench_returns = benchmark_returns.pct_change().fillna(0)
        
        if len(daily_bench_returns) > 1 and len(daily_returns) > 1:
            X = daily_bench_returns.values.reshape(-1, 1)
            y = daily_returns.values
            lin_reg = LinearRegression().fit(X, y)
            beta = lin_reg.coef_[0]
            alpha = (y.mean() - beta * X.mean()) * 252
        else:
            alpha, beta = 0.0, 0.0

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        annual_return = daily_returns.mean() * 252
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0.0

        max_drawdown = (portfolio_returns / portfolio_returns.expanding().max() - 1).min() if not portfolio_returns.empty else 0.0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        return {
            'trade_log': trade_log,
            'feature_importances': self.ml_ranker.feature_importances_,
            'portfolio_values': portfolio_returns.tolist(),
            'benchmark_values': benchmark_returns.tolist(),
            'dates': portfolio_returns.index.strftime('%Y-%m-%d').tolist(),
            'portfolio_compositions': portfolio_compositions,
            'performance_metrics': {
                'total_return': float(total_return),
                'benchmark_total_return': float(benchmark_total_return),
                'alpha': float(alpha),
                'beta': float(beta),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'sharpe_ratio': float(annual_return / (daily_returns.std() * np.sqrt(252))) if daily_returns.std() > 0 else 0.0,
                'max_drawdown': float(max_drawdown),
                'annualized_volatility': float(daily_returns.std() * np.sqrt(252))
            }
        }

# --- Flask App and API Endpoints ---
app = Flask(__name__)
strategy = AdvancedTradingStrategy()

@app.route('/')
def index():
    return render_template('index.html')

NIFTY_50_STOCKS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS',
    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LTIM.NS',
    'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS',
    'SUNPHARMA.NS', 'TATAMOTORS.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATASTEEL.NS',
    'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

@lru_cache(maxsize=1)
def get_stock_info_cached(tickers):
    infos = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info and info.get('marketCap'):
                infos.append({
                    'ticker': ticker,
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'ev_ebitda': info.get('enterpriseToEbitda'),
                    'roe': info.get('returnOnEquity')
                })
        except Exception:
            continue
    return infos

@app.route('/api/nifty50_stocks')
def get_nifty50_stocks():
    sort_by = request.args.get('sort_by', 'market_cap')
    stock_infos = get_stock_info_cached(tuple(NIFTY_50_STOCKS))
    reverse_sort = sort_by in ['market_cap', 'dividend_yield', 'roe']
    stock_infos.sort(
        key=lambda x: x.get(sort_by) if x.get(sort_by) is not None else (
            float('-inf') if reverse_sort else float('inf')
        ), 
        reverse=reverse_sort
    )
    return jsonify(stock_infos)

@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    try:
        data = request.json
        
        strategy.sharpe_lookback = int(data.get('sharpe_lookback', 30))
        strategy.rebalance_frequency = int(data.get('rebalance_frequency', 30))
        strategy.max_stocks = int(data.get('max_stocks', 15))
        transaction_cost_bps = int(data.get('transaction_cost_bps', 0))
        
        results = strategy.run_backtest(
            data.get('start_date'), 
            data.get('end_date'), 
            data.get('selected_stocks'), 
            transaction_cost_bps
        )
        
        if 'error' in results:
            return jsonify(results), 400
            
        return jsonify(results)
        
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
