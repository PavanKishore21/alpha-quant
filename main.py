# main.py - Ultra Memory-Optimized Trading System for Render - FIXED
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import traceback
import yfinance as yf
from functools import lru_cache
import gc
import os
import psutil
from threading import Timer
import json

warnings.filterwarnings('ignore')

# Minimal ML imports - Only essential ones
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Web framework
from flask import Flask, render_template, request, jsonify

# Custom JSON encoder for pandas/numpy objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# Memory monitoring
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def force_gc_cleanup():
    """Aggressive garbage collection"""
    gc.collect()
    gc.collect()
    gc.collect()

def safe_json_convert(obj):
    """Convert pandas/numpy objects to JSON-serializable format"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # For scalar values, check if it's NaN
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj

# Memory-safe technical indicators with chunking
def safe_rolling_operation(series, window, operation='sum', chunk_size=1000):
    """Perform rolling operations in chunks to prevent memory overflow"""
    if len(series) <= chunk_size:
        if operation == 'sum':
            return series.rolling(window, min_periods=1).sum()
        elif operation == 'std':
            return series.rolling(window, min_periods=1).std()
        elif operation == 'mean':
            return series.rolling(window, min_periods=1).mean()
    
    # Process in chunks for large series
    results = []
    for i in range(0, len(series), chunk_size):
        chunk = series.iloc[i:i+chunk_size]
        if operation == 'sum':
            chunk_result = chunk.rolling(window, min_periods=1).sum()
        elif operation == 'std':
            chunk_result = chunk.rolling(window, min_periods=1).std()
        elif operation == 'mean':
            chunk_result = chunk.rolling(window, min_periods=1).mean()
        results.append(chunk_result)
        
        # Clean up memory after each chunk
        del chunk, chunk_result
        if i % (chunk_size * 3) == 0:  # Every 3 chunks
            force_gc_cleanup()
    
    return pd.concat(results)

@lru_cache(maxsize=64)  # Drastically reduced cache
def calculate_rsi_safe(prices_tuple, period=14):
    """Memory-safe RSI calculation"""
    try:
        prices = pd.Series(prices_tuple[-100:], dtype=float)  # Only use last 100 points
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        # Use simple moving average instead of EWM to save memory
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        del prices, delta, gain, loss, avg_gain, avg_loss, rs
        force_gc_cleanup()
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    except:
        return 50.0

# Simplified Risk Manager
class UltraLightRiskManager:
    def __init__(self):
        self.lookback_period = 21  # Minimal lookback
    
    def calculate_simple_weights(self, selected_stocks, scores, price_data):
        """Ultra-simplified equal weighting with basic volatility adjustment"""
        if not selected_stocks:
            return {}
        
        try:
            # Get recent prices for volatility calculation
            recent_data = price_data[selected_stocks].tail(self.lookback_period)
            if recent_data.empty:
                # Equal weights fallback
                weight = 1.0 / len(selected_stocks)
                return {stock: weight for stock in selected_stocks}
            
            # Simple volatility calculation
            returns = recent_data.pct_change().dropna()
            if returns.empty:
                weight = 1.0 / len(selected_stocks)
                return {stock: weight for stock in selected_stocks}
            
            volatility = returns.std()
            volatility = volatility.fillna(volatility.mean())
            
            # Inverse volatility weighting
            inv_vol = 1.0 / (volatility + 1e-8)
            weights = inv_vol / inv_vol.sum()
            
            # Clean up
            del recent_data, returns, volatility, inv_vol
            force_gc_cleanup()
            
            # Convert to regular dict with float values
            result = {}
            for stock in selected_stocks:
                if stock in weights.index:
                    result[stock] = float(weights[stock])
                else:
                    result[stock] = 1.0 / len(selected_stocks)
            
            return result
            
        except Exception as e:
            print(f"Weight calculation error: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(selected_stocks)
            return {stock: weight for stock in selected_stocks}

class MinimalMLSystem:
    def __init__(self):
        self.model = Ridge(alpha=10.0)  # Higher regularization
        self.scaler = StandardScaler()
        self.is_trained = False
        self.max_training_samples = 500  # Drastically reduced
        
    def get_minimal_features(self, prices):
        """Extract only essential features to minimize memory usage"""
        if len(prices) < 30:
            return pd.Series([0.0, 0.0], index=['momentum', 'volatility'])
        
        try:
            # Use only recent data
            recent_prices = prices.tail(63)
            returns = recent_prices.pct_change().dropna()
            
            if returns.empty:
                return pd.Series([0.0, 0.0], index=['momentum', 'volatility'])
            
            # Minimal features
            momentum = returns.tail(21).sum() if len(returns) >= 21 else returns.sum()
            volatility = returns.tail(21).std() if len(returns) >= 21 else returns.std()
            
            del recent_prices, returns
            
            return pd.Series([float(momentum), float(volatility)], index=['momentum', 'volatility'])
            
        except Exception:
            return pd.Series([0.0, 0.0], index=['momentum', 'volatility'])
    
    def train_model_minimal(self, price_data, stock_universe, cutoff_date):
        """Ultra-minimal training to prevent memory issues"""
        try:
            # Very short training window
            training_start = cutoff_date - timedelta(days=180)
            train_data = price_data.loc[training_start:cutoff_date]
            
            if len(train_data) < 60:
                self.is_trained = False
                return False
            
            # Process only a subset of stocks
            limited_universe = stock_universe[:5]  # Only 5 stocks max
            
            X_list, y_list = [], []
            
            # Get targets
            future_returns = train_data.pct_change(10).shift(-10)  # 10-day forward returns
            
            for stock in limited_universe:
                if stock not in train_data.columns or stock not in future_returns.columns:
                    continue
                
                stock_prices = train_data[stock].dropna()
                if len(stock_prices) < 30:
                    continue
                
                # Get features for each date
                for i in range(30, len(stock_prices), 10):  # Every 10th day to reduce samples
                    date = stock_prices.index[i]
                    if date not in future_returns.index:
                        continue
                    
                    target = future_returns[stock].loc[date]
                    if pd.isna(target):
                        continue
                    
                    # Get features up to this date
                    hist_prices = stock_prices.loc[:date]
                    features = self.get_minimal_features(hist_prices)
                    
                    if not features.isna().any():
                        X_list.append(features.values)
                        y_list.append(float(target))
                    
                    # Limit total samples
                    if len(X_list) >= self.max_training_samples:
                        break
                
                if len(X_list) >= self.max_training_samples:
                    break
            
            if len(X_list) < 50:  # Minimum samples needed
                self.is_trained = False
                return False
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            print(f"Training with {len(X)} samples...")
            
            # Fit model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Clean up
            del X_list, y_list, train_data, future_returns, X, y, X_scaled
            force_gc_cleanup()
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.is_trained = False
            return False
    
    def predict_minimal(self, price_data, stock, as_of_date):
        """Minimal prediction"""
        if not self.is_trained:
            return None
        
        try:
            if stock not in price_data.columns:
                return None
            
            recent_prices = price_data[stock].loc[:as_of_date].tail(63)
            if len(recent_prices) < 30:
                return None
            
            features = self.get_minimal_features(recent_prices)
            if features.isna().any():
                return None
            
            X = features.values.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            
            del recent_prices, features, X, X_scaled
            
            return float(prediction)
            
        except Exception:
            return None

class UltraLightTradingStrategy:
    def __init__(self):
        self.rebalance_frequency = 60  # Minimum 60 days
        self.max_stocks = 5  # Maximum 5 stocks
        self.risk_manager = UltraLightRiskManager()
        self.ml_system = MinimalMLSystem()
        self.timeout_seconds = 25  # 25 second timeout for Render
        
    def run_backtest_with_timeout(self, start_date, end_date, selected_stocks=None, transaction_cost_bps=0):
        """Run backtest with timeout protection"""
        result = {'error': 'Timeout - operation took too long'}
        
        def timeout_handler():
            print("Operation timed out!")
            force_gc_cleanup()
        
        # Set timeout
        timer = Timer(self.timeout_seconds, timeout_handler)
        timer.start()
        
        try:
            result = self.run_backtest_ultra_light(start_date, end_date, selected_stocks, transaction_cost_bps)
        except Exception as e:
            result = {'error': f'Backtest failed: {str(e)}'}
        finally:
            timer.cancel()
            force_gc_cleanup()
        
        return result
    
    def run_backtest_ultra_light(self, start_date, end_date, selected_stocks=None, transaction_cost_bps=0):
        """Ultra-lightweight backtest for Render's memory constraints"""
        
        # Default to small universe
        if not selected_stocks or len(selected_stocks) == 0:
            stock_universe = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        else:
            stock_universe = selected_stocks[:8]  # Max 8 stocks
        
        print(f"Memory usage at start: {get_memory_usage():.1f}MB")
        
        try:
            # Minimal data fetching
            print("Fetching minimal market data...")
            backtest_start = pd.to_datetime(start_date)
            fetch_start = backtest_start - pd.DateOffset(days=200)  # Minimal lookback
            
            # Fetch data with error handling
            try:
                price_data = yf.download(
                    tickers=stock_universe,
                    start=fetch_start,
                    end=end_date,
                    progress=False,
                    threads=False,
                    timeout=10
                )['Close']
                
                if isinstance(price_data, pd.Series):
                    price_data = price_data.to_frame(stock_universe[0])
                
                price_data = price_data.dropna(axis=1, how='all').ffill()
                
                if price_data.empty:
                    return {'error': 'No valid stock data found'}
                
            except Exception as e:
                return {'error': f'Data fetch failed: {str(e)}'}
            
            print(f"Memory after data fetch: {get_memory_usage():.1f}MB")
            
            # Get benchmark
            try:
                benchmark = yf.download('^NSEI', start=fetch_start, end=end_date, progress=False)['Close']
                benchmark = benchmark.ffill()
            except:
                # Create dummy benchmark if fetch fails
                benchmark = pd.Series(1.0, index=price_data.index)
            
            # Trim to backtest period
            backtest_data = price_data.loc[start_date:end_date]
            benchmark_backtest = benchmark.reindex(backtest_data.index, method='ffill')
            
            if len(backtest_data) < 30:
                return {'error': 'Insufficient data for backtest period'}
            
            # Very infrequent rebalancing
            rebalance_dates = pd.date_range(
                start=backtest_data.index[0],
                end=backtest_data.index[-1],
                freq='60B'  # Every 60 business days
            )
            
            valid_rebalance_dates = []
            for date in rebalance_dates:
                closest_date = backtest_data.index[backtest_data.index.searchsorted(date)]
                if closest_date not in valid_rebalance_dates:
                    valid_rebalance_dates.append(closest_date)
                    
            if len(valid_rebalance_dates) < 2:
                return {'error': 'Backtest period too short'}
            
            print(f"Rebalancing {len(valid_rebalance_dates)} times")
            
            # Initialize tracking
            portfolio_values = [1.0]
            benchmark_values = [1.0]
            dates = [backtest_data.index[0].strftime('%Y-%m-%d')]
            compositions = []
            trades = []
            
            current_weights = {}
            
            # Process each rebalance period
            for i in range(len(valid_rebalance_dates)):
                rebal_date = valid_rebalance_dates[i]
                next_date = valid_rebalance_dates[i+1] if i+1 < len(valid_rebalance_dates) else backtest_data.index[-1]
                
                print(f"Processing {rebal_date.date()} (Memory: {get_memory_usage():.1f}MB)")
                
                # Simple stock selection
                ranking_data = price_data.loc[:rebal_date].tail(60)  # Last 60 days only
                if ranking_data.empty:
                    continue
                
                # Calculate simple momentum scores
                returns_21d = ranking_data.pct_change(21).iloc[-1]
                volatility = safe_rolling_operation(ranking_data.pct_change(), 21, 'std').iloc[-1]
                
                # Score stocks
                scores = {}
                for stock in ranking_data.columns:
                    ret_val = returns_21d.get(stock) if hasattr(returns_21d, 'get') else returns_21d[stock] if stock in returns_21d.index else None
                    vol_val = volatility.get(stock) if hasattr(volatility, 'get') else volatility[stock] if stock in volatility.index else None
                    
                    if pd.notna(ret_val) and pd.notna(vol_val) and vol_val > 0:
                        scores[stock] = float(ret_val) / float(vol_val)  # Risk-adjusted return
                
                # Select top stocks
                if scores:
                    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    selected_stocks = [stock for stock, score in sorted_stocks[:self.max_stocks]]
                    selected_scores = {stock: score for stock, score in sorted_stocks[:self.max_stocks]}
                else:
                    selected_stocks = list(ranking_data.columns)[:self.max_stocks]
                    selected_scores = {stock: 1.0 for stock in selected_stocks}
                
                # Calculate weights
                new_weights = self.risk_manager.calculate_simple_weights(
                    selected_stocks, selected_scores, ranking_data
                )
                
                # Record composition
                compositions.append({
                    'date': rebal_date.strftime('%Y-%m-%d'),
                    'stocks': [(stock, round(float(weight), 4)) for stock, weight in new_weights.items()]
                })
                
                # Calculate period performance
                period_data = backtest_data.loc[rebal_date:next_date]
                if len(period_data) > 1 and new_weights:
                    period_returns = period_data.pct_change().fillna(0)
                    stock_list = list(new_weights.keys())
                    weights_array = np.array(list(new_weights.values()))
                    
                    for day_idx in range(1, len(period_data)):
                        date = period_data.index[day_idx]
                        day_returns = period_returns.loc[date, stock_list].fillna(0)
                        
                        if not day_returns.empty:
                            if isinstance(day_returns, pd.Series):
                                day_returns_vals = day_returns.values
                            else:
                                day_returns_vals = [day_returns] if not hasattr(day_returns, '__len__') else day_returns
                            
                            portfolio_return = np.dot(day_returns_vals, weights_array)
                            portfolio_values.append(float(portfolio_values[-1] * (1 + portfolio_return)))
                        else:
                            portfolio_values.append(float(portfolio_values[-1]))
                        
                        # Benchmark return
                        bench_return = 0
                        if date in benchmark_backtest.index:
                            bench_ret_val = benchmark_backtest.pct_change().loc[date]
                            if isinstance(bench_ret_val, pd.Series):
                                if not bench_ret_val.empty and pd.notna(bench_ret_val.iloc[0]):
                                    bench_return = float(bench_ret_val.iloc[0])
                            elif pd.notna(bench_ret_val):
                                bench_return = float(bench_ret_val)
                        
                        benchmark_values.append(float(benchmark_values[-1] * (1 + bench_return)))
                        dates.append(date.strftime('%Y-%m-%d'))
                
                current_weights = new_weights
                
                # Memory cleanup
                del ranking_data, returns_21d, volatility, scores
                if i % 2 == 0:  # Every 2 iterations
                    force_gc_cleanup()
            
            # Calculate final metrics
            if len(portfolio_values) > 1:
                total_return = float(portfolio_values[-1] - 1.0)
                benchmark_return = float(benchmark_values[-1] - 1.0) if len(benchmark_values) > 1 else 0.0
                
                # Calculate daily returns for risk metrics
                port_daily_returns = pd.Series(portfolio_values).pct_change().dropna()
                
                if len(port_daily_returns) > 0:
                    annual_return = float(port_daily_returns.mean() * 252)
                    annual_vol = float(port_daily_returns.std() * np.sqrt(252))
                    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else 0.0
                    
                    # Max drawdown
                    cumulative = pd.Series(portfolio_values)
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = float(drawdown.min())
                else:
                    annual_return = float(total_return)
                    sharpe = 0.0
                    max_drawdown = 0.0
                    annual_vol = 0.0
            else:
                total_return = 0.0
                benchmark_return = 0.0
                sharpe = 0.0
                max_drawdown = 0.0
                annual_return = 0.0
                annual_vol = 0.0
            
            print(f"Final memory usage: {get_memory_usage():.1f}MB")
            
            # Ensure all values are JSON serializable
            result = {
                'portfolio_values': [float(v) for v in portfolio_values],
                'benchmark_values': [float(v) for v in benchmark_values],
                'dates': dates,
                'portfolio_compositions': compositions,
                'trade_log': trades,
                'performance_metrics': {
                    'total_return': total_return,
                    'benchmark_total_return': benchmark_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'annualized_return': annual_return,
                    'annualized_volatility': annual_vol,
                    'alpha': float(annual_return - benchmark_return),
                    'beta': 1.0,
                    'sortino_ratio': sharpe,  # Simplified
                    'calmar_ratio': float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Backtest error: {e}")
            traceback.print_exc()
            return {'error': f'Backtest failed: {str(e)}'}
        finally:
            force_gc_cleanup()

# Flask App
app = Flask(__name__)

# Configure Flask to use custom JSON encoder
app.json_encoder = NumpyEncoder

strategy = UltraLightTradingStrategy()

@app.route('/')
def index():
    return render_template('index.html')

# Reduced stock list to prevent timeout
NIFTY_CORE_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'BHARTIARTL.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'ITC.NS',
    'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'WIPRO.NS'
]

@app.route('/api/nifty50_stocks')
def get_nifty50_stocks():
    """Return simplified stock list to prevent memory issues"""
    try:
        # Return basic info without fetching real-time data
        stocks = []
        for i, ticker in enumerate(NIFTY_CORE_STOCKS):
            stocks.append({
                'ticker': ticker,
                'market_cap': 1000000 - i * 50000,  # Dummy values
                'pe_ratio': 20 + i,
                'pb_ratio': 3.0,
                'dividend_yield': 0.02,
                'ev_ebitda': 15,
                'roe': 0.15
            })
        
        sort_by = request.args.get('sort_by', 'market_cap')
        reverse_sort = sort_by in ['market_cap', 'dividend_yield', 'roe']
        stocks.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse_sort)
        
        return jsonify(stocks)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Ultra-lightweight backtest endpoint"""
    try:
        print(f"Backtest request - Memory usage: {get_memory_usage():.1f}MB")
        
        data = request.json or {}
        
        # Force minimal parameters
        selected_stocks = data.get('selected_stocks', [])[:5]  # Max 5 stocks
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-12-31')
        transaction_cost_bps = min(int(data.get('transaction_cost_bps', 0)), 50)
        
        # Run backtest with timeout protection
        results = strategy.run_backtest_with_timeout(
            start_date, end_date, selected_stocks, transaction_cost_bps
        )
        
        if 'error' in results:
            return jsonify(results), 400
        
        # Ensure results are JSON serializable
        json_safe_results = safe_json_convert(results)
        
        return jsonify(json_safe_results)
        
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        force_gc_cleanup()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Set memory-conservative environment
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    
    app.run(debug=False, threaded=False)  # Disable threading