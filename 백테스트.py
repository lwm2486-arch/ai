import ccxt
import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import time
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 1. 바이낸스 15분봉, 1시간봉 데이터 뽑는 함수
def fetch_binance_ohlcv(symbol, timeframe, n=5, limit=1000):
    exchange = ccxt.binance()
    all_ohlcv = []
    now = exchange.milliseconds()
    mins = int(timeframe.replace('m','')) if 'm' in timeframe else int(timeframe.replace('h',''))*60
    since = now - n * limit * mins * 60 * 1000
    for i in range(n):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv: break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + mins*60*1000
        time.sleep(0.5)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated()]
    return df

# 2. 데이터 수집
data_15m = fetch_binance_ohlcv('BTC/USDT', '15m', n=5)
data_1h = fetch_binance_ohlcv('BTC/USDT', '1h', n=5)

# 3. 지표 계산
def add_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = df['BB_High'] - df['BB_Low'] 
    return df

data_15m = add_indicators(data_15m)
data_1h = add_indicators(data_1h)

# 4. 1시간봉 EMA를 15분봉에 병합(리샘플+ffill)
data_1h_resampled = data_1h[['EMA12', 'EMA26']].resample('15min').ffill()
data_15m = data_15m.merge(data_1h_resampled, left_index=True, right_index=True, suffixes=('', '_1h'))

# 5. 신호 생성
def make_signals(df):
    df['Signal'] = 0
    bb_width_threshold = df['BB_Width'].rolling(50).median().iloc[-1] * 1.2
    slope_threshold = 0.003
    momentum_buy = (
        (df['EMA12'] > df['EMA26']) &
        (df['Close'] > df['EMA12']) &
        (df['EMA12'] > df['EMA12'].shift())&
        (df['EMA12_1h'] > df['EMA26_1h'])&
        (df['BB_Width'] > df['BB_Width'].rolling(50).mean())&
        (df['RSI'] < 65)&(df['ADX'] > 20)&
        ((df['RSI'].rolling(10).max() - df['RSI'].rolling(10).min()) > 10)

)
    
    early_buy = (
    (df['RSI'] < 50) &                         
    (df['RSI'] > df['RSI'].shift(1)) &
    (df['RSI'] > df['RSI'].shift(2)) &
    (df['RSI'].shift(3) < 35) & 
    (df['Close'] > df['Open']) &
    (df['Close'] > df['BB_Low']) &
    (df['BB_Width'] > df['BB_Width'].mean() * 1.2)&
    (df['ADX'] > 25) &
    (df['Close'] > df['EMA12'])
)
    bb_rebound = (
    (df['Close'] < df['BB_Low']) &
    (df['RSI'] > df['RSI'].shift()) &
    (df['Close'] > df['Open']) &
    (df['EMA12'] > df['EMA26'])&
    (df['ADX'] > 20)
)
    early_breakout = (
    (df['RSI'] > df['RSI'].shift()) &
    (df['Close'] > df['Open']) &
    ((df['Close'] - df['Open']) / df['Open'] > 0.002) &   # 양봉 강도
    (df['Close'] > df['EMA12']) &
    (df['EMA12'] > df['EMA26']) &
    (df['EMA12'].shift() <= df['EMA26'].shift()) &
    (df['ADX'] > 20) &                                     # 추세 필터
    (df['BB_Width'] > bb_width_threshold)  # 횡보 제거
)
    buy_cond = (
   momentum_buy
)

    # RSI 과매수 꺾임 + 양봉 끝나고 음봉 시작
    rsi_reversal = (
    (df['RSI'].shift(1) > 70) &
    (df['RSI'] < df['RSI'].shift(1)) &
    (df['Close'] < df['Open'])              # 음봉 전환
)

# EMA 데드크로스 + 하락 ADX
    ema_cross = (
    (df['EMA12'] < df['EMA26']) &
    (df['EMA12'].shift(1) >= df['EMA26'].shift(1)) &
     (df['Close'] < df['EMA12']) 
)

# 고점 돌파 실패 + 캔들 전환
    potential_peak = (
    (df['High'] < df['High'].shift(1)) &
    (df['Close'] < df['Open']) &
    (df['RSI'] > 65)
)

# MA이탈 청산 조건 (보조)
    ema_fail = (
    (df['Close'] < df['EMA12']) &
    (df['Close'].shift(1) > df['EMA12'].shift(1)) &
    (df['EMA12'] < df['EMA26'])
)

# 최종 매도 조건
    sell_cond = (
    rsi_reversal |
    ema_cross |
    potential_peak |
    ema_fail
)

    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

data_15m = make_signals(data_15m)

# 6. 백테스트(수수료 반영)
def run_backtest_with_fee(tp_ratio, sl_ratio, df, fee_rate=0.001):
    position = 0
    entry_price = 0
    returns = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        if row['Signal'] == 1 and position == 0:
            position = 1
            entry_price = row['Close'] * (1 + fee_rate)
        elif position == 1:
            current_price = row['Close'] * (1 - fee_rate)
            pnl = (current_price - entry_price) / entry_price
            if row['Signal'] == -1 or pnl >= tp_ratio or pnl <= -sl_ratio:
                returns.append(pnl)
                position = 0
                entry_price = 0
            else:
                returns.append(0)
        else:
            returns.append(0)
    cumulative = (1 + pd.Series(returns).fillna(0)).cumprod()
    return cumulative.iloc[-1]
def run_backtest_with_logs(tp_ratio, sl_ratio, df, fee_rate=0.001):
    position = 0
    entry_price = 0
    entry_time = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if row['Signal'] == 1 and position == 0:
            position = 1
            entry_price = row['Close'] * (1 + fee_rate)
            entry_time = row.name
            trades.append({
                'Type': 'Buy',
                'Time': entry_time,
                'Price': entry_price
            })

        elif position == 1:
            current_price = row['Close'] * (1 - fee_rate)
            pnl = (current_price - entry_price) / entry_price
            exit_reason = None

            if row['Signal'] == -1:
                exit_reason = 'Signal Sell'
            elif pnl >= tp_ratio:
                exit_reason = 'Take Profit'
            elif pnl <= -sl_ratio:
                exit_reason = 'Stop Loss'

            if exit_reason:
                trades.append({
                    'Type': 'Sell',
                    'Time': row.name,
                    'Price': current_price,
                    'PnL': pnl,
                    'Reason': exit_reason
                })
                position = 0
                entry_price = 0
                entry_time = None

    return pd.DataFrame(trades)
# 신호 생성 후, 바로 아래에 추가
data_15m = make_signals(data_15m)


# 🚨 신호 발생 개수 프린트
print("매수 신호 개수:", (data_15m['Signal'] == 1).sum())
print("매도 신호 개수:", (data_15m['Signal'] == -1).sum())
print("전체 row 수:", len(data_15m))
# 7. 파라미터 튜닝 & 시각화
tp_list = [0.03, 0.05, 0.07, 0.1]
sl_list = [0.005, 0.008, 0.01, 0.012]
results = []

for tp in tp_list:
    for sl in sl_list:
        ret = run_backtest_with_fee(tp, sl, data_15m)
        results.append({'TP': tp, 'SL': sl, 'Return': ret})

results_df = pd.DataFrame(results)
pivot = results_df.pivot(index='TP', columns='SL', values='Return')

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("익절/손절 비율별 누적 수익률 (MTF, 수수료 반영, Binance 15m+1h)")
plt.xlabel("손절 비율")
plt.ylabel("익절 비율")
plt.tight_layout()
plt.show()
trade_log = run_backtest_with_logs(0.05, 0.01, data_15m)  # 파라미터는 예시

# plotly로 캔들 + 매수/매도 + 수익률 시각화
import plotly.graph_objects as go

df = data_15m.dropna().copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], mode='lines', name='EMA12', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'], mode='lines', name='EMA26', line=dict(color='orange', dash='dash')))

# 매수
buy_trades = trade_log[trade_log['Type'] == 'Buy']
fig.add_trace(go.Scatter(
    x=buy_trades['Time'],
    y=buy_trades['Price'],
    mode='markers+text',
    name='Buy',
    marker=dict(symbol='arrow-up', size=14, color='blue'),
    text=['진입'] * len(buy_trades),
    textposition="top center"
))

# 매도 + 수익률 + 색구분
sell_trades = trade_log[trade_log['Type'] == 'Sell']
def get_color(reason):
    return 'green' if reason == 'Take Profit' else 'red' if reason == 'Stop Loss' else 'gray'

fig.add_trace(go.Scatter(
    x=sell_trades['Time'],
    y=sell_trades['Price'],
    mode='markers+text',
    name='Sell',
    marker=dict(
        symbol='arrow-down',
        size=14,
        color=[get_color(r) for r in sell_trades['Reason']]
    ),
    text=[f"{r}<br>{pnl*100:.2f}%" for r, pnl in zip(sell_trades['Reason'], sell_trades['PnL'])],
    textposition="bottom center"
))

fig.update_layout(
    title='BTC/USDT 진입-청산 시각화 (수익률 & 익절/손절 색구분)',
    xaxis_rangeslider_visible=True,
    template='plotly_white',
    height=850,
    yaxis=dict(fixedrange=False)
)

fig.show()
import matplotlib.pyplot as plt

# 📌 1. 수익률 통계 요약
# 📌 수익률 기준으로 익절/손절/기타 구분
sell_trades = trade_log[trade_log['Type'] == 'Sell'].copy()

sell_trades['Outcome'] = sell_trades['PnL'].apply(lambda x: 'Take Profit' if x > 0 else 'Stop Loss' if x < 0 else 'Break Even')

take_profit = sell_trades[sell_trades['Outcome'] == 'Take Profit']
stop_loss = sell_trades[sell_trades['Outcome'] == 'Stop Loss']
break_even = sell_trades[sell_trades['Outcome'] == 'Break Even']

total_trades = len(sell_trades)
win_rate = len(take_profit) / total_trades * 100 if total_trades > 0 else 0

print(f"\n✅ 총 거래 수: {total_trades}")
print(f"🟢 익절 수 (PnL > 0): {len(take_profit)}")
print(f"🔴 손절 수 (PnL < 0): {len(stop_loss)}")
print(f"⚪ 본전 청산 수 (PnL = 0): {len(break_even)}")
print(f"🏆 승률 (수익률 기준): {win_rate:.2f}%")


# 📌 2. 누적 수익률 계산
cumulative_return = (1 + sell_trades['PnL'].fillna(0)).cumprod()

# 📈 누적 수익률 그래프
plt.figure(figsize=(10, 5))
plt.plot(sell_trades['Time'], cumulative_return, marker='o', label='누적 수익률')
plt.axhline(1.0, color='gray', linestyle='--')
plt.title("누적 수익률 변화")
plt.xlabel("청산 시점")
plt.ylabel("수익률 (배)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
buy_logs = trade_log[trade_log['Type'] == 'Buy'].reset_index(drop=True)
sell_logs = trade_log[trade_log['Type'] == 'Sell'].reset_index(drop=True)

paired_log = pd.DataFrame({
    '진입시각': buy_logs['Time'],
    '진입가격': buy_logs['Price'],
    '청산시각': sell_logs['Time'],
    '청산가격': sell_logs['Price'],
    '수익률(%)': (sell_logs['PnL'] * 100).round(2),
    '청산사유': sell_logs['Reason']
})

print(paired_log)
