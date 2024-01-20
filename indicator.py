import pandas as pd
import numpy as np
import talib

def calculate_sma(df, window=20):
    sma = df['Close'].rolling(window=window).mean()
    return sma

def calculate_ema(df, window=20):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return ema

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    macd_line, signal_line, _ = talib.MACD(df['Close'], fastperiod=short_window, slowperiod=long_window, signalperiod=signal_window)
    return macd_line, signal_line

def calculate_rsi(df, window=14):
    rsi = talib.RSI(df['Close'], timeperiod=window)
    return rsi

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    upper_band, middle_band, lower_band = talib.BBANDS(df['Close'], timeperiod=window, nbdevup=num_std_dev, nbdevdn=num_std_dev)
    return upper_band, lower_band

def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=k_window, slowk_period=d_window, slowd_period=d_window)
    return stoch_k, stoch_d

def calculate_adx(df, window=14):
    adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=window)
    return adx

def calculate_atr(df, window=14):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=window)
    return atr

def calculate_cci(df, window=20):
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=window)
    return cci

def calculate_roc(df, window=12):
    roc = talib.ROC(df['Close'], timeperiod=window)
    return roc

def calculate_obv(df):
    obv = talib.OBV(df['Close'], df['Volume'])
    return obv

def calculate_mfi(df, window=14):
    mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=window)
    return mfi

def calculate_williams_r(df, window=14):
    williams_r = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=window)
    return williams_r

def calculate_parabolic_sar(df, acceleration=0.02, maximum=0.2):
    sar = talib.SAR(df['High'], df['Low'], acceleration=acceleration, maximum=maximum)
    return sar

def calculate_detrended_price_oscillator(df, window=20):
    dpo = df['Close'].shift(window // 2 + 1) - df['Close'].rolling(window=window).mean()
    return dpo

def calculate_price_rate_of_change(df, window=12):
    proc = df['Close'].pct_change(periods=window)
    return proc

def calculate_chande_momentum_oscillator(df, window=14):
    cmo = talib.CMO(df['Close'], timeperiod=window)
    return cmo

def calculate_commodity_channel_index(df, window=20):
    cci = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=window)
    return cci

def calculate_rate_of_change(df, window=12):
    roc = talib.ROC(df['Close'], timeperiod=window)
    return roc

def calculate_ehlers_fisher_transform(df, window=10):
    m = 0.66
    x = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher_m = fisher.rolling(window=window).mean()
    fisher_m = fisher_m.shift(-(window // 2))
    fisher_m = fisher_m.replace([np.inf, -np.inf], np.nan).dropna()
    return fisher_m

def calculate_fibonacci_retracements(df):
    # Implementing Fibonacci Retracements
    high_max = df['High'].max()
    low_min = df['Low'].min()

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    retracement_levels = []

    for level in fib_levels:
        retracement = high_max - (level * (high_max - low_min))
        retracement_levels.append(retracement)

    return retracement_levels 

def calculate_dmi(df, window=14):
    adx, di_plus, di_minus = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=window), \
                             talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=window), \
                             talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=window)
    return adx, di_plus, di_minus

def calculate_aroon(df, window=25):
    aroon_up, aroon_down = talib.AROON(df['High'], df['Low'], timeperiod=window)
    return aroon_up, aroon_down

def calculate_atr_bands(df, window=14, num_std_dev=2):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=window)
    upper_band = df['Close'] + num_std_dev * atr
    lower_band = df['Close'] - num_std_dev * atr
    return upper_band, lower_band


def calculate_chaikin_money_flow(df, window=20):
    money_flow_multiplier = ((2 * df['Close'] - df['Low'] - df['High']) / (df['High'] - df['Low'])).clip(lower=-1, upper=1)
    money_flow_volume = money_flow_multiplier * df['Volume']
    chaikin_money_flow = money_flow_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return chaikin_money_flow


def calculate_keltner_channels(df, window=20, multiplier=2):
    middle_band = df['Close'].rolling(window=window).mean()
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=window)
    upper_band = middle_band + multiplier * atr
    lower_band = middle_band - multiplier * atr
    return upper_band, lower_band


def calculate_elder_ray_index(df, window=13):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    bull_power = df['High'] - ema
    bear_power = ema - df['Low']
    return bull_power, bear_power


# def calculate_ichimoku_cloud(df):
#     # Implementing Ichimoku Cloud
#     # Note: Ichimoku Cloud consists of various components like Kijun-sen, Tenkan-sen, Senkou Span A, Senkou Span B, and Chikou Span.
#     # The calculation involves multiple parameters, and you may need to customize it based on your preference.
#     # Below is a simplified example for illustration purposes.

#     tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
#     kijun_sen = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
#     senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
#     senkou_span_b = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
#     chikou_span = df['Close'].shift(-26)

#     values = {
#         'Tenkan-sen': tenkan_sen,
#         'Kijun-sen': kijun_sen,
#         'Senkou Span A': senkou_span_a,
#         'Senkou Span B': senkou_span_b,
#         'Chikou Span': chikou_span
#     }

#     return values

# def calculate_volume_profile(df):
#     # Implementing Volume Profile
#     # Note: Volume Profile is a histogram that shows the volume traded at each price level.
#     # The calculation involves dividing the price range into segments and calculating the volume in each segment.

#     price_range = df['High'] - df['Low']
#     num_segments = 20  # Adjust as per your preference
#     segment_width = price_range / num_segments

#     volume_profile = []
#     for i in range(num_segments):
#         segment_start = df['Low'] + i * segment_width
#         segment_end = segment_start + segment_width
#         segment_volume = df[(df['High'] >= segment_start) & (df['Low'] <= segment_end)]['Volume'].sum()
#         volume_profile.append(segment_volume)

#     return volume_profile 

# def calculate_pivot_points(df):
#     # Implementing Pivot Points
#     # Note: Pivot Points involve calculating support and resistance levels based on the previous day's high, low, and close prices.
#     # The calculation involves multiple levels (support 1, support 2, resistance 1, resistance 2, etc.).

#     pivot_point = (df['High'] + df['Low'] + df['Close']) / 3
#     support1 = 2 * pivot_point - df['High']
#     resistance1 = 2 * pivot_point - df['Low']
#     support2 = pivot_point - (df['High'] - df['Low'])
#     resistance2 = pivot_point + (df['High'] - df['Low'])

#     values = {
#         'Pivot Point': pivot_point,
#         'Support 1': support1,
#         'Resistance 1': resistance1,
#         'Support 2': support2,
#         'Resistance 2': resistance2
#     }

#     return values