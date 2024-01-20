from indicator import *
import pandas as pd


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df['sma1'] = calculate_sma(df, window=5) 
    df['sma2'] = calculate_sma(df, window=10) 
    df['sma3'] = calculate_sma(df, window=30) 

    df['ema1'] = calculate_ema(df, window=5) 
    df['ema2'] = calculate_ema(df, window=10) 
    df['ema3'] = calculate_ema(df, window=30) 

    macd_line, macd_signal = calculate_macd(df, short_window=12, long_window=26, signal_window=9)
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal

    df['rsi1'] = calculate_rsi(df, 14)
    df['rsi2'] = calculate_rsi(df, 21)

    df['adx1'] = calculate_adx(df, window=14)
    df['adx2'] = calculate_adx(df, window=21)

    df['atr1'] = calculate_atr(df, window=14)
    df['atr2'] = calculate_atr(df, window=21)

    df['cci1'] = calculate_cci(df, window=14)
    df['cci2'] = calculate_cci(df, window=21)

    df['roc1'] = calculate_roc(df, window=12)
    df['roc2'] = calculate_roc(df, window=24)

    df['obv'] = calculate_obv(df)

    df['mfi1'] = calculate_mfi(df, window=14)
    df['mfi2'] = calculate_mfi(df, window=21)

    df['william1'] = calculate_williams_r(df, window=14)
    df['william2'] = calculate_williams_r(df, window=21)

    df['sar'] = calculate_parabolic_sar(df, acceleration=0.02, maximum=0.2)

    df['d_oscillator'] = calculate_detrended_price_oscillator(df, window=20)

    df['price_rate_change'] = calculate_price_rate_of_change(df, window=12)

    df['m_oscillator'] = calculate_chande_momentum_oscillator(df, window=14)

    df['cm_channel'] = calculate_commodity_channel_index(df, window=20)

    df['rate_change'] = calculate_rate_of_change(df, window=12)

    df['euler_fisher_transform'] = calculate_ehlers_fisher_transform(df, window=10)

    fib1, fib2, fib3, fib4, fib5 = calculate_fibonacci_retracements(df)
    df['fib1'] = fib1 
    df['fib2'] = fib2 
    df['fib3'] = fib3 
    df['fib4'] = fib4 
    df['fib5'] = fib5 

    adx, di_plus, di_minus = calculate_dmi(df, window=14) 
    df['dmi'] = adx
    df['dmi'] = di_plus 
    df['dmi'] = di_minus

    aroon_up, aroon_down = calculate_aroon(df, window=25)
    df['aroon_up'] = aroon_up
    df['aroon_down'] = aroon_down

    atr_upper, atr_lower = calculate_atr_bands(df, window=14, num_std_dev=2)
    df['atr_upper'] = atr_upper
    df['atr_lower'] = atr_lower

    df['chaikin_flow'] = calculate_chaikin_money_flow(df, window=20)

    keltner_upper, keltner_lower = calculate_keltner_channels(df, window=20, multiplier=2)
    df['keltner_upper'] = keltner_upper 
    df['keltner_lower'] = keltner_lower

    elder_bull, elder_bear = calculate_elder_ray_index(df, window=13)
    df['elder_bull'] = elder_bull
    df['elder_bear'] = elder_bear

    upper, lower = calculate_bollinger_bands(df, window=20, num_std_dev=2)
    df['bollinger_upper'] = upper 
    df['bollinger_lower'] = lower 

    stoch_k, stoch_d = calculate_stochastic_oscillator(df, k_window=14, d_window=3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    df = df.dropna()

    return df



