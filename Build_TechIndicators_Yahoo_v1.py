import sqlite3
from timeit import default_timer as timer
import talib as ta
import pandas as pd
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from calendar import isleap

desired_width = 360
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
dates_df = pd.DataFrame()


def get_symbols(symbols_file):
    df = pd.read_csv(symbols_file, index_col=False)
    df['validDataDate'] = pd.to_datetime(df['validDataDate'], infer_datetime_format=True)
    df['endDate'] = pd.to_datetime(df['endDate'], infer_datetime_format=True)
    df['stDate'] = pd.to_datetime(df['stDate'], infer_datetime_format=True)
    # df = df.sort_values(['Symbol'], ascending=True)
    return df


def chunks_to_df(gen):
    chunks = []
    for df in gen:
        chunks.append(df)
    return pd.concat(chunks).reset_index().drop('index', axis=1)


def get_stockhistory(hist_st_date, hist_end_date, symbols_df):
    conn = sqlite3.connect("NewDatabase.db")
    conn.text_factory = lambda b: b.decode(errors='ignore')
    start = timer()

    df_chunks = pd.read_sql_query(
        'select Date, symbol, Open, High, Low, Close, Volume from NYSE_NASDAQ_AMEX_Stocks_Yahoo_MSTR2 where Date >= (?) and Date <= (?)',
        conn, params=(hist_st_date, hist_end_date), chunksize=1000000)
    df = chunks_to_df(df_chunks)
    conn.close()
    print('db query time for stock history read', timer() - start)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    # start = timer()
    # print(df.shape[0])
    # = df.drop_duplicates(subset=['symbol', 'Date'], keep='first')
    # print(df.shape[0])
    # print('time for drop dups', timer() - start)
    # unique_symbols_df = df.groupby('symbol', as_index=False).agg({"Date": "min", 'Date': 'max'})
    # unique_symbols_df = df.groupby('symbol', as_index=False).agg({'Date': [min, max]})
    # unique_symbols_df.columns = ['symbol', 'stDate', 'endDate']
    # unique_symbols_df['offsetDays'] = 100
    # temp = unique_symbols_df['offsetDays'].apply(np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
    # unique_symbols_df['validDataDate'] = unique_symbols_df['stDate'] + temp
    # unique_symbols_df.to_csv('unique_symbols_df.csv')
    df = df.merge(symbols_df, how='left')

    print('stock history records from db:', df.shape[0])
    # df['Date'] = df['Date'].dt.tz_localize(tz=None)
    # df = df.round(decimals=4)

    return df


def BBANDS(df, window):
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['Close'], window, 2, 2)
    # df['percBB'] = (df['Close'] - df['lowerband']) / (df['upperband'] - df['lowerband'])
    return (df['Close'] - df['lowerband']) / (df['upperband'] - df['lowerband'])


def streaks(df):
    # df['chg4Streak'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    sign = np.sign(df['chg4Streak'])
    s = sign.groupby((sign != sign.shift()).cumsum()).cumsum()
    return df.assign(w_streak=s.where(s > 0, 0.0), l_streak=s.where(s < 0, 0.0).abs())


def get_FVNextOpen(df):
    df['Next_OPCPerc'] = ((df['Nextopen'] - df['Close']) / df[
        'Close']) * 100
    df['Next_COPerc'] = ((df['Nextclose'] - df['Nextopen']) / df[
        'Nextopen']) * 100
    df['Next_CCPerc'] = ((df['Nextclose'] - df['Close']) / df[
        'Close']) * 100
    spy_df = df[df['symbol'] == 'SPY'][
        ['Date', 'Next_OPCPerc', 'Close', 'Next_COPerc', 'Next_CCPerc', 'validDataDate', 'endDate']]
    spy_df.columns = ['Date', 'SPYNext_OPCPerc', 'SPYClose', 'SPYNext_COPerc', 'SPYNext_CCPerc', 'validDataDate',
                      'endDate']
    spy_df2 = spy_df.copy(deep=True)
    spy_df.drop(columns=['validDataDate','endDate'], inplace=True)

    spy_df.sort_values(['Date'], inplace=True, ascending=True)
    df = df.merge(spy_df, how='left')
    df['FVNextOpen'] = (1 + (df['SPYNext_OPCPerc'] / 100)) * df['Close']
    df['ActFVVar'] = ((df['Nextopen'] - df['FVNextOpen']) / df[
        'FVNextOpen']) * 100

    return df, spy_df2


def get_NextTradesideCOPerc(df, trade):
    # df['Next_COPerc'] = ((df['Nextclose'] - df['Nextopen']) / df['Nextopen']) * 100
    df['tradeSideNextCOPerc'] = df['Next_COPerc']
    df['tradeSideNextSPYCOPerc'] = df['SPYNext_COPerc']
    if trade == 'Short':
        df['tradeSideNextCOPerc'] = df['tradeSideNextCOPerc'] * -1
        # for comparision with benchmark, SPY should always be long
        #df['tradeSideNextSPYCOPerc'] = df['tradeSideNextSPYCOPerc'] * -1
    return df


def plot(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df['Date'], df['equity'])
    ax1.set_ylabel('Equity')

    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['drawDowns'], 'r-')
    ax2.set_ylabel('drawDowns', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    # df.plot(x='Date', y='drawDowns')
    plt.show()


def calcCAGR(df):
    end_date = df['Date'].max()
    start_date = df['Date'].min()
    end_price = df.loc[(df['Date'] == end_date), 'cumReturns'].values[0]
    start_price = df.loc[df['Date'] == start_date, 'cumReturns'].values[0]
    diffyears = end_date.year - start_date.year
    difference = end_date - start_date.replace(end_date.year)
    days_in_year = isleap(end_date.year) and 366 or 365
    number_years = diffyears + difference.days / days_in_year
    cagr = (pow((end_price / start_price), (1 / number_years)) - 1) * 100
    total_return = round(((end_price - start_price) / start_price) * 100)
    return total_return, cagr


def write_results_to_db(result_bydate_df, spy_df, summary_df):
    conn = sqlite3.connect("NewDatabase.db")
    summary_df.to_sql('AllSymbolsResults_TBL', conn, if_exists='replace', index=False)
    result_bydate_df.sort_values(['Date'], inplace=True, ascending=False)
    result_bydate_df.to_sql('ByDateResults_TBL', conn, if_exists='replace', index=False)
    spy_df.to_sql('SPYResults_TBL', conn, if_exists='replace', index=False)
    conn.close()


def build_daily_metrics(result_bydate_df):
    result_bydate_df['endingCapital'] = result_bydate_df['startingCapital'] + result_bydate_df['dayPL']
    result_bydate_df['equity'] = result_bydate_df['dayPL'].cumsum()
    result_bydate_df['returnPerc'] = (result_bydate_df['dayPL'] / result_bydate_df['startingCapital']) * 100
    result_bydate_df['cumReturns'] = ((result_bydate_df['returnPerc'] / 100) + 1).cumprod()
    result_bydate_df['drawDowns'] = 1 - result_bydate_df['cumReturns'].div(result_bydate_df['cumReturns'].cummax())
    result_bydate_df['drawDowns'] = result_bydate_df['drawDowns'] * 100
    """reusing streaks method to calculate streak of win and losing days"""
    result_bydate_df['chg4Streak'] = np.where((result_bydate_df['returnPerc'] > 0), 1, -1)
    result_bydate_df = streaks(result_bydate_df)
    result_bydate_df['win_streak'] = result_bydate_df['w_streak']
    result_bydate_df['loss_streak'] = result_bydate_df['l_streak']
    """Calculating max Drawdown duration and max duration without a Drawdown by reusing streaks"""
    result_bydate_df['chg4Streak'] = np.where(result_bydate_df['drawDowns'] > 0, 1, -1)
    result_bydate_df = streaks(result_bydate_df)
    result_bydate_df['DDw_streak'] = result_bydate_df['w_streak']
    result_bydate_df['DDl_streak'] = result_bydate_df['l_streak']
    return result_bydate_df

def get_final_perf(result_bydate_df):
    '''losses and streaks'''
    maxLoss = result_bydate_df['dayPL'].min()
    maxProfit = result_bydate_df['dayPL'].max()
    maxWStreak = result_bydate_df['w_streak'].max()
    maxLStreak = result_bydate_df['l_streak'].max()
    maxLStreakDate = result_bydate_df.loc[result_bydate_df['l_streak'] == maxLStreak, 'Date'].values[0]
    avgPLWindays = result_bydate_df[result_bydate_df['dayPL'] > 0]['dayPL'].mean()
    avgPLLossdays = result_bydate_df[result_bydate_df['dayPL'] < 0]['dayPL'].mean()
    winByLossdayagg = abs(round(avgPLWindays / avgPLLossdays, 2))
    avgretPcWindays = result_bydate_df[result_bydate_df['returnPerc'] > 0]['returnPerc'].mean()
    avgretPcLossdays = result_bydate_df[result_bydate_df['returnPerc'] < 0]['returnPerc'].mean()
    winByLossdayaggByretPc = abs(round(avgretPcWindays / avgretPcLossdays, 2))
    '''drawdowns'''
    maxddDuration = result_bydate_df['DDw_streak'].max()
    maxDurwithOutDD = result_bydate_df['DDl_streak'].max()
    maxddDate = result_bydate_df.loc[result_bydate_df['DDw_streak'] == maxddDuration, 'Date'].values[0]
    maxDDPerc = result_bydate_df['drawDowns'].max()
    """collecting aggregated results for all days """
    avgRetPc = result_bydate_df['returnPerc'].mean()
    stdByRetPc = result_bydate_df['returnPerc'].std()
    avgRetDlrs = result_bydate_df['dayPL'].mean()
    stdPLDlrs = result_bydate_df['dayPL'].std()
    grossPL = result_bydate_df['dayPL'].sum()
    totRet, cagr = calcCAGR(result_bydate_df)
    MAR = round(cagr / maxDDPerc, 2)
    Sharpe = (avgRetPc / stdByRetPc) * math.sqrt(252)
    avgCapPerDay = result_bydate_df['startingCapital'].mean()
    totDayswSignals = result_bydate_df.shape[0]

    winDaysPc = ((result_bydate_df[result_bydate_df['dayPL'] > 0].shape[0]) / totDayswSignals) * 100
    results_df = pd.DataFrame(
        [{'avgRetPc': avgRetPc, 'stdByRetPc': stdByRetPc, 'avgRetDlrs': avgRetDlrs, 'stdPLDlrs': stdPLDlrs,
          'grossPL': grossPL, 'maxDDPerc': maxDDPerc, 'maxddDuration': maxddDuration, 'CAGR': cagr, 'MAR': MAR,
          'Sharpe': Sharpe, 'avgCapPerDay': avgCapPerDay, 'totDayswSignals': totDayswSignals,
          'winDaysPc': winDaysPc, 'totRet': totRet,
          'maxLoss': maxLoss, 'maxProfit': maxProfit, 'maxWStreak': maxWStreak, 'maxLStreak': maxLStreak,
          'maxLStreakDate': maxLStreakDate, 'avgPLWindays': avgPLWindays, 'avgPLLossdays': avgPLLossdays,
          'winByLossdayagg': winByLossdayagg,
          'avgretPcWindays': avgretPcWindays, 'avgretPcLossdays': avgretPcLossdays,
          'winByLossdayaggByretPc': winByLossdayaggByretPc,
          'maxDurwithOutDD': maxDurwithOutDD, 'maxddDate': maxddDate

          }], index=None)
    return results_df



def build_results(capital, summary_df):
    '''calculate position sizes and total positions less than max capital allowed for the strategy'''
    summary_df['posSize'] = summary_df['shares'] * summary_df['Nextopen']
    summary_df.sort_values(['Date', 'rocRank'], inplace=True, ascending=True)
    # summary_df.sort_values(['Date', 'rsiRank'], inplace=True, ascending=True)
    # summary_df.sort_values(['Date', 'adxRank'], inplace=True, ascending=True)
    summary_df['cumPosSize'] = summary_df.groupby('Date')['posSize'].cumsum()
    summary_df = summary_df[summary_df['cumPosSize'] <= capital]

    summary_df['posPLdlr'] = (summary_df['tradeSideNextCOPerc'] * summary_df['posSize']) / 100

    """Aggregation By Date and then renaming columns for better readability"""
    result_bydate_df = summary_df.groupby('Date', as_index=False).agg(
        {'symbol': 'count', 'posPLdlr': 'sum', 'SPYClose': 'mean', 'tradeSideNextSPYCOPerc': 'mean',
         'SPYNext_CCPerc': 'mean'
            , 'cumPosSize': 'max', 'shares': 'sum'}).round(decimals=2)
    result_bydate_df.columns = ['Date', 'symbolsCount', 'dayPL', 'SPYClose', 'tradeSideNextSPYCOPerc', 'SPYNext_CCPerc',
                                'startingCapital', 'totalShares']
    result_bydate_df['daySpyCOPL'] = (result_bydate_df['tradeSideNextSPYCOPerc'] * result_bydate_df[
        'startingCapital']) / 100
    result_bydate_df['daySpyCCPL'] = (result_bydate_df['SPYNext_CCPerc'] * result_bydate_df[
        'startingCapital']) / 100

    '''making a copy for benchmark calculations'''
    result_bydate_df_SPYCO = result_bydate_df.copy(deep=True)
    result_bydate_df_SPYCO['dayPL'] = result_bydate_df['daySpyCOPL']
    # result_bydate_df_SPYCC = result_bydate_df.copy(deep = True)
    # result_bydate_df_SPYCC['dayPL'] = result_bydate_df['daySpyCCPL']

    daily_metric_df = build_daily_metrics(result_bydate_df)
    daily_metric_df_SPYCO = build_daily_metrics(result_bydate_df_SPYCO)
    # daily_metric_df_SPYCC = build_daily_metrics(result_bydate_df_SPYCC)

    """Correlation with SPY CO returns with """
    corrWdailySPYCO = result_bydate_df['dayPL'].corr(result_bydate_df_SPYCO['dayPL'])
    corrWdailySPYCORetPc = result_bydate_df['returnPerc'].corr(result_bydate_df_SPYCO['returnPerc'])

    winByLossallSigs = abs(round((summary_df[summary_df['tradeSideNextCOPerc'] > 0]['tradeSideNextCOPerc'].mean()) / (
        summary_df[summary_df['tradeSideNextCOPerc'] < 0]['tradeSideNextCOPerc'].mean()), 2))
    totSymbols = result_bydate_df['symbolsCount'].sum()
    print('corrWdailySPYCO', round(corrWdailySPYCO, 2), 'corrWdailySPYCORetPc', round(corrWdailySPYCORetPc, 2)
          , 'totSymbols', totSymbols, 'winByLossallSigs', winByLossallSigs)

    finalPerf_df = get_final_perf(daily_metric_df)
    finalPerf_dfSPYCO = get_final_perf(daily_metric_df_SPYCO)
    # finalPerf_dfSPYCC = get_final_perf(daily_metric_df_SPYCC)

    print('Strategy Performance Results')
    print(finalPerf_df.round(decimals=2))
    print('SPY CO Performance Returns')
    print(finalPerf_dfSPYCO.round(decimals=2))
    # print('SPY CC Performance Returns')
    # print(finalPerf_dfSPYCC.round(decimals=2))

    return result_bydate_df.round(decimals=2), summary_df.round(decimals=2)




def build_techindicators(df, write_to_db, cutoff_date, hist_end_date2):
    Open = df['Open'].to_numpy(copy=True)
    High = df['High'].to_numpy(copy=True)
    Low = df['Low'].to_numpy(copy=True)
    Close = df['Close'].to_numpy(copy=True)
    Volume = df['Volume']
    start = timer()
    df['atr'] = ta.ATR(High, Low, Close, timeperiod=5)
    # df['natr'] = ta.NATR(High, Low, Close, timeperiod=5)
    df['roc'] = ta.ROC(Close, timeperiod=6)
    # df['rsi'] = ta.RSI(Close, timeperiod=3)
    # df['stddev'] = ta.STDDEV(Close, timeperiod=255)
    # df['sma150'] = ta.SMA(Close, timeperiod=150)
    # df['sma200'] = ta.SMA(Close, timeperiod=200)
    # df['ema150'] = ta.EMA(Close, timeperiod=150)
    # df['ema50'] = ta.EMA(Close, timeperiod=50)
    # df['adx'] = ta.ADX(High, Low, Close, timeperiod=7)
    # df['max'] = ta.MAX(Close, timeperiod=14)
    # df['min'] = ta.MIN(Close, timeperiod=14)
    # df['percBB5'] = BBANDS(df, window=5)
    # df['percBB3'] = BBANDS(df, window=3)
    # df.drop(columns=['upperband','middleband', 'lowerband'], inplace=True)
    # df['obv'] = ta.OBV(Close, Volume)
    df['avgvol10'] = Volume.rolling(window=10).mean()
    df['avgvol50'] = Volume.rolling(window=50).mean()
    df['Nextopen'] = df['Open'].shift(-1)
    df['Nextclose'] = df['Close'].shift(-1)
    # df['Nexthigh'] = df['High'].shift(-1)
    # df['Nextlow'] = df['Low'].shift(-1)
    # df['PrevClose'] = df['Close'].shift(1)
    # This includes SPY values so has to be before any price related filters to ensure SPY is also included in it the df
    df, spy_df = get_FVNextOpen(df)
    # df['SPYsma200'] = ta.SMA(df['SPYClose'].to_numpy(copy=True), timeperiod=200)
    # spy_df = df[df['symbol'] == 'SPY']
    print('Time taken for calculations', timer() - start)

    df = df[(df['Close'] >= 5) & (df['Close'] <= 50) & (df['avgvol50'] > 500000) & (df['Volume'] > 500000) & (
            df['avgvol10'] > 500000) & (
                    df['Date'] > df['validDataDate']) & (df['Date'] < df['endDate']) & (
                    df['Date'] > cutoff_date) & (df['Date'] < hist_end_date2)].round(decimals=2)
    spy_df = spy_df[(spy_df['Date'] > spy_df['validDataDate']) & (spy_df['Date'] < spy_df['endDate']) & (
                spy_df['Date'] > cutoff_date) & (spy_df['Date'] < hist_end_date2)].round(decimals=2)
    # This can be after price filters as we need streaks within the filtered universe. Re-using streaks method in multiple places.
    df['chg4Streak'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    df = streaks(df)
    print('Total filtered records:', df.shape[0])

    if write_to_db == 'Y':
        start = timer()
        conn = sqlite3.connect("NewDatabase.db")
        df.to_sql('All_records_No_Filters', conn, if_exists='replace', index=False)
        conn.close()
        print('time taken for db write', timer() - start)

    return df, spy_df


def main_method(hist_st_date, hist_end_date, hist_end_date2, write_to_db, cutoff_date, symbols_file, trade):
    print('Start Time:', dt.datetime.now(), hist_st_date, hist_end_date, write_to_db, cutoff_date)
    #
    # runMode = input("T for Test, F for Full run -")
    # if runMode == 'F':
    #     hist_st_date = '1994-01-01'
    #     hist_end_date = '2020-04-08'
    #     cutoff_date = '1995-01-01'
    #     write_to_db = 'N'

    symbols_df = get_symbols(symbols_file)
    df = get_stockhistory(hist_st_date, hist_end_date, symbols_df)
    df.sort_values(['symbol', 'Date'], inplace=True, ascending=True)
    df, spy_df = build_techindicators(df, write_to_db, cutoff_date, hist_end_date2)

    # One condition for each strategy
    # conditionS1 = ((df['natr'] > 2) & (df['rsi'] > 90)  & (df['w_streak'] >= 3)& (df['Nextopen'] > df['FVNextOpen']))
    conditionS2 = ( (df['roc'] > 20)  & (df['w_streak'] >= 2)& (df['Nextopen'] > df['FVNextOpen']))

    # conditionL2 = ( (df['Close'] > df['sma200']) & (df['SPYClose'] > df['SPYsma200']) & (df['Nextopen'] < df['FVNextOpen']) & (df['natr'] < 2) )
    # conditionL1 = ((df['Close'] > df['sma150']) & (df['Nextopen'] < df['FVNextOpen']) & (df['natr'] > 3) & (
    #             df['roc'] < -10))

    summary_df = df[conditionS2].round(decimals=2)
    # summary_df = df.copy(deep = True)
    print('Long Trend Low Volatility days tested:', df['Date'].unique().shape[0])
    print('Total Symbols with Setup:', summary_df.shape[0])
    summary_df = get_NextTradesideCOPerc(summary_df, trade)

    AvgPerfSymbolsAll = round(summary_df['tradeSideNextCOPerc'].mean(), 2)
    TotalSetupSymbols = summary_df.shape[0]
    WinPerc = round((summary_df[summary_df['tradeSideNextCOPerc'] > 0].shape[0] / summary_df.shape[0]) * 100)
    WinByLoss = abs(round((summary_df[summary_df['tradeSideNextCOPerc'] > 0]['tradeSideNextCOPerc'].mean()) / (
        summary_df[summary_df['tradeSideNextCOPerc'] < 0]['tradeSideNextCOPerc'].mean()), 2))

    print('AvgPerfSymbolsAll:', AvgPerfSymbolsAll, 'TotalSetupSymbols:', TotalSetupSymbols, 'WinPerc:', WinPerc,
          'WinByLoss:', WinByLoss)

    capital = 100000
    maxCapPerPos = 10000
    maxRiskPerPos1 = 500
    maxRiskPerPos2 = 1000

    """Ranking starts here"""


    summary_df.sort_values(['Date'], inplace=True, ascending=True)
    # summary_df['adxRank'] = summary_df.groupby('Date')['adx'].rank('dense', ascending = False)
    # summary_df = summary_df[summary_df['adxRank']<11]
    summary_df['rocRank'] = summary_df.groupby('Date')['roc'].rank('dense', ascending=False)
    # summary_df['rocRank'] = summary_df.groupby('Date')['roc'].rank('dense', ascending=True)
    # summary_df['rsiRank'] = summary_df.groupby('Date')['rsi'].rank('dense', ascending=True)
    summary_df['shares'] = (maxCapPerPos / summary_df['FVNextOpen']).apply(np.floor)
    # summary_df['rsiRank'] = summary_df.groupby('Date')['rsi'].rank('dense', ascending = True)
    # summary_df = summary_df[summary_df['rsiRank']<11]
    """Ranking ends here"""

    """SPY max DDs for both CO and CC. Should match with public data"""
    # spysummary_df, spy_df = get_SPYDDs(spy_df)
    # print(spysummary_df.round(decimals = 2))

    """Fixed Capital Scenario"""
    summary_dfFC = summary_df.copy(deep=True)
    # summary_dfFC['shares'] = (maxCapPerPos / summary_df['Close']).apply(np.floor)
    result_bydate_dfFC, summary_dfFC_o = build_results(capital, summary_dfFC)
    ''''Max Risk 1'''
    summary_dfR1 = summary_df.copy(deep=True)
    summary_dfR1['shares'] = np.minimum((maxRiskPerPos1/summary_df['atr']).apply(np.floor), summary_df['shares'])
    result_bydate_dfR1, summary_dfR1_o = build_results(capital, summary_dfR1)
    ''''Max Risk 2'''
    summary_dfR2 = summary_df.copy(deep=True)
    summary_dfR2['shares'] = np.minimum((maxRiskPerPos2/summary_df['atr']).apply(np.floor), summary_df['shares'])
    result_bydate_dfR2, summary_dfR2_o = build_results(capital, summary_dfR2)

    spy_df['startingCapital'] = capital
    spy_df['dayPL'] = (spy_df['SPYNext_CCPerc'] * capital) / 100
    daily_metrics_df_SPYCCAllDays = build_daily_metrics(spy_df)
    finalPerf_dfSPYCCAllDays = get_final_perf(daily_metrics_df_SPYCCAllDays)
    print("SPY BENCHMARK METRICS CC ")
    print(finalPerf_dfSPYCCAllDays.round(decimals=2))

    # write_results_to_db(result_bydate_dfFC, spy_df, summary_dfFC_o)
    # write_results_to_db(result_bydate_dfR1, spy_df, summary_dfR1_o)
    write_results_to_db(result_bydate_dfR2, spy_df, summary_dfR2_o)

    #
    # summary_dfRisk2 = summary_df[summary_df['cumPosSizeRisk2'] < capital].copy(deep = True)
    # result_bydate_dfRisk2 = build_results(capital, summary_dfFC)

    plot(result_bydate_dfFC)

    print('End Time:', dt.datetime.now(), hist_st_date, hist_end_date, write_to_db, cutoff_date)


# main_method('1994-01-01', '2020-04-06', 'Distinct_Yahoo_DB_Unique2.csv', 'N', '1994-01-01')
# main_method('2019-01-01', '2020-04-08', 'Y', '2020-01-01', 'Yahoo_MSTR2_Symbols_Dates.csv', 'Long')
# main_method('1994-01-01', '2019-07-25', 'N', '1995-01-01', 'Yahoo_MSTR2_Symbols_Dates.csv', 'Long')
main_method('1994-01-01', '2020-04-08', '2020-04-06', 'N', '1994-12-29', 'Yahoo_MSTR2_Symbols_Dates.csv', 'Short')
# main_method('2018-01-01', '2020-04-08', '2020-04-06', 'Y', '2019-01-01', 'Yahoo_MSTR2_Symbols_Dates.csv', 'Short')
"""Hist St Date, Hist End Date, Hist End Date2(this is reqd because of the SQL query taking Hist End Date-1 and next day calculations), 
DB Write, cutoff date ( because of indicators nan values), file name with valid data dates, tradeside"""
#
