import vectorbt as vbt
import numpy as np
import talib

import requests                    # for "get" request to API
import json                        # parse json into a list
import pandas as pd                # working with data frames
import datetime as dt              # working with dates
import matplotlib.pyplot as plt    # plot data

import datetime as dt

# UT Bot Parameters
SENSITIVITY = 1
ATR_PERIOD = 10
 
# Ticker and timeframe
TICKER = "BTCUSDT"
INTERVAL = "1d"
 
# Backtest start/end date
startTime = dt.datetime(2017,8,17)
endTime   = dt.datetime.now()

# Get data from Binance
def get_binance_bars(symbol, interval, startTime, endTime):
 
    url = "https://api.binance.com/api/v3/klines"
 
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'
 
    req_params = {"symbol" : symbol, 'interval' : interval, 'startTime' : startTime, 'endTime' : endTime, 'limit' : limit}
 
    df = pd.DataFrame(json.loads(requests.get(url, params = req_params).text))
 
    if (len(df.index) == 0):
        return None
     
    df = df.iloc[:, 0:6]
    df.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
 
    df.Open      = df.Open.astype("float")
    df.High      = df.High.astype("float")
    df.Low       = df.Low.astype("float")
    df.Close     = df.Close.astype("float")
    df.Volume    = df.Volume.astype("float")
    
    df['adj_close'] = df['Close']
     
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
 
    return df
pd_data = get_binance_bars(TICKER, INTERVAL, startTime, endTime)

# Compute ATR And nLoss variable
pd_data["xATR"] = talib.ATR(pd_data["High"], pd_data["Low"], pd_data["Close"], timeperiod=ATR_PERIOD)
pd_data["nLoss"] = SENSITIVITY * pd_data["xATR"]
 
#Drop all rows that have nan, X first depending on the ATR preiod for the moving average
pd_data = pd_data.dropna()
pd_data = pd_data.reset_index()

# Function to compute ATRTrailingStop
def xATRTrailingStop_func(close, prev_close, prev_atr, nloss):
    if close > prev_atr and prev_close > prev_atr:
        return max(prev_atr, close - nloss)
    elif close < prev_atr and prev_close < prev_atr:
        return min(prev_atr, close + nloss)
    elif close > prev_atr:
        return close - nloss
    else:
        return close + nloss
 
# Filling ATRTrailingStop Variable
pd_data["ATRTrailingStop"] = [0.0] + [np.nan for i in range(len(pd_data) - 1)]
 
for i in range(1, len(pd_data)):
    pd_data.loc[i, "ATRTrailingStop"] = xATRTrailingStop_func(
        pd_data.loc[i, "Close"],
        pd_data.loc[i - 1, "Close"],
        pd_data.loc[i - 1, "ATRTrailingStop"],
        pd_data.loc[i, "nLoss"],
    )


# Calculating signals
ema = vbt.MA.run(pd_data["Close"], 1, short_name='EMA', ewm=True)
 
pd_data["Above"] = ema.ma_crossed_above(pd_data["ATRTrailingStop"])
pd_data["Below"] = ema.ma_crossed_below(pd_data["ATRTrailingStop"])
 
pd_data["Buy"] = (pd_data["Close"] > pd_data["ATRTrailingStop"]) & (pd_data["Above"]==True)
pd_data["Sell"] = (pd_data["Close"] < pd_data["ATRTrailingStop"]) & (pd_data["Below"]==True)


# Run the strategy
pf = vbt.Portfolio.from_signals(
    pd_data["Close"],
    entries=pd_data["Buy"],
    short_entries=pd_data["Sell"],
    upon_opposite_entry='ReverseReduce', 
    freq = "d"
)
print(pf.stats())

# Show the chart 
# fig = pf.plot(subplots=['orders','trade_pnl','cum_returns'])
# ema.ma.vbt.plot(fig=fig)
# fig.show()