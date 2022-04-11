# -*- coding: utf-8 -*-

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

df = yf.download('BTC-USD', start='2020-01-10', end='2022-04-10', progress=False)
df = df.reset_index()

df = df[['Date','Close']]
df = df.rename(columns={'Date':'ds','Close':'y'})
df.tail(10)

fbp = Prophet(daily_seasonality=True)
fbp.fit(df)
future = fbp.make_future_dataframe(periods=365)
forecast = fbp.predict(future)
plot_plotly(fbp,forecast,figsize=(1280,786))
