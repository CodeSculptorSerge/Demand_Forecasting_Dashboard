import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Загрузка данных
df = pd.read_csv('input_data/train.csv')

# Преобразование столбца 'date' в формат datetime
df['date'] = pd.to_datetime(df['date'])

# Уменьшение памяти за счет преобразования типов данных
df['sales'] = df['sales'].astype('int32')
df['store'] = df['store'].astype('category')
df['item'] = df['item'].astype('category')

# Группировка данных по дате
daily_sales = df.groupby('date')['sales'].sum().asfreq('D')

# Прогнозирование с использованием всех данных для будущих 90 дней
train = daily_sales

# Прогнозирование с ARIMA
model_arima = ARIMA(train, order=(5,1,0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=90)

# Прогнозирование с Prophet
prophet_df = train.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future_prophet = model_prophet.make_future_dataframe(periods=90)
forecast_prophet = model_prophet.predict(future_prophet)

# Создание приложения Dash
app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Прогнозирование спроса с использованием моделей временных рядов'),

    dcc.Graph(
        id='sales-graph',
        figure=go.Figure(
            data=[
                go.Scatter(x=daily_sales.index, y=daily_sales.values, name='Ежедневные продажи', line=dict(color='blue')),
                go.Scatter(x=pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=90), 
                           y=forecast_arima.values, name='Прогноз ARIMA', line=dict(color='red')),
                go.Scatter(x=forecast_prophet['ds'][-90:], y=forecast_prophet['yhat'][-90:], name='Прогноз Prophet', line=dict(color='green'))
            ],
            layout=go.Layout(
                title='Дэшборд прогнозов продаж',
                xaxis_title='Дата',
                yaxis_title='Продажи',
                showlegend=True
            )
        )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

