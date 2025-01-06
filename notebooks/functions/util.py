import os
import datetime
import time
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path



# we use that
def trigger_request(url:str):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the JSON content from the response
        data = response.json()
    else:
        print("Failed to retrieve data. Status Code:", response.status_code)
        raise requests.exceptions.RequestException(response.status_code)

    return data


def get_stock_price(symbol: str, ALPHAVANTAGE_API_KEY: str):
    """
    Returns DataFrame with stock price as dataframe
    """
    # The API endpoint URL
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"

    # Make a GET request to fetch the data from the API
    data = trigger_request(url)

    # Extract the latest date
    latest_date = data["Meta Data"]["3. Last Refreshed"]

    # Extract the "close" price for the latest date
    latest_close_price = data["Time Series (Daily)"][latest_date]["4. close"]

    sp_df = pd.DataFrame()
    
    sp_df['timestamp'] = [latest_date]
    sp_df['timestamp'] = pd.to_datetime(sp_df['timestamp'])
    
    sp_df['price'] = [latest_close_price]
    sp_df['price'] = sp_df['price'].astype('double')

    return sp_df


def get_crypto_price(symbol: str, ALPHAVANTAGE_API_KEY: str):
    """
    Returns DataFrame with stock price as dataframe
    """
    # The API endpoint URL
    url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={ALPHAVANTAGE_API_KEY}"
    # Make a GET request to fetch the data from the API
    data = trigger_request(url)

    # Extract the latest date
    latest_date = data['Meta Data']['6. Last Refreshed']
    latest_date_only = datetime.datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S').date()
    latest_close_price = data['Time Series (Digital Currency Daily)'][str(latest_date_only)]['4. close']

    sp_df = pd.DataFrame()
    
    sp_df['timestamp'] = pd.to_datetime([latest_date_only])
    
    sp_df['price'] = [latest_close_price]
    sp_df['price'] = sp_df['price'].astype('double')

    return sp_df

def get_sentiment_score(symbol: str, ALPHAVANTAGE_API_KEY: str):
    """
    Returns DataFrame with sentiment score as dataframe
    """
    # The API endpoint URL
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    yesterday_formatted = yesterday.strftime('%Y%m%d') + 'T0000'
    today_formatted = yesterday.strftime('%Y%m%d') + 'T2359'
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&limit=1000&time_from={yesterday_formatted}&time_to={today_formatted}&tickers={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    
    # Make a GET request to fetch the data from the API
    response = requests.get(url)
    sentim_data = response.json()
    
    sentim_df = pd.read_json(json.dumps(sentim_data))
    
    # Checks if new data available
    if 'feed' in sentim_data and sentim_data['feed']:
        sentim_df = pd.read_json(json.dumps(sentim_data))
        sentim_df["timestamp"] = sentim_df["feed"].apply(lambda x: x.get("time_published"))
        sentim_df["overall_sentiment_score"] = sentim_df["feed"].apply(lambda x: x.get("overall_sentiment_score"))
        sentim_df.drop(columns=['feed','items','sentiment_score_definition','relevance_score_definition'], inplace=True)
        sentim_df["timestamp"] = pd.to_datetime(sentim_df["timestamp"]).dt.date
        sentim_df = sentim_df.groupby("timestamp").mean().reset_index()
        return sentim_df
    else:
        # Return a score of 0 if no new data available
        sentim_df = pd.DataFrame()
        sentim_df['timestamp'] = [yesterday.date()]
        sentim_df['overall_sentiment_score'] = [0]
        return sentim_df

def plot_stock_price_forecast(df: pd.DataFrame, file_path: str, name: str, hindcast=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    day = pd.to_datetime(df['date']).dt.date
    # Plot each column separately in matplotlib
    ax.plot(day, df['predicted_price'], label='Predicted Price', color='blue', linewidth=2, marker='o', markersize=5, markerfacecolor='red')

    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_title(f"Stock Price Prediction for {name}")
    ax.set_ylabel('Price')

    colors = ['green', 'yellow', 'orange', 'red']
    labels = ['Very Low', 'Low', 'Medium', 'High']
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
    for color, (start, end) in zip(colors, ranges):
        ax.axhspan(start, end, color=color, alpha=0.3)

    # Add a legend for the different Price Categories
    patches = [Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}") for i in range(len(colors))]
    legend1 = ax.legend(handles=patches, loc='upper right', title="Price Categories", fontsize='x-small')

    # Aim for ~10 annotated values on x-axis, will work for both forecasts and hindcasts
    if len(df.index) > 11:
        every_x_tick = len(df.index) / 10
        ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast:
        ax.plot(day, df['price'], label='Actual Price', color='black', linewidth=2, marker='^', markersize=5, markerfacecolor='grey')
        legend1 = ax.legend(loc='upper left', fontsize='x-small')
        ax.add_artist(legend1)

    # Ensure everything is laid out neatly
    plt.tight_layout()

    # Save the figure, overwriting any existing file with the same name
    plt.savefig(file_path)
    return plt

# def plot_stockpred_forecast(city: str, street: str, df: pd.DataFrame, file_path: str, hindcast=False):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     day = pd.to_datetime(df['date']).dt.date
#     # Plot each column separately in matplotlib
#     ax.plot(day, df['price'], label='Predicted PM2.5', color='red', linewidth=2, marker='o', markersize=5, markerfacecolor='blue')

#     # Set the y-axis to a logarithmic scale
#     ax.set_yscale('log')
#     ax.set_yticks([0, 10, 25, 50, 100, 250, 500])
#     ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
#     ax.set_ylim(bottom=1)

#     # Set the labels and title
#     ax.set_xlabel('Date')
#     ax.set_title(f"PM2.5 Predicted (Logarithmic Scale) for {city}, {street}")
#     ax.set_ylabel('PM2.5')

#     colors = ['green', 'yellow', 'orange', 'red', 'purple', 'darkred']
#     labels = ['Good', 'Moderate', 'Unhealthy for Some', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
#     ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
#     for color, (start, end) in zip(colors, ranges):
#         ax.axhspan(start, end, color=color, alpha=0.3)

#     # Add a legend for the different Air Quality Categories
#     patches = [Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}") for i in range(len(colors))]
#     legend1 = ax.legend(handles=patches, loc='upper right', title="Air Quality Categories", fontsize='x-small')

#     # Aim for ~10 annotated values on x-axis, will work for both forecasts ans hindcasts
#     if len(df.index) > 11:
#         every_x_tick = len(df.index) / 10
#         ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

#     plt.xticks(rotation=45)

#     if hindcast == True:
#         ax.plot(day, df['pm25'], label='Actual PM2.5', color='black', linewidth=2, marker='^', markersize=5, markerfacecolor='grey')
#         legend2 = ax.legend(loc='upper left', fontsize='x-small')
#         ax.add_artist(legend1)

#     # Ensure everything is laid out neatly
#     plt.tight_layout()

#     # # Save the figure, overwriting any existing file with the same name
#     plt.savefig(file_path)
#     return plt


def delete_feature_groups(fs, name):
    try:
        for fg in fs.get_feature_groups(name):
            fg.delete()
            print(f"Deleted {fg.name}/{fg.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature group found")

def delete_feature_views(fs, name):
    try:
        for fv in fs.get_feature_views(name):
            fv.delete()
            print(f"Deleted {fv.name}/{fv.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature view found")

def delete_models(mr, name):
    models = mr.get_models(name)
    if not models:
        print(f"No {name} model found")
    for model in models:
        model.delete()
        print(f"Deleted model {model.name}/{model.version}")

def delete_secrets(proj, name):
    secrets = secrets_api(proj.name)
    try:
        secret = secrets.get_secret(name)
        secret.delete()
        print(f"Deleted secret {name}")
    except hopsworks.client.exceptions.RestAPIError:
        print(f"No {name} secret found")



def secrets_api(proj):
    host = "c.app.hopsworks.ai"
    # api_key = os.environ.get('HOPSWORKS_API_KEY')
    api_key = os.environ["HOPSWORKS_API_KEY"]
    conn = hopsworks.connection(host=host, project=proj, api_key_value=api_key)
    return conn.get_secrets_api()

# we use this
def check_file_path(file_path):
    my_file = Path(file_path)
    if my_file.is_file() == False:
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")

def backfill_predictions_for_monitoring(weather_fg, air_quality_df, monitor_fg, model):
    features_df = weather_fg.read()
    features_df = features_df.sort_values(by=['date'], ascending=True)
    features_df = features_df.tail(10)
    features_df['predicted_pm25'] = model.predict(features_df[['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max', 'wind_direction_10m_dominant']])
    df = pd.merge(features_df, air_quality_df[['date','pm25','street','country']], on="date")
    df['days_before_forecast_day'] = 1
    hindcast_df = df
    df = df.drop('pm25', axis=1)
    monitor_fg.insert(df, write_options={"wait_for_job": True})
    return hindcast_df
