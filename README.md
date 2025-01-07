# StockMarketPrediciton

This project uses the **S&P500** price and a *sentiment score*, which describes the sentiment of news articles about the stock, to predict the share price of **Spotify** and **Google**, and the price of **Bitcoin**. 
Data is fetched from the [`AlphaVantage`][1]. [Hopsworks][2]  is our feature-store of choice, to store the feature groups and models.

* Notebook 1 covers the backfill of historical data into feature groups for the tradables. It uses the [`AlphaVantage`][1] API to source the historical data and the sentiment scores regarding the tradables. Then it creates relevant dataframes and pushes the feature groups to [Hopsworks][2].
* Notebook 2 runs daily, gets today's stock values and news sentiments from the API and appends it into the existing feature groups.
* Notebook 3 creates feature views from the created feature groups, preprocesses the data in them to create training data for the ML models. Then it trains, validates a [XGBRegressor][3] and finally saves it to [Hopsworks][2].
* Notebook 4 is the inference pipeline that runs daily to predict the stock price after the latest data is appended to the feature groups.

The forecast of the stock price as well as a hindcast to check how well the models predict can be found [here](https://theresahoesl.github.io/StockMarketPrediction/ "Link to Output Dashboard").

[1]: https://alphavantage.co "Link to AlphaVantage Website"
[2]: https://www.hopsworks.ai/ "Link to Hopsworks Website"
[3]: https://xgboost.readthedocs.io/en/stable/python/python_api.html "XGBoost Python API Documentation"