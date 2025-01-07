# StockMarketPrediciton

This project uses the S&P500 price and a "sentiment score", which describes the sentiment of news articles about the stock, to predict the share price of Spotify and Google, and the price of bitcoin. 
Data is fetched from alphavantage.co. Hopsworks is used to store the data.
* Notebook 1 covers the backfill of historical data into feature groups in Hopsworks
* Notebook 2 runs daily, gets todays prices and sentiment scores and stores it into the existing feature groups
* Notebook 3 trains, validates a XGBRegressor and finally saves it
* Notebook 4 runs daily to predict the stock price

The forecast of the stock price as well as a hindcast to check how well the models predict can be found here: https://theresahoesl.github.io/StockMarketPrediciton/
