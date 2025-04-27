# Rohlik Sales Forecasting Challenge

This repo contains my solution to the Rohlik Sales Forecasting Challenge, hosted by [Rohlik Group](https://www.rohlik.group/) on [Kaggle](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/overview). Rohlik Group is an innovative, e-grocery store in Europe, operating out of warehouses in Czechia, Germany, and Hungary.

The goal of the competition was to predict the sales of over a thousand products over a future window of 14 days.

## My solution

I used a single LightGBM gradient-boosted tree model as my foundation. Besides being the go-to choice for tabular data, there are some other strong reasons for choosing this model. First of all, the data is very heterogenous. We have product information, product relationships, geographic data, calendar data, order application data, etc. Any model that tries to directly combine numerical data with different units is a poor choice; all we care about is the relative ordering that a numerical column provides.

At the same time, it's absolutely necessary to integrate temporal smearing into each datapoint. A datapoint unaware of its sales history is missing incredibly useful information. It's common for forecasters to include lagged features into the model (i.e., sales yesterday or sales a week ago). I deviated from this approach. For one, adding lots of extra features exposes tree-based model to severe overfitting. Tree-based models are already notorious for this. 

I used a tree-based model because of the heterogeneity of the data, but historical sales for different lags is *homogeneous* data. This makes linear regression (by product id) a solid choice for synthesizing this multi-scale temporal data. Further, we can include other temporal data with the same units, like different moving averages, to make the regression stronger. In fact, combining these lags (based on partial autocorrelation) with moving averages means we're fitting a stronger version of an ARIMA model. Synthesizing the temporal information prior to plugging it into the GBT model gives us both better performance and improved robustness.