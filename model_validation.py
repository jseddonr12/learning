import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


home_data_filepath = '/Users/jaredroberts/Downloads/train.csv'
home_data = pd.read_csv(home_data_filepath)

tsla_filepath = '/Users/jaredroberts/Desktop/tsla_raw_data.csv'
tsla = pd.read_csv(tsla_filepath)

melbourne_filepath = '/Users/jaredroberts/Downloads/melb_data.csv'
melb = pd.read_csv(melbourne_filepath)
filtered_melb_data = melb.dropna(axis=0)

# Predication error = actual value - predicted value
# Mean Absolute Error is when we take the absolute value of each error and then take the average
# of those absolute errors
# This is simply a measure of model quantity
# All a Mean Absolute Error metric is stating is "on average, our predications are off by X"

y = filtered_melb_data.Price
melbourne_features = ['Price']
X = filtered_melb_data[melbourne_features]
print(X.describe())
melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(X, y)

r_score = melb_model.score(X, y)
print(r_score)

predicted_home_prices = melb_model.predict(X)
print(predicted_home_prices)

mav = mean_absolute_error(y, predicted_home_prices)
print(mav)

# Validation Data 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

# What is happening here is the functions are splitting the data into training and validation data
# for both features and prediciton target

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
