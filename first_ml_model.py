import pandas as pd
from sklearn.tree import DecisionTreeRegressor

tsla_filepath = '/Users/jaredroberts/Desktop/tsla_raw_data.csv'
tsla = pd.read_csv(tsla_filepath)

melbourne_filepath = '/Users/jaredroberts/Downloads/melb_data.csv'
melb = pd.read_csv(melbourne_filepath)

home_data_filepath = '/Users/jaredroberts/Downloads/train.csv'
home_data = pd.read_csv(home_data_filepath)

price_min = tsla.low.idxmin()
print(price_min)
date_of_lowest = tsla.loc[price_min, 'date']

#recall that map() is for Series and apply() is for DataFrame

lowest_price = tsla.groupby('volume').low.min()
print(lowest_price)

description = tsla.describe()
print(description)

melb_summary = melb.describe()
print(melb_summary)

avg_land_size = melb.Landsize.mean()
print(round(avg_land_size, 0))

newest_home = melb.YearBuilt.map(lambda p: 2023 - p)
answer = newest_home.min()
print(answer)

see_columns = melb.columns
print(see_columns)

# Y is the conventional notation for the Prediction Target
# X is the conventional notation for the Features -- columns inputted into model

y = melb.Price
melbourne_features = ['Price', 'Distance', 'Landsize']
X = melb[melbourne_features]
X.describe()


melb_model = DecisionTreeRegressor(random_state=1)
melb_model.fit(X, y)

print('Making predictions for the following 5 houses: ')
print(X.head())
print('The predictions are: ')
print(melb_model.predict(X.head()))

# Problem set

print(home_data.columns)

y = home_data.SalePrice
home_features = ['LotArea', 'YearBuilt', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[home_features]

print(X.describe())

home_model = DecisionTreeRegressor(random_state=1)
home_model.fit(X, y)
predictions = home_model.predict(X)
print(predictions)
