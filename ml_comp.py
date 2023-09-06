from turtle import home
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

home_data_filepath = '/Users/jaredroberts/Downloads/train.csv'
home_data = pd.read_csv(home_data_filepath)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = home_data.SalePrice
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# Now train the data on the actual data, not the training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)
test_X = home_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': home_data.Id, 'SalePrice': test_preds})
print(output)
output.to_csv('submission.csv', index=False)
