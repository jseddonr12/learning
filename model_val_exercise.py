import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

home_data_filepath = '/Users/jaredroberts/Downloads/train.csv'
home_data = pd.read_csv(home_data_filepath)

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = home_data.SalePrice
X = home_data[features]

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

print('First in-sample prediction: ', iowa_model.predict(X.head()))
print('Actual target value for homes: ', y.head().tolist())

# tolist simply changes an array into a list

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Now using training data

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
print('Predictions with training data: ', iowa_model.predict(X.head()))
print('Actual target value for homes: ', y.head().tolist())

print('The mean absolute error is: ', mean_absolute_error(val_y, val_predictions))

# UNDERFITTING AND OVERFITTING 
# max_leaf_nodes = argument to control over/under fitting

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_value = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_value)
    return mae

# This function helps us calculate mae for different values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d  \t\t Mean Absolute Error: %d' %(max_leaf_nodes, my_mae))

