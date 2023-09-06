from turtle import update
import pandas as pd

drugs = pd.read_csv(r'/Users/jaredroberts/Desktop/VSRR_Provisional_Drug_Overdose_Death_Counts.csv')
tsla = pd.read_csv(r'/Users/jaredroberts/Desktop/tsla_raw_data.csv')

### DATA SUMMARY FUNCTIONS AND MAPS

# describe gives up mean, std, min, etc.

print(drugs.Year.describe())
drugs.describe()

# Mean

print(drugs.Year.mean())

# Unique -- Provides unique data points in the given column

print(drugs.Indicator.unique())

# Value_counts

print(drugs.Year.value_counts())

### MAPS
# Mapping a set of values means to take one set of values and "map" them onto another set values
# Map() takes a single value from the Series and returns the transformed value

tsla_avg_close = tsla.close.mean()
remean_tsla_close = tsla.close.map(lambda p: p - tsla_avg_close)

# Another method of mapping is apply()
# Equivalent of Map() -- only difference is map is for Series and apply is for DataFrame
# apply calls a custom method on each row

def remean_tsla(row):
    row.close = row.close - tsla_avg_close
    return row

updated_table = tsla.apply(remean_tsla, axis='columns')

# Kaggle Project

print(tsla.head())
print(tsla.avg_vol_20d.median())

# Find the most volatile day of trading -- widest spread on high and low

widest_spread = (tsla.high / tsla.low).idxmax()
volatility = tsla.loc[widest_spread, 'date']
print(widest_spread)
print(volatility)
