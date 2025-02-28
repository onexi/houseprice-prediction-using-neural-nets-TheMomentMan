import pandas as pd

df = pd.read_csv('train.csv')

# Compute the number of NaN values per column and reset the index to get a DataFrame.
nan_counts = df.isna().sum().reset_index()
nan_counts.columns = ['Column Name', 'NaNCount']

#sort the dataframe by NanCount in descending order
nan_counts=nan_counts.sort_values(by='NaNCount',ascending=False).reset_index(drop=True)

print(nan_counts)