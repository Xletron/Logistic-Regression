import pandas as pd

# read in the original CSV file
df = pd.read_csv('data.csv')

# drop rows containing "Missing" or "Not Asked on This Form"
df = df[~df.isin(['Missing', 'Not Asked on This Form']).any(axis=1)]

# write cleaned data to a new CSV file
df.to_csv('cleaned_file.csv', index=False)