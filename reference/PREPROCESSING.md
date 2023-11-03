# Data Preprocessing Guide

This document provides step-by-step instructions on how to preprocess data using the provided Python code. The code is designed to prepare a dataset from a CSV file named 'pacific.csv'.

## Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Step 2: Load the Dataset
```
df = pd.read_csv('pacific.csv')
```

## Step 3: Visualize Missing Data
```
plt.figure(figsize=(15, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Before Preprocessing')
plt.show()
```

## Step 4: Data Transformation

### Convert Latitude and Longitude
```
def convert_lat_long(val):
    direction = val[-1]
    val = float(val[:-1])
    if direction in ['S', 'W']:
        val *= -1
    return val

df['Latitude'] = df['Latitude'].apply(convert_lat_long)
df['Longitude'] = df['Longitude'].apply(convert_lat_long)
```

### Combine Date and Time into DateTime
```
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df['Time'] = df['Time'].apply(lambda x: "{:04d}".format(x))
df['Time'] = pd.to_timedelta(df['Time'].str[:2] + ':' + df['Time'].str[2:] + ':00')
df['DateTime'] = df['Date'] + df['Time']
df drop(['Date', 'Time'], axis=1, inplace=True)
```

### Ensure Numeric Data Types
```
numeric_cols = ['Maximum Wind', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW', 'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
```
## Step 5: Handling Missing Data
```
missing_data_threshold = 0.1  # 10% threshold for removal
imputation_threshold = 0.8   # 80% threshold for imputation
```

### Calculate Missing Data Percentage
```
missing_percentage_by_id = df.isnull().groupby(df['ID']).mean().max(axis=1)
```

### Plot Missing Data Distribution
```
plt.figure(figsize=(15, 6))
missing_percentage_by_id.hist(bins=50)
plt.title('Histogram of Missing Data Percentage by Hurricane')
plt.xlabel('Percentage of Missing Data')
plt.ylabel('Count of Hurricanes')
plt.axvline(x=missing_data_threshold, color='red', linestyle='dashed', linewidth=2)
plt.text(missing_data_threshold+0.01, 5, '10% Threshold', rotation=0, color='red')
plt.show()
```

### Identify IDs for Removal
```
ids_to_remove = missing_percentage_by_id[missing_percentage_by_id > missing_data_threshold].index
```

### Remove Data with High Missing Values
```
df_cleaned = df[~df['ID'].isin(ids_to_remove)].copy()
```

### Impute Missing Values
```
def impute_missing(df, column):
    mean_value = df[column].mean()
    
    if df[column].isnull().mean() < missing_data_threshold:
        df_cleaned[column].fillna(mean_value, inplace=True)
```

### Apply Imputation to Numeric Columns
```
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    impute_missing(df, col)
```

## Step 6: Data Visualization
```
plt.figure(figsize=(15, 6))
sns.histplot(df['Maximum Wind'], kde=True, stat="density", linewidth=0, color='blue', label='Before Imputation')
sns.histplot(df_cleaned['Maximum Wind'], kde=True, stat="density", linewidth=0, color='orange', label='After Imputation')
plt.title('Distribution of Maximum Wind Before and After Imputation')
plt.legend()
plt.show()
```

## Step 7: Conclusion
```
print(df_cleaned.head(), df_cleaned.size)
```
