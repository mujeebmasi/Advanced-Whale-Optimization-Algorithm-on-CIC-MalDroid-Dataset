import pandas as pd
import numpy as np

df = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv', low_memory=False)
print("Shape:", df.shape)
print("\nFirst column name:", df.columns[0])
print("\nFirst 5 values of first column:")
print(df.iloc[:5, 0])
print("\nLast column name:", df.columns[-1])
print("\nFirst 5 values of last column:")
print(df.iloc[:5, -1])
print("\nData types (first 3 cols):")
print(df.dtypes[:3])
print("\nSample of first row values (first 10 cols):")
print(df.iloc[0, :10].tolist())
print("\nUnique values in last column:")
print(df.iloc[:, -1].value_counts().head(10))
print("\nUnique values in first column (sample):")
print(df.iloc[:, 0].head(10).tolist())