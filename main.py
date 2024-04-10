import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Video Games Sales.csv')

df = data
#print(df.head(5))

#print(df.info)
df = df.dropna(axis=0) #Drop null values
#print(df.info)

print(df.columns)
print(df['Platform'].unique())
print(df['Genre'].unique())
