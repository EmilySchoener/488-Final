import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from tabulate import tabulate

# reads initial file
data = pd.read_csv('Video Games Sales.csv')
df = data
df = df.dropna(axis=0)  # Drop null values

# removes all columns except platform, genre, and global sales
df = df.drop(labels='index', axis=1)
df = df.drop(labels='Rank', axis=1)
df = df.drop(labels='Game Title', axis=1)
df = df.drop(labels='Year', axis=1)
df = df.drop(labels='Publisher', axis=1)
df = df.drop(labels='North America', axis=1)
df = df.drop(labels='Europe', axis=1)
df = df.drop(labels='Japan', axis=1)
df = df.drop(labels='Rest of World', axis=1)
df = df.drop(labels='Global', axis=1)

# print(df.columns)
# print(df['Platform'].unique())
# print(df['Genre'].unique())


# converts values to an index for fitting / calculations
def game_scoring(value):
    if value >= 80:
        return 1
    else:
        return 0


# converts values to an index for fitting / calculations
def platform(value):
    if value == 'Wii':
        return 0
    elif value == 'NES':
        return 1
    elif value == 'GB':
        return 2
    elif value == 'DS':
        return 3
    elif value == 'PS2':
        return 4
    elif value == 'SNES':
        return 5
    elif value == 'X360':
        return 6
    elif value == 'GBA':
        return 7
    elif value == 'PS3':
        return 8
    elif value == 'N64':
        return 9
    elif value == 'PC':
        return 10
    elif value == 'PS':
        return 11
    elif value == 'XB':
        return 12
    elif value == '3DS':
        return 13
    elif value == 'PSP':
        return 14
    elif value == 'GC':
        return 15
    elif value == 'GEN':
        return 16
    elif value == 'DC':
        return 17
    elif value == 'SAT':
        return 18
    elif value == 'WiiU':
        return 19
    elif value == 'SCD':
        return 20
    else:
        return 21


# converts values to an index for fitting / calculations
def genre(value):
    if value == 'Sports':
        return 0
    elif value == 'Platform':
        return 1
    elif value == 'Racing':
        return 2
    elif value == 'Puzzle':
        return 3
    elif value == 'Misc':
        return 4
    elif value == 'Shooter':
        return 5
    elif value == 'Simulation':
        return 6
    elif value == 'Role-Playing':
        return 7
    elif value == 'Action':
        return 8
    elif value == 'Fighting':
        return 9
    elif value == 'Adventure':
        return 10
    elif value == 'Strategy':
        return 11


# sets up target and data
target_df = pd.DataFrame(data=df, columns=['Review'])
target_df['Review'] = target_df['Review'].apply(game_scoring)
# df = df.drop(labels='Review', axis=1)
data_df = pd.DataFrame(data=df, columns=df.columns)
data_df['Platform'] = data_df['Platform'].apply(platform)
data_df['Genre'] = data_df['Genre'].apply(genre)
data_df['Review'] = data_df['Review'].apply(game_scoring)

# Sets up Naive Bayes calculations
prior = target_df.groupby('Review').size().div(len(data_df))
likelihood = {}
likelihood['Platform'] = data_df.groupby(['Review', 'Platform']).size().div(len(data_df)).div(prior)
likelihood['Genre'] = data_df.groupby(['Review', 'Genre']).size().div(len(data_df)).div(prior)


# attempts to find likelyhood of review score being over 80 (can be changed)
def find_score(platform, genre):
    try:
        probability_of_good = likelihood['Platform'][1][platform] * likelihood['Genre'][1][genre]
        probability_of_bad = likelihood['Platform'][0][platform] * likelihood['Genre'][0][genre]
        if probability_of_good >= probability_of_bad:
            return 'good'
        else:
            return 'bad'
    except KeyError:
        return 'none'


# list of header / row names
platforms = ['Wii', 'NES', 'GB', 'DS', 'PS2', 'SNES', 'X360', 'GBA', 'PS3', 'N64', 'PC', 'PS', 'XB', '3DS', 'PSP', 'GC', 'GEN', 'DC', 'SAT', 'WiiU', 'SCD', 'PSV']
genres = ['Sports', 'Platform', 'Racing', 'Puzzle', 'Misc', 'Shooter', 'Simulation', 'Role-Playing', 'Action', 'Fighting', 'Adventure', 'Strategy']

# allocates array for table
array = []
row = genres
for i in range(len(platforms)):
    row = [platforms[i]]
    for j in range(len(genres)):
        row.append(find_score(i, j))
    array.append(row)

# prints out final results
array.append(row)
print(tabulate(array, headers=genres, tablefmt="grid"))

# Naive Bayes unused (low accuracy)
'''
# sets up target and data
target_df = pd.DataFrame(data=df, columns=['Review'])
target_df['Review'] = target_df['Review'].apply(game_scoring)
df = df.drop(labels='Review', axis=1)
data_df = pd.DataFrame(data=df, columns=df.columns)
data_df['Platform'] = data_df['Platform'].apply(platform)
data_df['Genre'] = data_df['Genre'].apply(genre)
data_df['Review'] = data_df['Review'].apply(game_scoring)

X_train, X_test, y_train, y_test = train_test_split(data_df, target_df, test_size=0.5, random_state=125)
model = GaussianNB()
model.fit(X_train, y_train.values.ravel())
predicted = model.predict(X_test)

print("Actual Value:", y_test)
print("Predicted Value:", predicted)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

'''
