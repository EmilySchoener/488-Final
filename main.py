import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, RocCurveDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor

# reads initial file
data = pd.read_csv('Video Games Sales.csv')
df = data
# removes all columns except platform, genre, and global sales

df = df.drop(labels='index', axis=1)

# df = df.drop(labels='Rank', axis=1)

df = df.drop(labels='Game Title', axis=1)

# df = df.drop(labels='Year', axis=1)
# df = df.drop(labels='Publisher', axis=1)
# df = df.drop(labels='North America', axis=1)
# df = df.drop(labels='Europe', axis=1)
# df = df.drop(labels='Japan', axis=1)
# df = df.drop(labels='Rest of World', axis=1)
# df = df.drop(labels='Global', axis=1)

print(df.columns)
# print ('The train data has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))
# print(df['Platform'].unique())
# print(df['Genre'].unique())

# identifying datatypes for columns
'''
objects_cols = ['object']
objects_lst = list(df.select_dtypes(include=objects_cols).columns)
print("Total number of categorical columns are ", len(objects_lst))
print("There names are as follows: ", objects_lst)

int64_cols = ['int64']
int64_lst = list(df.select_dtypes(include=int64_cols).columns)
print("Total number of numerical columns are ", len(int64_lst))
print("There names are as follows: ", int64_lst)

float64_cols = ['float64']
float64_lst = list(df.select_dtypes(include=float64_cols).columns)
print("Total number of float64 columns are ", len(float64_lst))
print("There name are as follow: ", float64_lst)

#count the total number of rows and columns.
print('The new dataset has {0} rows and {1} columns'.format(df.shape[0],df.shape[1]))
'''
# print(df.isna().sum()) #Identify where null values are
df.dropna(inplace=True)  # Drop null values

# Encoding categorical data values
le = LabelEncoder()
df.Platform = le.fit_transform(df.Platform)
df.Genre = le.fit_transform(df.Genre)
df.Publisher = le.fit_transform(df.Publisher)

# Graph data before fixing skew
df.hist(bins=50, figsize=(10, 10))
plt.suptitle("Data before fixing skew")
plt.show()


# Function to identify skewness
def right_nor_left(df, float64_lst):
    temp_skewness = ['column', 'skewness_value', 'skewness (+ve or -ve)']
    temp_skewness_values = []

    temp_total = ["positive (+ve) skewed", "normal distribution", "negative (-ve) skewed"]
    positive = 0
    negative = 0
    normal = 0

    for value in float64_lst:

        rs = round(df[value].skew(), 4)

        if rs > 0:
            temp_skewness_values.append([value, rs, "positive (+ve) skewed"])
            positive = positive + 1

        elif rs == 0:
            temp_skewness_values.append([value, rs, "normal distribution"])
            normal = normal + 1

        elif rs < 0:
            temp_skewness_values.append([value, rs, "negative (-ve) skewed"])
            negative = negative + 1

    skewness_df = pd.DataFrame(temp_skewness_values, columns=temp_skewness)
    skewness_total_df = pd.DataFrame([[positive, normal, negative]], columns=temp_total)

    return skewness_df, skewness_total_df


float64_cols = ['float64']
float64_lst_col = list(df.select_dtypes(include=float64_cols).columns)

skew_df, skew_total_df = right_nor_left(df, float64_lst_col)
print(skew_df)  # The sales data is skewed

df['NA_Sales_up'] = np.sqrt(df['North America'])
print(df['NA_Sales_up'].skew())

df['EU_Sales_up'] = np.sqrt(df['Europe'])
print(df['EU_Sales_up'].skew())

df['JP_Sales_up'] = np.sqrt(df['Japan'])
print(df['JP_Sales_up'].skew())

df['Other_Sales_up'] = np.sqrt(df['Rest of World'])
print(df['Other_Sales_up'].skew())

df['Global_Sales_up'] = np.sqrt(df['Global'])
print(df['Global_Sales_up'].skew())

# Drop old columns
df = df.drop(['North America', 'Europe', 'Japan', 'Rest of World', 'Review', "Global"], axis=1)

# Graph data after fixing skew
df.hist(bins=50, figsize=(10, 10))
plt.suptitle("Data after fixing skew")
plt.show()

# correlation plot
sns.set(rc={'figure.figsize': (7, 7)})
corr = df.corr().abs()
sns.heatmap(corr, annot=True)
plt.show()

df = df.drop(labels='Rank', axis=1)
# df = df.drop(labels='Year', axis=1)
df = df.drop(labels='Publisher', axis=1)

target = 'Global_Sales_up'
X = df.drop(target, axis=1)
y = df[target]
# y = y.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


print("\n\n", X.shape, y.shape)
print("test data Accuracy: ", model.score(X_test, y_test))

# R2 score
print("Linear Regression R2: ", model.score(X_train, y_train))
# Predict the values on X_test_scaled dataset
y_predicted = model.predict(X_test)
rg = r2_score(y_test, y_predicted)*100
print("\nThe accuracy is: {}".format(rg))

regressor = RandomForestRegressor(max_depth=2, random_state=0)
regressor.fit(X_train, y_train)
RandomForestRegressor(max_depth=2, random_state=0)
# R2 score
print("RF R2:", regressor.score(X_train, y_train))

# predict the values on X_test_scaled dataset
y_predicted = regressor.predict(X_test)
rg = r2_score(y_test, y_predicted)*100
print("\nThe accuracy is: {}".format(rg))


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


df = data
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


# attempts to find likelihood of review score being over 80 (can be changed)
def find_score(platform, genre):
    try:
        probability_of_good = likelihood['Platform'][1][platform] * likelihood['Genre'][1][genre]
        probability_of_bad = likelihood['Platform'][0][platform] * likelihood['Genre'][0][genre]
        if probability_of_good >= probability_of_bad:
            return 'Higher'
        else:
            return 'Lower'
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
print("Review Ratings over 80%")
print(tabulate(array, headers=genres, tablefmt="grid"))

# Naive Bayes unused (low accuracy)
target_df = pd.DataFrame(data=df, columns=['Review'])
target_df['Review'] = target_df['Review'].apply(game_scoring)
df = df.drop(labels='Review', axis=1)
data_df = pd.DataFrame(data=df, columns=['Platform', 'Genre'])
data_df['Platform'] = data_df['Platform'].apply(platform)
data_df['Genre'] = data_df['Genre'].apply(genre)
# data_df['Review'] = data_df['Review'].apply(game_scoring)

X_train, X_test, y_train, y_test = train_test_split(data_df, target_df, test_size=0.3, random_state=125)
model = GaussianNB()
model.fit(X_train, y_train.values.ravel())
predicted = model.predict(X_test)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy NB:", accuray)
print("F1 Score NB:", f1)
