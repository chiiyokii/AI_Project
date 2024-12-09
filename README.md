# **Welcome to our AI project !** 

## How not to get scammed buying your house
![image](https://github.com/user-attachments/assets/886da2b9-05e3-429e-b11e-361063ae8221)
 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/fe026e1b-5a4e-4e4b-a60b-25fcf9e982ae)




**Group members :**

|        Name       |        Major           |
|-------------------|------------------------|
| Ra'iarii Ollivier | Electronic engineering |
| Antoine Mayla     | Electronic engineering |
| Alexia Mansiat    | Mechanical engineering |
| William Rabuel    | Computer Science       |



**Here is a quick glance about what we plan to do :**


&emsp;Brittany is one of the best region of France, with its cows, crêpes and kouign amann, there is everything you could need to have a really good time during vacations. But with the raising popularity of this destination among french people,the housing market skyrocketed. Indeed, everyday about a hundred of apartments and houses are sold. Today there is 13% more buyers than sellers so the prices are going up. But, as a buyer you should be careful that the lovely house you want to buy is priced fairly. That is why we wanted to create this AI model that can predict the selling price of accomodations. Our goal with this model is to create a machine learning model able to precisely predict the price of a house depending on the market, whether the user wants to buy or sell an accomodation.Indeed, people will be able to use it to easily make their first opinion on the price of a house without having to go through a long process with real estate agencies.


Here is the Dataset from kaggle we will use : [DatasetOfHousingPrice](https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr)

This dataset is composed of 10 columns: selling date, gps coordinates, position on x axis, position on y axis,category (e.g. house or condo), area of living, area of the land, number of rooms, the shape of the building and the price. We have numerical features : selling date,square meters, number of rooms and the position and also boolean features such as if it is a house or an condo.We chose this dataset because the features it has are really interesting and the values are values from real houses in Ille et Vilaine so the price prediction makes more sense in this configuration.   

After analyzing the content of the dataset, we noticed that some of the features were not usefull, indeed having the gps position and the position on x and y axis is not necessary. Thus we decided not to use the gps coordinates along with the shape of the house. 

Our first step is to understand our dataset with statistic analysis. Especially to look for null and unique values. This process also allows us to understand all the correlations between the features. In our case, we are looking for the features strongly correlated with the price, and identify those who are not.

The second step is to train multiple ML model to compare results and accuracy. We plan to use between three and four different ML techniques to achieve our goal.

## **A Quick Data Analysis:**

![image](https://github.com/user-attachments/assets/75db4345-3d2f-4b10-b9fd-83c894216b39)

Before we start building our Machine Learning model, we need to understand our dataset on a deeper level. As said previously, it is composed of ten features : Selling date, x coordinates, y coordinates, category, area of living (in square meters), area of the land (in square meters), number of rooms, the shape of the building and the price. Our model should be able to predict the price of a house, meaning that this should be our target feature. Thus, we need to analyse our dataset to help us identify the usefull features reamaining in our dataset. It is for this purpose that we did a statiscal analysis of our dataset using a correlation heat map. 

![image](https://github.com/user-attachments/assets/f1eaeed6-51b5-44c8-b1ce-415667f9dc89)

The results of this analysis are shown above in this heat map. As we can see, the features that have the strongest correlation to the price of a house are, as we could have expected, the number of rooms and the area of living, making these two features the most important for our model. The area of land, the selling date and the y_axis position are also correlated but on the opposite, the category of the accomodation and the x_axis position are negatively correlated to the price. We also decided not to include the gps coordinates because of the existance of the x and y positions in the dataset. We thought about setting aside the shape of the apartment because of the lack of data.


Thus we now have seven features, making our data set a [148 279,7] matrix. Our target feature being the price, we have now six features to work with.
To be able to predict the price we are going to use several methods such as Random forest, decision tree and XGB to see which one is the most effective.

**Random Forest:**

Now that we have sorted our data, we can start to build our Machine Learning model by using at first randomforest.
Here are our results :
![image](https://github.com/user-attachments/assets/40c36697-9194-48a1-aed5-bfe7b161d53c)
Random Forest Model Evaluation:

Mean Squared Error (MSE): 14671097024.148306

Root Mean Squared Error (RMSE): 121124.3040192525

R² Score: 0.6834712840691937

**Decision Tree**

Code : 
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\alexi\Documents\Cours\A3\IA project\housing-prices-35-cleaned.csv"
data = pd.read_csv(file_path)

data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].view('int64') // 10**9  # Convertir en timestamp UNIX

data = pd.get_dummies(data, columns=['category'], drop_first=True)

X = data.drop(columns=['price'])
y = data['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred)
r2_dt = r2_score(y_test, y_pred)

print("Évaluation du modèle Decision Tree :")
print(f"Mean Squared Error (MSE) : {mse_dt}")
print(f"R² Score : {r2_dt}")
```

Results :
Mean Squared Error (MSE) : 19024580900.28144
R² Score : 0.589544929491228



**eXtreme Gradient Boosting:**

The second method we will be using XGBoost to try to predict more precisely the prices of accomodation. XGB is a powerful and efficient machine learning algorithm designed for both classification and regression tasks. It is based on the principle of gradient boosting, where multiple decision trees are built sequentially, and each tree learns to correct the errors of the previous ones.

We will need use those libraries in order to build our model.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
```
Now that we imported our libraries and since we already checked our datas, we can start to modify our datas so they can be included more easily in our model, such as the selling and the type of accomodation.
```python
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Selling_date'] = (df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df.drop(columns=['date'], inplace=True)
if 'category' in df.columns:
    df['category'] = df['category'].map({'H': 0, 'C': 1}).fillna(-1)
```
We convert our date into integers and the type of accomodation into either a 0 or a 1.

Once this is done, we can start to prepare our model.

```python
df['price'] = pd.factorize(df['price'])[0] + 1
X = df.drop(['price'], axis = 1)
y = df['price']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=101)

scaler_train= StandardScaler()
scaler_train.fit(X_train)
scaler_test= StandardScaler()
scaler_test.fit(X_test)
X_train_scaled= scaler_train.transform(X_train)
X_test_scaled= scaler_test.transform(X_test)

XGB_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
XGB_model.fit(X_train_scaled,y_train)
```

```python
y_pred = XGB_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MSE_XGB = metrics.mean_squared_error(y_test, y_pred)
RMSE_XGB =np.sqrt(MSE_XGB)
print(RMSE_XGB)
```
The result we get with the last cell is the root mean squared error of the model, which is 2437.544988756981.
Thus, we now know that our model built with the XBG method can predict the price of an accomodation with an error of 2438 euros.

**Machine Learning Pipeline**

As a reference to other existing Machine learning Algorithm, let's try using pipelines from the library SKLearn. This algorithm, compared to the previous one, is not as effective and we are expecting a big error. To test the accuracy, we will apply a Mean Squarred Error. 

These are the libraries used during this whole process : 

```ruby
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
```

First of all we need to prepare our dataset for an effective training.

```ruby
houses = pd.read_csv('housing-prices-35.csv')
houses = houses.drop(columns=['shape_wgs','position_wgs','x_lbt93','category'])
df = pd.DataFrame(houses)
```

To use the date, we need to convert into seconds : 

```ruby
df['date'] = pd.to_datetime(df['date']).astype('int64')/10**10
```

Before scaling our datas, let's try with our raw dataset using 80% of it for training and 20% for testing :

```ruby
#Preparing training and testing datas

features = df[['date','y_lbt93','area_living','area_land','n_rooms']]
target = df['price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Pipeline creation and training

numeric_features = [ 'date','y_lbt93', 'area_living', 'area_land', 'n_rooms']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Train the model

pipeline.fit(X_train, y_train)

# Evaluate the model

y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error:", rmse)
```





