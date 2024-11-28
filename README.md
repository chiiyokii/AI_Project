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


&emsp;Brittany is one of the best region of France, with it's cows, crêpes and kouign amann, there is everything you need to spend a really good time during vacations. But with the raising popularity of this destination among french people,the housing market skyrocketed. Indeed, everyday about a hundred of apartments and houses are sold. Today there is 13% more buyers than sellers so the prices are going up. But, as a buyer you should be careful that the lovely house you want to buy is priced fairly. That is why we wanted to create this AI model that can predict the selling price of accomodations. Our goal with this model is to create a machine learning modele able to precisely predict the price of a house depending on the market, whether the user want to buy or sell an accomodation.Indeed, people will be able to use it to easily make their first opinion on the price of a house without having to go through a long process with real estate agencies.


Dataset from kaggle we will use : [DatasetOfHousingPrice](https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr)

This dataset is composed of 10 columns: selling date, gps coordinates, position on x axis, position on y axis,category (e.g. house or condo), area of living, area of the land, number of rooms, the shape of the building and the price. We have numerical features : selling date,square meters, number of rooms and the position and also boolean features such as if it is a house or an condo.We chose this dataset because the features it has are really interesting and the values are values from real houses in Ille et Vilaine so the price prediction makes more sense in this configuration.   

After analyzing the content of the dataset, we noticed that some of the features were not usefull, indeed having the gps position and the position on x and y axis is not necessary. Thus we decided not to use the gps coordinates along with the shape of the house. 

Our first step is to understand our dataset with statistic analysis. Especially to look for null and unique values. This process also allows us to understand all the correlations between the features. In our case, we are looking for the features strongly correlated with the price, and identify those who are not.

The second step is to train multiple ML model to compare results and accuracy. We plan to use between three and four different ML techniques to achieve our goal.

## **A Quick Data Analysis:**

![image](https://github.com/user-attachments/assets/75db4345-3d2f-4b10-b9fd-83c894216b39)

Before we start building our Machine Learning model, we need to understand our dataset at a deeper level. As said previously, it is composed of ten features : Selling date, x coordinates, y coordinates, category, area of living (in square meters), area of the land (in square meters), number of rooms, the shape of the building and the price. Our model should be able to predict the price of a house, meaning that this should be our target feature. Thus, we need to analyse our dataset to help us identify the usefull features reamaining in our dataset. It is for this purpose that we did a statiscal analysis of our dataset using a correlation heat map. 

![image](https://github.com/user-attachments/assets/f1eaeed6-51b5-44c8-b1ce-415667f9dc89)

The results of this analyse are shown above in this heat map. As we can see, the features that have the strongest correlation to the price of a house are, as we could have expected, the number of rooms and the area of living, making these two features the most important for our model. The area of land, the selling date and the y_axis position are also correlated but on the opposite, the category of the accomodation and the x_axis position are negatively correlated to the price. We also decided not to include the gps coordinates because of the existance of the x and y positions in the dataset. We thought about setting aside the shape of the apartment because of the lack of data.


Thus we now have seven features, making our data set a [148 279,7] matrix. Our target feature being the price, we have now six features to work with.
To be able to predict the price we are going to use several methods such as Random forest, decision tree and XGB to see which one is the most effective.

**Random Forest:**

Now that we have sorted our data, we can start to build our Machine Learning model by using at first randomforest.


**eXtreme Gradient Boosting:**

The second method we will be using XGBoost to try to predict more precisely the prices of accomodation. XGB is a powerful and efficient machine learning algorithm designed for both classification and regression tasks. It is based on the principle of gradient boosting, where multiple decision trees are built sequentially, and each tree learns to correct the errors of the previous ones.

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
features = df[['date','y_lbt93','area_living','area_land','n_rooms']]
target = df['price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```




