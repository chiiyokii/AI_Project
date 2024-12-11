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



## **Here is a quick glance about what we plan to do :**


&emsp;Brittany is one of the most beautiful regions in France. With its charming cows, delicious crêpes, and famous kouign-amann, it offers everything you need for an unforgettable vacation. 

However, the growing popularity of this destination among French travelers has caused the housing market to skyrocket. Every day, around a hundred apartments and houses are sold, and currently, there are 13% more buyers than sellers, driving prices even higher. 

As a buyer, it’s essential to ensure that the lovely home you want is fairly priced. To help with this, we’ve developed an AI model designed to predict the selling prices of properties. Our goal is to create a machine learning tool that accurately estimates house prices based on market conditions, whether you're looking to buy or sell. 

As a seller, this model also allows users to quickly get an initial idea of a **property's value** without going through lengthy processes with real estate agencies.


Here is the Dataset from kaggle we will use : [DatasetOfHousingPrice](https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr)

## **Description of the dataset**

&emsp;&emsp;This dataset contains 10 columns: **selling date, GPS coordinates, position on the x-axis, position on the y-axis, category** (e.g., house or condo)**, living area, land area, number of rooms, building shape, and price**. The dataset includes **numerical features** such as selling date, living area (in square meters), number of rooms, and coordinates (x and y positions). Additionally, it has boolean features, such as whether the property is a house or a condo. 

We chose this dataset because its features are highly relevant for predicting housing prices. Moreover, the data comes from **actual properties** in Ille-et-Vilaine, making the price predictions more realistic and applicable to the local market.   

After analyzing the dataset, we identified **some features as unnecessary**. Specifically, having both GPS coordinates and x/y position data is redundant, so we decided to exclude the GPS coordinates. We chose not to use the "shape of the house" feature, as it is not directly relevant to our price predictions. 

## **Approach and Methodology**

### **1. Dataset Analysis**
&emsp;&emsp;Our first step is to analyze the dataset using statistical methods. This includes checking for null values, identifying unique values, and understanding correlations between features. In particular, we aim to identify features that are strongly correlated with the price and exclude those that are not.

### **2. Model Training and Comparison**
&emsp;&emsp;Next, we will train several machine learning models and compare their performance and accuracy. We plan to use three to four different ML techniques to determine which approach yields the most reliable price predictions.

## **Quick Data Analysis:**



&emsp;&emsp;&emsp;![image](https://github.com/user-attachments/assets/75db4345-3d2f-4b10-b9fd-83c894216b39)

### **1. Relevant features**
&emsp;&emsp;Before building our machine learning model, we need to gain a deeper **understanding of our dataset**. As mentioned earlier, it consists of ten features: **selling date, x coordinates, y coordinates, category** (e.g., house or condo)**, living area** (in square meters)**, land area** (in square meters)**, number of rooms, shape of the building, and price**. The goal of our model is to predict the price of a house, making it our **target feature**. To achieve this, we need to analyze the dataset thoroughly to identify the most useful features to include in the model. For this purpose, we conducted a statistical analysis, using a **correlation heatmap** to examine the relationships between features and determine which ones are most relevant for predicting price.

![image](https://github.com/user-attachments/assets/273b4f09-c597-472c-85d3-b7bbb27810a6)

&emsp;&emsp;The results of our analysis are shown in the heatmap above. As expected, the features most strongly correlated with house prices are the **number of rooms and the living area**, making them the most important variables for our model. Additionally, land area, selling date, and y-axis position also show notable correlations with price.On the other hand, the category of accommodation and the x-axis position are negatively correlated with price. To simplify our model, we decided not to include GPS coordinates, as the dataset already contains x and y position data. Furthermore, we chose to exclude the shape of the building due to insufficient data for this feature.

Thus we now have seven features, making our data set a [148 279,7] matrix. Our target feature being the price, we have now six features to work with.
To be able to predict the price we are going to use several methods such as Random forest, decision tree and XGB to see which one is the most effective.

### **2. Additional analysis**

&emsp;Here is a map of all the prices by quantiles.

![Image](https://github.com/user-attachments/assets/104b1458-8e9a-4991-b626-ab2c5e6e2e40)

It is logical that the highest prices are grouped around the large city of Rennes and around the seaside resort of Saint Malo (in the north).

&emsp;The **time and date** is a relevant indicator of the price as it is the only variable reflecting the market ups and downs.

![Screenshot 2024-12-11 165825](https://github.com/user-attachments/assets/8afc3f43-767a-495a-b542-eb0526b36a53)


## **I. Random Forest**

&emsp;&emsp;For our first machine learning model, we chose Random Forest. It is a popular algorithm and one of the first machine learning techniques we learned, making it a natural starting point for this project.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

Here we convert the date in timestamp UNIX to be able to have the date in int because that will make it easier to use it

```python
data = pd.read_csv('housing-prices-35.csv')
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].view('int64') // 10**9  
```

Seperating the target feature from the rest of the features and scaling all the values

```python
data = pd.get_dummies(data, columns=['category'], drop_first=True)
#Here we define the caracteristics (X) and the target (y)
X = data.drop(columns=['price'])
y = data['price']
#We put the caracteristics on the right scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Let's train our Random Forest model and test the accuracy. We will use 80% of the dataset to train and the remaining to test the accuracy of our model

```python
# Then we split the data between training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# We create and train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# We do the prediction
y_pred = rf_model.predict(X_test)

# Finally, we calculate the mean squared error and r2 to be able to see how accurate our model is.
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)
```

**Mean Squared Error (MSE) : 14601154703.797789
Root Mean Squared Error (RMSE) : 120835.23783978657
R² Score : 0.685**


## **II. Decision Tree**

To train the decision tree model, we are going to use the same code except for the libraries and the training parts.

Here are the necessary libraries we plan to use.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

Training part adapted to decision tree.

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred)
r2_dt = r2_score(y_test, y_pred)
```

Results :
**Mean Squared Error (MSE) : 19024580900.28144
Root Mean Squared Error (RMSE) : 137929.62299767748
R² Score : 0.589544929491228**

## **III. Pipeline using RandomForestRegressor and a different way to clean the dataset**

As a reference to the RandomForest ML model we previously made, we decided to use pipelines from the SKLearn library to try the same algorithm on a different cleaned dataset. The pipelines act as the assembly lines for our ML model, which will be working on **RandomForest Regressor**. To evaluate its accuracy, we will use different error indicators such as the **Mean Squared Error (MSE)** as our performance metrics.

These are the libraries used during this whole process : 

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

### **1. Dataset Preparation**

&emsp;&emsp;First of all, we need to prepare our dataset for an effective training. We will first get the dataset and remove all duplicates.

```python
df = pd.read_csv("housing-prices-35.csv")
df2 = df
df2 = df2.drop_duplicates()
```

&emsp;&emsp;Next step is to **drop** the useless features and **encode** both the **date** and **category** to optimize the machine learning process.

```python
df_ml = df2.drop(['position_wgs', 'shape_wgs'], axis=1)
df_ml = df_ml.sort_values('date')
df_ml['date_encoded'] = range(len(df_ml))
df_ml = df_ml.drop('date', axis=1)
label_encoder = LabelEncoder()
df_ml['category_encoded'] = label_encoder.fit_transform(df_ml['category'])
df_ml = df_ml.drop('category', axis=1)
```

&emsp;&emsp;To use the dataset, we need to normalize every column. After testing scaling (as we did for XGB algorithm) and normalizing, normalizing proves to be the most accurate way to do. This allows the algorithm to work with smaller values and be more efficient : 

```python
column_to_normalise = ['x_lbt93', 'y_lbt93', 'area_living', 'area_land', 'n_rooms']
scaler = StandardScaler()
df_ml[column_to_normalise] = scaler.fit_transform(df_ml[column_to_normalise])
```

&emsp;&emsp;Let's try using 80% of our dataset for training and 20% for testing :

```python
X = df_ml.drop('price', axis=1)
y = df_ml['price']
features = df_ml[['x_lbt93','y_lbt93','area_living','area_land','n_rooms','date_encoded','category_encoded']]
target = df_ml['price']
```
### **2. Training the model**
&emsp;&emsp;The only thing left is to train our model using RandomForestRegressor as a base.

```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)
rf_model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('regressor', rf_model)
])
pipeline.fit(X_train, y_train)
```

Let's evaluate the accuracy.

```python
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")
```

**Root Mean Squared Error** : **74949.64 euros**.
**Mean Absolute Error** : **38363.59**
**Mean Squared Error** : **5617448228.**
**R2** : **0.696**

## **IV. eXtreme Gradient Boosting:**

The third method we will be using XGBoost to try to predict more precisely the prices of accomodation. XGB is a powerful and efficient machine learning algorithm designed for both classification and regression tasks. It is based on the principle of gradient boosting, where multiple decision trees are built sequentially, and each tree learns to correct the errors of the previous ones.

We will need use those libraries in order to build our model.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import metrics

```
Now that we have imported our libraries and since we already checked our datas, we can start to modify our datas so they can be included more easily in our model, such as the selling and the type of accomodation.
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
X = df.drop(['price'], axis = 1)
y = df['price']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=101)
```
In the code cell above, we select our target feature and we split our data set into a test part and a training part. Here, we decided to use 30% of our data set as a testing part.

```python
scaler_train= StandardScaler()
scaler_train.fit(X_train)
scaler_test= StandardScaler()
scaler_test.fit(X_test)
X_train_scaled= scaler_train.transform(X_train)
X_test_scaled= scaler_test.transform(X_test)

XGB_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
XGB_model.fit(X_train_scaled,y_train)
```
In this part of the code, we first use a standard scaler to standardize our features, ensuring they are all on a similar scale. Standardization is use to make sure our XGboost will perform better. 
Then we finally build our model using the XGBRegressor method, which in our case will use 1000 estimators corresponding to the number of trees the model will use, and a learning of 0.05. 
```python
y_pred = XGB_model.predict(X_test_scaled)
y_pred = pd.DataFrame(y_pred)
MSE_XGB = metrics.mean_squared_error(y_test, y_pred)
RMSE_XGB =np.sqrt(MSE_XGB)
print(RMSE_XGB)
```
After the training and the testing part, we use the root mean squared error formula to calculate the error of our model.
The value we get with this method of calculation is 86084.0307.


## **V. Linear Regression**

For a Linear Regression based ML model, we'll use the same code as the pipeline, but using the 'LinearRegression' library instead of the pipeline one. We will also adapt the training part.

```python
from sklearn.linear_model import LinearRegression
```

### **Training the model**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

Linear regression accuracy : 
**Root Mean Squared Error: 105405.09555290533
MAE: 62463.86799364371
MSE: 11110234168.517103
R2: 0.3988599701918526**

# **Evaluation & Analysis**



|      Model       |       RMSE        |
|------------------|-------------------|
|  Random Forest   | 120835.24         |
| Decision Tree    | 137929.62         |
|       XGB        | 86084.40          |
|     Pipeline     | 74949.64          |
|Linear Regression | 105405.10         |

## **Diagram of the importance of the features**

Sample of code we are using to analyse the most relevant features of our ML models.

```python
feature_importances = model.feature_importances_

# We convert the caracteristics names in a Numpy array
feature_names = np.array(X.columns)  
sorted_idx = np.argsort(feature_importances) 

plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx], color='skyblue')
plt.xlabel("Importance des caractéristiques")
plt.ylabel("Caractéristiques")
plt.title("Importance des caractéristiques (Random Forest)")
plt.tight_layout()
plt.show()
```
### Random Forest

Here is our results :

![image](https://github.com/user-attachments/assets/f52c03a7-2a16-403a-a9f4-04a02664f2e2)

### Decision tree

![image](https://github.com/user-attachments/assets/c6d3bbd5-c849-4af4-9c8a-87c1bd1ca78c)


### XBG

![image](https://github.com/user-attachments/assets/d8887d6d-50fa-43f7-9d6e-d3fcc8196242)


### Pipeline

![Screenshot 2024-12-11 170305](https://github.com/user-attachments/assets/62859049-987c-46aa-a164-917f20cf459f)

### Linear Regression

Since the features_importance_ does not exist for linear regression, we will instead examine the coefficient of the linear model using this code:

```python
feature_importances = np.abs(lr.coef_)  
feature_names = np.array(X.columns)
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx], color='skyblue')
plt.xlabel("Importance of Features (Coefficient Magnitude)")
plt.ylabel("Features")
plt.title("Feature Importance for Linear Regression")
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/e86d449c-d0e2-4392-929b-af13f74b54d4)

On these graphics, we can see that a certain tendency appears in relation to the distribution of the values. Indeed, all the models tend to have a very large difference between the importance of the features. It also seems that for the decision tree, the random forest model, the XGB and the linear regression, the most important feature is by far the area of living of the accomodation, whereas for the pipeline this feature is the third more important after the date and the y axis position. With these graphics, we can see why the random forest, the decision tree and the linear regression all have a RMSE of the same order of magnitude, whereas the pipeline model and the XGB which have a lower error are using the same pattern for the importance of the features but not in the same order.




## **Error distribution by price**

For each ML model, let's show the error of each predicted value compared to the actual test values.
Here is a sample of the code used.

```python
individual_rmse = np.sqrt((y_pred - y_test) ** 2)
print(individual_rmse)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, individual_rmse, color='blue', alpha=0.7, edgecolor='k')
# labels and title
plt.xlabel('True Values (y_test)')
plt.ylabel('Individual RMSE')
plt.title('Individual RMSE vs True Values')
plt.grid(True, linestyle='--', alpha=0.6)
# Display the plot
plt.show()
```

### RandomForest

![grapgh3](https://github.com/user-attachments/assets/b9fb30aa-413d-441c-b086-94b736b94eab)

### Pipeline

![grapgh](https://github.com/user-attachments/assets/05226707-26e2-42ba-b7fe-f782e3f6def7)

### XGB

![grapgh4](https://github.com/user-attachments/assets/887b3480-eec9-42d7-879e-a21b1fc8397e)


### Linear Regression

![grapgh2](https://github.com/user-attachments/assets/ab64db2a-cde6-4a50-9c3a-4756691458b3)

### Analysis

&emsp;&emsp;For most of the models, the error follows a proportional relationship with the price. The higher is the price to predict, the higher is the error. 


# **Conclusion**

The raw dataset contained several unnecessary features and values on different scales. Understanding and preprocessing the dataset was crucial, as it significantly impacts the accuracy and efficiency of the machine learning model. We tested various machine learning models to determine which provided the best accuracy. Ultimately, the Random Forest Regressor, combined with pipelines and a dataset of normalized values, delivered the best performance, resulting in a RMSE equal to 80 000 euros. Thus we can say that our model fullfilled its purpose of being able to roughly predict the price of an accomodation in order to help sellers and buyers to make their first opinion on the prices on the housing market.


# **Related Work**

Inspired by these sources for RandomForest Regressor : [SkLearn_RandomForest_Regressor1](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [Randomforest_regression](https://www.geeksforgeeks.org/random-forest-regression-in-python/)

Inspired by these sources for SkLearn Pipelines : [SkLearn_Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html), [SkLearn_Pipeline2](https://velog.io/@imfromk/MLsklearn-Pipeline)

Inspired by these sources for XGB : [How to train XGBoost models in Python](https://www.youtube.com/watch?v=aLOQD66Sj0g)

General inspiration for the project : [housing price in Paris](https://www.kaggle.com/datasets/mssmartypants/paris-housing-classification)

We used ChatGPT to help generate and understand certain parts of the code during the development of this project.
