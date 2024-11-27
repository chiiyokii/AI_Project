# **Welcome to our AI project !** 

## How not to get scammed buying your house (BZH)
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


&emsp;Brittany is one of the best region of France, with it's cows, crêpes and kouign amann, there is everything you need to spend a really good time during vacations. But with the raising popularity of this destination among french people,the housing market skyrocketed. Indeed, everyday about a hundred of apartments and houses are sold. Today there is 13% more buyers than sellers so the prices are going up. But, as a buyer you should be careful that the lovely house you want to buy is priced fairly. That is why we wanted to create this AI model that can predict the selling price of a houses based on several criteria such as location, surface, yard ...

Dataset from kaggle we will use : [DatasetOfHousingPrice](https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr)

This dataset is composed of 10 columns: selling date, gps coordinates, position on x axis, position on y axis,category (e.g. house or condo), area of living, area of the land, number of rooms, the shape of the building and the price. We have numerical features : selling date,square meters, number of rooms and the position and also boolean features such as if it is a house or an condo.  

After analyzing the content of the dataset, we noticed that some of the features were not usefull, indeed having the gps position and the position on x and y axis is not necessary. Thus we decided not to use the gps coordinates along with the shape of the house. 

Our first step is to understand our dataset with statistic analysis. Especially to look for null and unique values.
We plan to use four different ML techniques to achieve our goal.

**A Quick Data Analysis:**

![image](https://github.com/user-attachments/assets/75db4345-3d2f-4b10-b9fd-83c894216b39)

Before we start building our Machine Learning model, we need to understand our dataset at a deeper level. As said previously, it is composed of ten features : Selling date, x coordinates, y coordinates, category, area of living (in square meters), area of the land (in square meters), number of rooms, the shape of the building and the price. Our model should be able to predict the price of a house, meaning that this should be our target feature. Thus, we need to analyse our dataset to help us identify the usefull features reamaining in our dataset. It is for this purpose that we did a statiscal analysis of our dataset using correlation matrix and a heat map. 

![image](https://github.com/user-attachments/assets/f1eaeed6-51b5-44c8-b1ce-415667f9dc89)

The results of this analyse are shown above in this heat map. As we can see, the features that have the strongest correlation to the price of a house are, as we could have expected, the number of rooms and the area of living, making these two features the most important for our model. The area of land, the selling date and the y_axis position are also correlated, making them useful for us. On the opposite, the category of the accomodation and the x_axi position are negatively correlated to the price, making them useless for our model making. We also decided not to include the gps coordinates because of the existance of the x and y positions in the dataset. We thought about setting aside the shape of the apartment because of the lack of data.

Thus we now have seven features, making our data set a [148 279,7] matrix. 

**Random Forest:**

Now that we have sorted our data, we can start to build our Machine Learning modele by using at first randomforest.


![image](https://github.com/user-attachments/assets/8f3a5618-8347-4151-a424-7c4e416c321b)




