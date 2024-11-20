# **Welcome to our AI project !** 

## How not ot get scammed buying your house (BZH)
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
