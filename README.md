# Ex-08-Data-Visualization-
# AIM
To Perform Data Visualization on the given dataset and save the data to a file.

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature generation and selection techniques to all the features of the data set

## STEP 4
Apply data visualization techniques to identify the patterns of the data.

# CODE
```
NAME:YUVABHARATHI.B

REGISTER NUMBER:212222230181

#Import required libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#Load the dataset

df = pd.read_csv('Superstore2.csv', encoding='unicode_escape')

#Data Cleaning: Drop unnecessary columns

df.drop(['Row ID', 'Order ID', 'Ship Date', 'Customer ID', 'Postal Code', 'Product ID'], axis=1, inplace=True)

#Feature Generation: Extract Year and Month from Order Date

df['Year'] = pd.DatetimeIndex(df['Order Date']).year

df['Month'] = pd.DatetimeIndex(df['Order Date']).month_name()

#1. Which Segment has Highest sales?

segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(x='Segment', y='Sales', data=segment_sales)

plt.title('Segment-wise Sales')

plt.show()

#2. Which City has Highest profit?

city_profit = df.groupby('City')['Profit'].sum().reset_index().sort_values(by='Profit', ascending=False)

plt.figure(figsize=(12,8))

sns.barplot(x='City', y='Profit', data=city_profit.head(10))

plt.title('Top 10 Cities by Profit')

plt.show()

#3. Which ship mode is profitable?

shipmode_profit = df.groupby('Ship Mode')['Profit'].sum().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(x='Ship Mode', y='Profit', data=shipmode_profit)

plt.title('Ship Mode-wise Profit')

plt.show()

#4. Sales of the product based on region

region_sales = df.groupby('Region')['Sales'].sum().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(x='Region', y='Sales', data=region_sales)

plt.title('Region-wise Sales')

plt.show()

#5. Find the relation between sales and profit

plt.figure(figsize=(8,5))

sns.scatterplot(x='Sales', y='Profit', data=df)

plt.title('Sales vs. Profit')

plt.show()

#6. Find the relation between sales and profit based on the following category.

#i) Segment

segment_sales_profit = df.groupby('Segment')['Sales', 'Profit'].mean().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(x='Segment', y='Sales', data=segment_sales_profit, color='blue', alpha=0.5, label='Sales')

sns.barplot(x='Segment', y='Profit', data=segment_sales_profit, color='green', alpha=0.5, label='Profit')

plt.title('Segment-wise Sales and Profit')

plt.legend()

plt.show()

#ii) City

city_sales_profit = df.groupby('City')['Sales', 'Profit'].mean().reset_index().sort_values(by='Profit', ascending=False).head(10)

plt.figure(figsize=(12,8))

sns.barplot(x='City', y='Sales', data=city_sales_profit, color='blue', alpha=0.5, label='Sales')

sns.barplot(x='City', y='Profit', data=city_sales_profit, color='green', alpha=0.5, label='Profit')

plt.title('Top 10 Cities by Sales and Profit')

plt.legend()

plt.show()

#iii) States

plt.figure(figsize=(8,5))

sns.scatterplot(x='Sales', y='Profit', hue='State', data=df)

plt.title('Sales vs. Profit based on State')

plt.show()

#iv) Segment and Ship Mode

plt.figure(figsize=(8,5))

sns.scatterplot(x='Sales', y='Profit', hue='Segment', style='Ship Mode', data=df)

plt.title('Sales vs. Profit based on Segment and Ship Mode')

plt.show()

#v) Segment, Ship mode and Region

plt.figure(figsize=(8,5))

sns.scatterplot(x='Sales', y='Profit', hue='Segment', style='Ship Mode', size='Region', data=df)

plt.title('Sales vs. Profit based on Segment, Ship Mode and Region')

plt.show()
```
# OUPUT:

![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/9d322ea8-c5b3-43f1-b89d-cf6f69b79a1d)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/702ae59d-ad76-49ba-83a2-689a1d94f062)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/eee60dfe-008b-4cc0-9b9a-03df856b442c)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/ed32e01c-e569-4a5a-815d-68f7786a9a84)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/eaa69043-2388-4220-8300-3d1b5fafaa1c)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/695c413a-2f0a-4b50-8bc2-ce2304ec9a0b)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/33523b49-b172-45ef-a10c-1dd5e926a2dc)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/ea455932-b9b8-49e3-b57a-3135a5dcb901)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/457c5009-ad1e-405b-9e80-0deffdca6402)
![image](https://github.com/yuvabharathib/Ex-08-Data-Visualization-/assets/113497404/32282299-9f36-4a29-8064-690b4d5e61f3)



# RESULT:
Hence the data visualization method for the given dataset applied successfully.

