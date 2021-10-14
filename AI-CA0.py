#!/usr/bin/env python
# coding: utf-8

# # In the Name of God
# Mohammad Hossein Ataie SID: 810197632

# # Part 1

# In[225]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


# In[221]:


dataset = pd.read_csv("FuelConsumptionCo2.CSV", converters={
                      "tokens": lambda tokenList: tokenList[1:].split(",")})


# # DataFrame.head(n)
# This function returns the first n rows for the object based on position.

# In[4]:


dataset.head(1)


# # DataFrame.tail(n)
# This function returns last n rows from the object based on position.

# In[5]:


dataset.tail(1)


# # DataFrame.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)
# Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
#
# Analyzes both numeric and object series, as well as DataFrame column sets of mixed data types. The output will vary depending on what is provided.

# In[6]:


dataDisc = dataset.describe()
dataDisc


# # Part 2

# # DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None)
# This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.

# In[7]:


dataset.info()


# In[8]:


obj_df = dataset.select_dtypes(include=['object']).copy()
obj_df.head()


# In[9]:


obj_df["MAKE"] = obj_df["MAKE"].astype('category')
obj_df["MODEL"] = obj_df["MODEL"].astype('category')
obj_df["VEHICLECLASS"] = obj_df["VEHICLECLASS"].astype('category')
obj_df["TRANSMISSION"] = obj_df["TRANSMISSION"].astype('category')
obj_df["FUELTYPE"] = obj_df["FUELTYPE"].astype('category')


# In[10]:


obj_df["MAKE_cat"] = obj_df["MAKE"].cat.codes
obj_df["MODEL_cat"] = obj_df["MODEL"].cat.codes
obj_df["VEHICLECLASS_cat"] = obj_df["VEHICLECLASS"].cat.codes
obj_df["TRANSMISSION_cat"] = obj_df["TRANSMISSION"].cat.codes
obj_df["FUELTYPE_cat"] = obj_df["FUELTYPE"].cat.codes
obj_df.dtypes


# The nice aspect of this approach is that you get the benefits of pandas categories (compact data size, ability to order, plotting support) but can easily be converted to numeric values for further analysis.

# # Part 3

# In[11]:


count = dataset["MODELYEAR"].isna().sum()
print("Number of 'Nan' in MODELYEAR:", count)
count = dataset["MAKE"].isna().sum()
print("Number of 'Nan' in MAKE:", count)
count = dataset["MODEL"].isna().sum()
print("Number of 'Nan' in MODEL:", count)
count = dataset["VEHICLECLASS"].isna().sum()
print("Number of 'Nan' in VEHICLECLASS:", count)
count = dataset["ENGINESIZE"].isna().sum()
print("Number of 'Nan' in ENGINESIZE:", count)
count = dataset["CYLINDERS"].isna().sum()
print("Number of 'Nan' in CYLINDERS:", count)
count = dataset["TRANSMISSION"].isna().sum()
print("Number of 'Nan' in TRANSMISSION:", count)
count = dataset["FUELTYPE"].isna().sum()
print("Number of 'Nan' in FUELTYPE:", count)
count = dataset["FUELCONSUMPTION_CITY"].isna().sum()
print("Number of 'Nan' in FUELCONSUMPTION_CITY:", count)
count = dataset["FUELCONSUMPTION_HWY"].isna().sum()
print("Number of 'Nan' in FUELCONSUMPTION_HWY:", count)
count = dataset["FUELCONSUMPTION_COMB_MPG"].isna().sum()
print("Number of 'Nan' in FUELCONSUMPTION_COMB_MPG:", count)
count = dataset["CO2EMISSIONS"].isna().sum()
print("Number of 'Nan' in CO2EMISSIONS:", count)


# In[12]:


dataset.isna().sum()


# In[125]:


column_means = dataset.mean()
df = dataset.fillna(column_means)
df["CO2EMISSIONS"] = dataset["CO2EMISSIONS"]
len(df)


# In[143]:


test = dataset[dataset["CO2EMISSIONS"].isnull()]
print(len(test))
train = dataset[dataset['CO2EMISSIONS'].notna()]
print(len(train))


# *** Dropping the columns that have Nan Values has an advantage like quick and simpler process.
# it also has disadvantages like a significant loss of data (about 40%)
#
# *** Filling the null values with the mean of a particular column makes no significant loss of data (about 0.1%)
#
# Disadvantages of this method are Inaccuracy- which will hamper the process of analysis and prediction
# For a particular category, all null values present for a column.
# For example, on a health questionnaire, heavier respondents may be less willing to disclose their weight. The mean of the observed values would be lower than the true mean for all respondents, and you'd be using that value in place of values that should actually be considerably higher.
# Using the mean is less of an issue if the reason the values are missing is independent of the missing values themselves.

# # Part 4

# In[228]:


start_time_vectorization = time.time()
s1 = dataset[dataset["CO2EMISSIONS"] < 240].mean()
end_time_vectorization = time.time() - start_time_vectorization
s1


# In[223]:


dataset[dataset["CO2EMISSIONS"] > 300].mean()


# # Part 5

# In[224]:


newDs = dataset["CO2EMISSIONS"].dropna()
newDs


# In[229]:


start_time_forloop = time.time()
count = temp = 0
for column in dataset[['CO2EMISSIONS']]:
    columnSeriesObj = dataset[column]
    columnSeriesObj = columnSeriesObj
    for i in range(len(dataset)):
        if (np.isnan(columnSeriesObj.values[i]) == False):
            if int(columnSeriesObj.values[i]) < 240:
                count += 1
                temp += dataset["FUELCONSUMPTION_CITY"][i]
print(temp/count)
end_time_forloop = time.time() - start_time_forloop


# In[230]:


count = temp = 0
for column in dataset[['CO2EMISSIONS']]:
    columnSeriesObj = dataset[column]
    columnSeriesObj = columnSeriesObj
    for i in range(len(dataset)):
        if (np.isnan(columnSeriesObj.values[i]) == False):
            if int(columnSeriesObj.values[i]) > 300:
                count += 1
                temp += dataset["FUELCONSUMPTION_CITY"][i]
print(temp/count)


# In[234]:


print("elapsed time in vectorization:", end_time_vectorization)
print("elapsed time in for-loop:     ", end_time_forloop)


# As you can see, the execution time of the program in vectorization mode is one-third of the time we use for-loop.

# # Part6

# In[149]:


hist = dataset["MODELYEAR"].hist(bins=50)


# In[150]:


hist = dataset["ENGINESIZE"].hist(bins=50)


# In[151]:


plt.xticks(rotation=90)
hist = dataset["MAKE"].hist(bins=50)


# In[152]:


hist = dataset["FUELTYPE"].hist(bins=10)


# In[153]:


plt.xticks(rotation=90)
hist = dataset["VEHICLECLASS"].hist(bins=50)


# In[154]:


plt.xticks(rotation=90)
hist = dataset["TRANSMISSION"].hist(bins=50)


# In[236]:


plt.xticks(rotation=90)
hist = dataset["FUELCONSUMPTION_CITY"].hist(bins=50)


# In[237]:


plt.xticks(rotation=90)
hist = dataset["FUELCONSUMPTION_HWY"].hist(bins=50)


# In[238]:


plt.xticks(rotation=90)
hist = dataset["FUELCONSUMPTION_COMB"].hist(bins=50)


# In[240]:


plt.xticks(rotation=90)
hist = dataset["FUELCONSUMPTION_COMB_MPG"].hist(bins=50)


# # Part7

# In[271]:


normalized_df = (train - train.mean())/train.std()
print(train["CO2EMISSIONS"].mean())
print(train["CO2EMISSIONS"].std())
normalized_df


# # Part8_A)

# In[156]:


x = np.array(train["MODELYEAR"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[157]:


plt.xticks(rotation=90)
x = np.array(train["ENGINESIZE"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[158]:


plt.xticks(rotation=90)
x = np.array(train["CYLINDERS"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[159]:


plt.xticks(rotation=90)
x = np.array(train["FUELCONSUMPTION_CITY"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[160]:


plt.xticks(rotation=90)
x = np.array(train["FUELCONSUMPTION_HWY"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[161]:


plt.xticks(rotation=90)
x = np.array(train["FUELCONSUMPTION_COMB"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# In[162]:


plt.xticks(rotation=90)
x = np.array(train["FUELCONSUMPTION_COMB_MPG"])
y = np.array(train["CO2EMISSIONS"])
plt.scatter(x, y, cmap='viridis')
plt.show()


# # Part8_B)

# As we studied in Statistics and Probability, When the "Y" variable tends to increase as the "x" variable increases, we say there is a positive correlation between the variables.
# The strongest linear relationship occurs when the slope is 1.  This means that
# when one variable increases by one, the other variable also increases by the
# same amount.  This line is at a 45 degree angle.
# FUELCONSUMPTION_COMB seems have the most linear correlation compare to others.

# # Part 9

# In[203]:


x_ = np.matrix([np.ones(len(normalized_df["CO2EMISSIONS"])),
               normalized_df["CO2EMISSIONS"]]).T
y_ = np.matrix(normalized_df["FUELCONSUMPTION_COMB"]).T


# Martix X:

# In[204]:


x_


#

# In[205]:


fn = np.linalg.inv(x_.T.dot(x_)).dot(x_.T).dot(y_)


# In[206]:


fn


# In[211]:


Y_true = normalized_df["CO2EMISSIONS"]
Y_pred = float(fn[1]) * normalized_df["FUELCONSUMPTION_COMB"] + float(fn[0])
MSE = np.square(np.subtract(Y_true, Y_pred)).mean()


# In[212]:


MSE


# # Part 11

# In[218]:


plt.xticks(rotation=90)
x = np.array(normalized_df["FUELCONSUMPTION_COMB"])
y = np.array(normalized_df["CO2EMISSIONS"])
plt.plot(x, y, "o")
y__ = float(fn[1]) * x + float(fn[0])
plt.plot(x, y__, '-r')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.grid()
plt.show()


# The
# points on the scatterplot closely
# resemble a straight line.  A
# relationship is linear if one
# variable increases by
# approximately the same rate as the
# other variables changes by one
# unit.
# Another important component to a scatterplot is the strength of the
# relationship between the two variables.
# The slope provides information on the strength of the relationship.
# The correlation coefficient is based on means and standard deviations, so it is
# not robust to outliers; it is strongly affected by extreme observations.  These
# individuals are sometimes referred to as influential observations because
# they have a strong impact on the correlation coefficient

# # Part 12

# In[272]:


test["FUELCONSUMPTION_COMB"] = (test["FUELCONSUMPTION_COMB"] -
                                test["FUELCONSUMPTION_COMB"].mean())/test["FUELCONSUMPTION_COMB"].std()
co2norm = float(fn[1]) * test["FUELCONSUMPTION_COMB"] + float(fn[0])
co2norm = (co2norm * train["CO2EMISSIONS"].std()) + \
    train["CO2EMISSIONS"].mean()
co2norm


# In[273]:


co2norm.to_csv('myfile.csv')
