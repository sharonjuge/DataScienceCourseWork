#!/usr/bin/env python
# coding: utf-8

# # Springboard Data Science Career Track Unit 4 Challenge - Tier 3 Complete
# 
# ## Objectives
# Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time. 
# 
# In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!
# 
# Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***
# 
# 
# A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.
# 
# ***This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!***
# 
# This challenge will make use of only what you learned in the following DataCamp courses: 
# - Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
# - Data Types for Data Science
# - Python Data Science Toolbox (Part One) 
# - pandas Foundations
# - Manipulating DataFrames with pandas
# - Merging DataFrames with pandas
# 
# Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
# - **pandas**
#     - **data ingestion and inspection** (pandas Foundations, Module One) 
#     - **exploratory data analysis** (pandas Foundations, Module Two)
#     - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
#     - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
#     - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
#     - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
#     - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
#     - **melting data** (Manipulating DataFrames with pandas, Module Three) 
#     - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
# - **matplotlib** (Intermediate Python for Data Science, Module One)
# - **fundamental data types** (Data Types for Data Science, Module One) 
# - **dictionaries** (Intermediate Python for Data Science, Module Two)
# - **handling dates and times** (Data Types for Data Science, Module Four)
# - **function definition** (Python Data Science Toolbox - Part One, Module One)
# - **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
# - **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

# ## The Data Science Pipeline
# 
# This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting. 
# 
# Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.
# 
# Data Science projects generally adhere to the four stages of Data Science Pipeline:
# 1. Sourcing and loading 
# 2. Cleaning, transforming, and visualizing 
# 3. Modeling 
# 4. Evaluating and concluding 
# 

# ### 1. Sourcing and Loading 
# 
# Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 
# 
# ***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

# #### 1.1. Importing Libraries

# In[1]:


# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np

# Load the pyplot collection of functions from matplotlib, as plt 
import matplotlib.pyplot as plt


# #### 1.2.  Loading the data
# Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets. 

# In[2]:


# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices = "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)

print(properties)


# ### 2. Cleaning, transforming, and visualizing
# This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.
# 
# The end goal of data cleaning is to have tidy data. When data is tidy: 
# 
# 1. Each variable has a column.
# 2. Each observation forms a row.
# 
# Keep the end goal in mind as you move through this process, every step will take you closer. 
# 
# 
# 
# ***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
# - Intermediate Python for data science, all modules.
# - pandas Foundations, all modules. 
# - Manipulating DataFrames with pandas, all modules.
# - Data Types for Data Science, Module Four.
# - Python Data Science Toolbox - Part One, all modules

# **2.1. Exploring your data** 
# 
# Think about your pandas functions for checking out a dataframe. 

# In[3]:


properties.shape
properties.head


# **2.2. Cleaning the data**
# 
# You might find you need to transpose your dataframe, check out what its row indexes are, and reset the index. You  also might find you need to assign the values of the first row to your column headings  . (Hint: recall the .columns feature of DataFrames, as well as the iloc[] method).
# 
# Don't be afraid to use StackOverflow for help  with this.

# In[4]:


tdproperties = properties.T
tdproperties.head


# In[5]:


tdproperties.index
tdproperties = tdproperties.reset_index()

tdproperties.index
tdproperties.head


# In[6]:


tdproperties.columns


# In[7]:


tdproperties.iloc[[0]]


# In[8]:


tdproperties.columns = tdproperties.iloc[0]


# In[9]:


tdproperties.head()


# In[10]:


tdproperties = tdproperties.drop(0)


# In[11]:


tdproperties.head()


# **2.3. Cleaning the data (part 2)**
# 
# You might we have to **rename** a couple columns. How do you do this? The clue's pretty bold...

# In[12]:


tdproperties = tdproperties.rename(columns = {'Unnamed: 0':'London_Borough', pd.NaT: 'ID'})


# In[13]:


tdproperties.head()


# In[14]:


tdproperties.columns


# **2.4.Transforming the data**
# 
# Remember what Wes McKinney said about tidy data? 
# 
# You might need to **melt** your DataFrame here. 

# In[15]:


meltproperties = pd.melt(tdproperties, id_vars= ['London_Borough', 'ID'])


# In[16]:


meltproperties.head()


# In[17]:


meltproperties = meltproperties.rename(columns = {0: 'Month', 'value': 'Average_Prices'})


# In[18]:


meltproperties.head()


# Remember to make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 

# In[19]:


meltproperties.dtypes


# In[20]:


meltproperties['Average_Prices'] = pd.to_numeric(meltproperties['Average_Prices'])


# In[21]:


meltproperties.dtypes


# In[22]:


meltproperties.count()


# **2.5. Cleaning the data (part 3)**
# 
# Do we have an equal number of observations in the ID, Average Price, Month, and London Borough columns? Remember that there are only 32 London Boroughs. How many entries do you have in that column? 
# 
# Check out the contents of the London Borough column, and if you find null values, get rid of them however you see fit. 

# In[23]:


meltproperties['London_Borough'].unique()


# In[24]:


meltproperties[meltproperties['London_Borough'] == 'Unnamed: 34'].head()


# In[25]:


meltproperties[meltproperties['London_Borough'] == 'Unnamed: 37'].head()


# In[26]:


meltproperties[meltproperties['London_Borough'] == 'Unnamed: 47'].head()


# In[27]:


meltproperties[meltproperties['ID'].isna()]


# In[28]:


tidypropertiesdata = meltproperties.dropna()


# In[29]:


tidypropertiesdata.head(50)


# In[30]:


tidypropertiesdata.count()


# In[31]:


tidypropertiesdata['London_Borough'].unique()


# There are only 32 Boroughs but 43 are listed. 
# 
# Quick search pulls up:
# City of London, Barking and Dagenham, Barnet, Bexley, Brent, Bromley, Camden, Croydon, Ealing, Enfield, Greenwich, Hackney, Hammersmith and Fulham, Haringey, Harrow, Havering, Hillingdon, Hounslow, Islington, Kensington and Chelsea, Kingston upon Thames, Lambeth, Lewisham, Merton, Newham, Redbridge, Richmond upon Thames, Southwark, Sutton, Tower Hamlets, Waltham Forest, Wandsworth, Westminster.
# 
# So these are NOT Boroughs:
# 'Inner London', 'Outer London', 'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER', 'EAST MIDLANDS', 'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST', 'SOUTH WEST', 'England'

# In[32]:


Not_Boroughs = ['Inner London', 'Outer London', 'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER', 'EAST MIDLANDS', 'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST', 'SOUTH WEST', 'England']


# In[33]:


tidypropertiesdata = tidypropertiesdata[~tidypropertiesdata.London_Borough.isin(Not_Boroughs)]


# In[34]:


tidypropertiesdata.head()


# In[81]:


BoroughDF = tidypropertiesdata

BoroughDF.sort_values(by=['Average_Prices'], ascending=True)


# In[36]:


BoroughDF.dtypes


# **2.6. Visualizing the data**
# 
# To visualize the data, why not subset on a particular London Borough? Maybe do a line plot of Month against Average Price?

# In[37]:


CityofLondon = BoroughDF[BoroughDF['London_Borough'] == 'City of London']


# In[38]:


z = CityofLondon.plot(kind ='line', x = 'Month', y='Average_Prices')
z.set_ylabel('Price')


# To limit the number of data points you have, you might want to extract the year from every month value your *Month* column. 
# 
# To this end, you *could* apply a ***lambda function***. Your logic could work as follows:
# 1. look through the `Month` column
# 2. extract the year from each individual value in that column 
# 3. store that corresponding year as separate column. 
# 
# Whether you go ahead with this is up to you. Just so long as you answer our initial brief: which boroughs of London have seen the greatest house price increase, on average, over the past two decades? 

# In[39]:


BoroughDF['Year'] = BoroughDF['Month'].apply(lambda a: a.year)

BoroughDF.tail()


# In[59]:


GroupedBDF = BoroughDF.groupby(by=['London_Borough', 'Year']).mean()

GroupedBDF.sample(32)


# In[41]:


GroupedBDF = GroupedBDF.reset_index()

GroupedBDF.head()


# **3. Modeling**
# 
# Consider creating a function that will calculate a ratio of house prices, comparing the price of a house in 2018 to the price in 1998.
# 
# Consider calling this function create_price_ratio.
# 
# You'd want this function to:
# 1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: dfg[dfg['London_Borough']=='Camden'].
# 2. Get the Average Price for that Borough, for the years 1998 and 2018.
# 4. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018.
# 5. Return that ratio.
# 
# Once you've written this function, you ultimately want to use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.
# 
# Bear in mind: you don't have to write a function like this if you don't want to. If you can solve the brief otherwise, then great! 
# 
# ***Hint***: This section should test the skills you acquired in:
# - Python Data Science Toolbox - Part One, all modules

# In[42]:


def year_price_ratio(d):
    year1998 = float(d['Average_Prices'][d['Year']==1998])
    year2018 = float(d['Average_Prices'][d['Year']==2018])
    ratio = [year2018/year1998]
    return ratio


# In[43]:


year_price_ratio(GroupedBDF[GroupedBDF['London_Borough']=='Barking & Dagenham'])


# In[44]:


ratiodict = {}


# In[45]:


for v in GroupedBDF['London_Borough'].unique():
    B = GroupedBDF[GroupedBDF['London_Borough'] == v]
    ratiodict[v] = year_price_ratio(B)
    
print(ratiodict)


# In[46]:


ratiodf = pd.DataFrame(ratiodict)


# In[47]:


ratiodf.head()


# In[48]:


tdratiodf = ratiodf.T
tdratiodf = tdratiodf.reset_index()
tdratiodf.head()


# In[49]:


tdratiodf.rename(columns={'index':'Borough', 0:'2018'}, inplace=True)
tdratiodf.head()


# In[50]:


sortratiodf = tdratiodf.sort_values(by='2018',ascending=False)


# In[51]:


topvalues = sortratiodf.head(15)


# In[52]:


graph = topvalues[['Borough','2018']].plot(kind='bar')

graph.set_xticklabels(topvalues.Borough)


# In[82]:


graphall = sortratiodf[['Borough','2018']].plot(kind='bar')

graph.set_xticklabels(sortratiodf.Borough)


# ### 4. Conclusion
# What can you conclude? Type out your conclusion below. 
# 
# Look back at your notebook. Think about how you might summarize what you have done, and prepare a quick presentation on it to your mentor at your next meeting. 
# 
# We hope you enjoyed this practical project. It should have consolidated your data hygiene and pandas skills by looking at a real-world problem involving just the kind of dataset you might encounter as a budding data scientist. Congratulations, and looking forward to seeing you at the next step in the course! 

# What did you find? 
# I found that prices have raised at least four times as much since 1998 in all 32 boroughs. Hackney has increased in 2018 by 6 times the average of 1998. 
# Which borough is the most expensive? 
# In the most current years, Kensington & Chelsea has been the most expensive places to live. However, Hackney has increased the most in value. 
# Any other interesting trends?
# Barking & Dagenham had the cheapest values in the 1990s. This is exactly the type of data/comparisons that Zillow uses.
# How did you arrive at your conclusion?
# Comparing the data spoke for itself. Comparing the first dataframes, and then the data shown on the graphs. 
# What were the main challenges you encountered?
# Cleaning the data was difficult for me, simply because of syntax and finding the technical terms.
# How did you overcome them? 
# By using other sites like stackoverflow and geeksforgeeks, and the resources I made for myself.
# What could you not overcome?
# Polishing up the graphs are still a challenge that I got too frustrated to fix.
# Is there anything you’d like to investigate deeper?
# I would love to compare this to the average salary of each area. Comparing the ratio of change in salary and housing prices, the change in salaries over the year, and current salaries to home prices. 
