#!/usr/bin/env python
# coding: utf-8

# ## Steps that are involved in this notebook while Predicting real Estate Valuation in machine learning are as follows:
# 
# * Gathering data.
# * Cleaning data.
# * Feature engineering.
# * Defining model.
# * Training, testing the model, and predicting the output.

# ### Importing Essential libraries

# In[1]:


import numpy as np
import pandas as pd
import pickle
# from matplotlib import pyplot as plt
# import seaborn as sns
# from matplotlib import rcParams as rcP
# import plotly.express as px 
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset from Drive.

# In[2]:


df = pd.read_csv("Pune house data2.csv")
# df.head()


# # Exploratory Data Analysis

# ## Descriptive statistics of the data

# In[3]:


# describe gives a summary of the statistics (only for numerical columns)
# df.describe(include='all').style.background_gradient("Blues")


# In[4]:


# The above command returns the number of unique values for each variable.
# df.describe(include=[np.object])


# * Used include=np.object for discrete variables.

# In[5]:


# df.dtypes


# In[6]:


# df.info()


# ### Checking dimension of the Dataframe.

# In[7]:


# df.ndim


# * The above dataframe is a 2-dimensional labeled data structure with columns of potentially different data types. 

# In[8]:


# df.shape


# In[9]:


# df.size


# #### The above command return size of the Dataframe which is equivalent to total number of elements. 
# That is Rows X Columns. 

# In[10]:


# Check the number of rows and columns of data
# print(f'Number of Rows = {df.shape[0]} ,  Number of columns = {df.shape[1]}')


# In[11]:


# df.count()


# In[12]:


# Exploring the dataset
# df.groupby('area_type')['area_type'].agg('count')


# In[13]:


# Exploring the dataset
# df.groupby('availability')['availability'].agg('count').sort_values(ascending=False)


# In[ ]:


# Exploring the dataset
# df.groupby('size')['size'].agg('count').sort_values(ascending=False)


# In[ ]:


#Exploring the dataset
# df.groupby('site_location')['site_location'].agg('count')


# # Univariate Analysis
# ### * Numerical Variable
# ### * Categorical Variable

# In[ ]:


# categorical=[]
# numerical=[]
# for i in range(df.columns.size):
#     if df.iloc[:,i].dtype=="object":
#         categorical.append(df.columns[i])
#     else:
#         numerical.append(df.columns[i])


# ## Numerical Variable

# In[ ]:


# def plot_hist(variable):
#     plt.figure(figsize=(15,5))
#     plt.hist(df[variable], bins=50)
#     plt.xlabel(variable)
#     plt.ylabel("Frequency")
#     plt.title("{} Distribution with Histogram".format(variable))
#     plt.show()


# In[ ]:


# for i in numerical:
#     plot_hist(i)


# ## Categorical Variable

# In[ ]:


# def bar_plot(variable):
#     var=df[variable]
#     varValue=var.value_counts()
#     plt.figure(figsize=(15,5))
#     plt.bar(varValue.index, varValue)
#     plt.xticks(varValue.index, varValue.index.values)
#     plt.ylabel("Frequency")
#     plt.title(variable)
#     plt.show()
#     print("{}\n{}".format(variable, varValue))


# In[ ]:


# for i in categorical:
#     bar_plot(i)


# In[ ]:


# df['size'].value_counts().plot.pie(autopct='%.2f',figsize=(12, 12));


# # Bivariate Analysis

# In[ ]:


# fig, ax = plt.subplots(figsize=(12,8))
# corr = df.corr(method='pearson')# plot the heatmap
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[ ]:


# pd.crosstab(df["site_location"], df["size"]).style.background_gradient(cmap='summer_r')


# # Multivariate Analysis 

# In[ ]:


# pd.crosstab([df["site_location"], df["size"]], df["bath"]).style.background_gradient(cmap='summer_r')


# # Data Cleaning Process

# In[3]:


# Drop unnecessary column for better understanding of data.

df = df.drop(['society','area_type','availability'],axis='columns')
# df.head()


# In[4]:


# Checking duplicates

# df.duplicated().sum()


# In[5]:


# df.shape


# In[6]:


# Dropping Duplicates

df = df.drop_duplicates()


# In[7]:


# df.shape


# ## Dealing with null values

# ### Now our goal is to deal with null values and try to understand for each one what can we do: maybe we can drop,replace them or maybe we can just skip them.

# In[19]:


# df.boxplot(column=['bath'],vert=False)


# ### The above boxplot of bath shows data distribution is positive skewed.

# In[20]:


# df.isnull().sum().sort_values(ascending=False)


# In[8]:


# Applying median to the balcony and  bath column.
from math import floor
bath_median = float(floor(df.bath.median()))
df.bath = df.bath.fillna(bath_median)


# In[9]:


df.balcony=df.balcony.mask(df.balcony==0).fillna(floor(df.balcony.median()))


# In[10]:


# df.balcony.value_counts()


# In[11]:


# df.bath.value_counts()


# In[12]:


# Checking the null values in the dataset again
# df.isnull().sum()


# In[13]:


# Dropping the rows with null values because the dataset is huge as compared to null values.

df = df.dropna()
# df.isnull().sum()


# In[14]:


# Converting the size column to bhk
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
df = df.drop('size', axis='columns')
df.groupby('bhk')['bhk'].agg('count')


# * Handlling Outlier

# In[31]:


# df.boxplot(column=['bhk'])


# In[15]:


# Outliers Calculation
def min_max_iqr(df,col):
    q1, q3 = df[col].quantile([0.25,0.75])
    IQR = q3-q1
    min_valid = q1 - 1.5*IQR
    max_valid = q3 + 1.5 * IQR
    return min_valid,max_valid


# In[16]:


min_valid,max_valid = min_max_iqr(df,'bhk')
# print("Lower outlier Range = ",min_valid)
# print("Higher outlier Range = ",max_valid)


# In[17]:


# Dropping rows more than 5 BHK

df = df.drop(df[df['bhk']>=5].index)
# df.groupby('bhk')['bhk'].count()


# In[18]:


df = df.drop(df[df['bath']>4].index)


# In[19]:


# num_coln = df.select_dtypes(include=np.number).columns.tolist()
# bins=10
# j=1
# fig = plt.figure(figsize = (20, 30))
# for i in num_coln:
#     plt.subplot(7,4,j)
#     plt.boxplot(df[i])
#     j=j+1
#     plt.xlabel(i)
#     plt.legend()
# plt.show()


# In[20]:


# Check the unique value in total_sqft
# df.total_sqft.unique()


#  * Since the total_sqft contains range values such as 1133-1384, lets filter out these values

# In[21]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[22]:


# Displaying all the rows that are not integers

df[~df['total_sqft'].apply(isfloat)]


# #### Now converting all values of total_sqft into a float value

# In[23]:


# Converting the range values to integer values and removing other types of error
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[24]:


df['new_total_sqft'] = df.total_sqft.apply(convert_sqft_to_num)
df = df.drop('total_sqft', axis='columns')
# df.head()


# In[25]:


# Removing the rows in new_total_sqft column that hase None values
# df.isna().sum()


# In[26]:


# Removing the rows in new_total_sqft column that hase None values
df = df.dropna()
# df.isna().sum()


# # Feature Enginnering for outlier detection dimension reduction

# * Feature engineering is the pre-processing step of machine learning, which is used to transform raw data into features that can be used for creating a predictive model using Machine learning or statistical Modelling.

# In[27]:


# Adding a new column of price_per_sqft
df1 = df.copy()

# In our dataset the price column is in Lakhs
df1['price_per_sqft'] = (df1['price']*100000)/df1['new_total_sqft']
# df1.head()


# In[28]:


# Checking unique values of 'location' column
locations = list(df['site_location'].unique())
# print(len(locations))


# In[29]:


# Removing the extra spaces at the end
df1.site_location = df1.site_location.apply(lambda x: x.strip())

# Calulating all the unqiue values in 'site_location' column
location_stats = df1.groupby('site_location')['site_location'].agg('count').sort_values(ascending=False)
# location_stats


# In[30]:


# Checking locations with less than 10 values
# print(len(location_stats[location_stats<=10]), len(df1.site_location.unique()))


# In[31]:


# df1.head()


# In[32]:


# Labelling the locations with less than or equal to 10 occurences to 'other'
locations_less_than_10 = location_stats[location_stats<=10]

df1.site_location = df1.site_location.apply(lambda x: 'other' if x in locations_less_than_10 else x)
# len(df1.site_location.unique())


# In[33]:


# df1.head()


# # Removing Outliers

# In[34]:


# df1.boxplot(column=['new_total_sqft'],vert=False)


# In[35]:


# Removing the rows that have 1 Room for less than 300sqft

df2 = df1[~(df1.new_total_sqft/df1.bhk<300)]
# print("Total rows whose Total Area per bhk is below 300 sqft = ",len(df2))
# print("Total rows whose Total Area per bhk is above 300 sqft = ",len(df1))
# # print(len(df2), len(df1))


# In[36]:


# df2.boxplot('price_per_sqft',vert=False)


# In[37]:


# df2.price_per_sqft.describe()


# In[38]:


# Since there is a wide range for 'price_per_sqft' column with min = Rs.267/sqft till max = Rs. 127470/sqft, we remove the extreme ends using the mean and SD
def remove_pps_outliers(df):
    
    df_out = pd.DataFrame()
    
    for key, sub_df in df.groupby('site_location'):
        m = np.mean(sub_df.price_per_sqft)
        sd = np.std(sub_df.price_per_sqft)
        reduce_df = sub_df[(sub_df.price_per_sqft>(m-sd)) & (sub_df.price_per_sqft<(m+sd))]
        df_out = pd.concat([df_out, reduce_df], ignore_index=True)
    
    return df_out

df3 = remove_pps_outliers(df2)
# print("Length of data before removing outlier = ",len(df2))
# print("Length of data before after outlier = ",len(df3))


# In[39]:


# def plot_scatter_chart(df, site_location):
#     bhk2 = df[(df.site_location == site_location) & (df.bhk == 2)]
#     bhk3 = df[(df.site_location == site_location) & (df.bhk == 3)]
#     rcP['figure.figsize'] = (15,10)
#     plt.scatter(bhk2.new_total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
#     plt.scatter(bhk3.new_total_sqft, bhk3.price, color='green', marker='+', label='3 BHK', s=50)
#     plt.xlabel('Total Square Feet Area')
#     plt.ylabel('Price (in Lakhs)')
#     plt.title(site_location)
#     plt.legend()
    
# plot_scatter_chart(df3, 'Hadapsar')


# In[40]:


# Here we observe that 3 BHK cost that same as 2 BHK in 'Hadapsar' location hence removing such outliers is necessary
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    
    for site_location, site_location_df in df.groupby('site_location'):
        bhk_stats = {}
        
        for bhk, bhk_df in site_location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        
        for bhk, bhk_df in site_location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    return df.drop(exclude_indices, axis='index')

df4 = remove_bhk_outliers(df3)
# print(len(df3), len(df4))


# In[41]:


# plot_scatter_chart(df4, 'Hadapsar')


# In[42]:


# plt.hist(df4.price_per_sqft, rwidth=0.5)
# plt.xlabel('Price Per Square Feet')
# plt.ylabel('Count')


# In[43]:


# plt.hist(df4.bath, rwidth=0.5)
# plt.xlabel('Number of Bathrooms')
# plt.ylabel('Count')


# In[44]:


# Removing the rows that have 'bath' greater than 'bhk'+1
df5 = df4[df4.bath<(df4.bhk+1)]
# print(len(df4), len(df5))


# In[45]:


# df5.shape


# In[46]:


# df5.tail()


# In[47]:


# pd.crosstab([df5["site_location"], df["bhk"]], df["bath"]).style.background_gradient(cmap='summer_r')


# # Model Building

# In[48]:


# Removing the unnecessary columns (columns that were added only for removing the outliers)
df6 = df5.copy()
df6 = df6.drop('price_per_sqft', axis='columns')


# In[ ]:


# df6.to_csv('.csv') 


# In[49]:


# df6.head()


# In[50]:


# fig, ax = plt.subplots(figsize=(12,8))
# corr = df6.corr(method='pearson')# plot the heatmap
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# ## Use One Hot Encoding For site_location

# In[51]:


# Converting the categorical_value into numerical_values using get_dummies method
dummy_cols = pd.get_dummies(df6.site_location)
df6 = pd.concat([df6,dummy_cols], axis='columns')


# In[52]:


df6.drop(['site_location'], axis='columns', inplace=True)
# df6.head(10)


# In[53]:


# Size of the dataset
# df6.shape


# In[54]:


# df6.to_csv('cleaned_data.csv')


# In[55]:


# Splitting the dataset into features and label
X = df6.drop('price', axis='columns')
y = df6['price']


# In[56]:


# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)


# In[57]:


# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import ShuffleSplit
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
# from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb
# from xgboost.sklearn import XGBRegressor


# In[58]:


# Creating a function for GridSearchCV

# def find_best_model(X, y):
#     models = {
#         'linear_regression': {
#             'model': LinearRegression(),
#             'parameters': {
#                 'normalize': [True,False]
#             }
#         },
        
#         'lasso': {
#             'model': Lasso(),
#             'parameters': {
#                 'alpha': [1,2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
        
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'parameters': {
#                 'criterion': ['mse', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         },
#         'GradientBoostRegressor':
#         {
#             'model': GradientBoostingRegressor(),
#             'parameters' :
#             {
#                 'n_estimators' : [ 50, 100, 150, 200],
#                 'subsample' : [0.6, 0.7, 0.8],
#                 'max_depth' :[ 5,6,7,8]
#             }
#         },
        
#         'Ridge':
#         {
#             'model':Ridge(),
#             'parameters':
#             {
#                 'alpha': [1,0.1,0.01,0.001,0.0001,0] ,
#                 "fit_intercept": [True, False],
#                 "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
#             }
#         },
        
#         'XGBRegressor':
#         {
#             'model':XGBRegressor(),
#             'parameters':
#             {
#               'nthread':[4], 
#               'objective':['reg:linear'],
#               'learning_rate': [.03, 0.05, .07], 
#               'max_depth': [5, 6, 7],
#               'min_child_weight': [4],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7],
#               'n_estimators': [500]
#             }
#         }
# #         'RandomForestRegressor':
# #        {
# #             'model': RandomForestRegressor(oob_score=True),
# #             'parameters':
# #             {
# #                 'n_estimators' : [ 50, 200, 500],
# #                 'criterion' : ['squared_error', 'absolute_error', 'poisson'],
# #                 'max_depth' :[ 3, 5,7],
# #                 'max_features' : ['sqrt', 'log2'],
# #                 'max_samples' : [0.7, 0.8, 0.9]
# #             }
# #         }
#     }
    
#     scores = []
#     cv_X_y = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    
#     for model_name, model_params in models.items():
#         gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=cv_X_y, return_train_score=False)
#         gs.fit(X, y)
#         scores.append({
#             'model': model_name,
#             'best_parameters': gs.best_params_,
#             'accuracy': gs.best_score_
#         })
        
#     return pd.DataFrame(scores, columns=['model', 'best_parameters', 'accuracy'])

# find_best_model(X_train, y_train)


# * Since the accuracy of Ridge Regression is Higher. Ridge Regression is Selected for training model

# In[71]:


model = Ridge(alpha = 1, fit_intercept = True, solver ='cholesky', random_state = 7)
model.fit(X_train, y_train)
# y_pred = model.predict(X_test)


# In[72]:


# model.score(X_test, y_test)


# In[61]:


# import pickle
pickle.dump(model, open('model2.pkl','wb'))
model2 = pickle.load(open('model2.pkl','rb'))


# * Based on above results we can say that Ridge Regression gives the best score. Hence we will use that.

# ### Predicting the Values using our trained model

# In[62]:


# X.columns


# In[63]:


# For finding the appropriate location
# np.where(X.columns=='Balaji Nagar')[0][0]


# In[64]:


# Creating a fuction to predict values

def prediction(location, bhk, bath, balcony, sqft):
    
    loc_index, area_index, avail_index = -1,-1,-1
        
    if location!='other':
        loc_index = int(np.where(X.columns==location)[0][0])
            
    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft
    
    if loc_index >= 0:
        x[loc_index] = 1 
    if x[3] in range(300,10000):
        x[3] = sqft
    else:
        return False
    return model.predict([x])[0]


# In[80]:


# df_check = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df_check = df_check.sample(25)
# #round(df_check,2)
# df_check.plot(kind='bar',figsize=(10,5))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.title('Performance of Random Forest')
# plt.xlabel('Random Rows')
# plt.ylabel('Price')
# # plt.savefig('Random-Forest-Performance.jpg')
# plt.show()


# In[ ]:


# locax = input("Enter the location : ")
# size = int(input("Enter BHK :"))
# bathing = int(input("Enter Bathroom : "))
# bal = int(input("Enter balcony : "))
# feet = int(input("Enter sqft : "))
# output = prediction(locax, size, bathing, bal, feet)
# print(round(output,2),"Lakhs")


# In[69]:


# Prediction 1
# Input in the form : Location, BHK, Bath, Balcony, Sqft.

# output = prediction('Alandi Road', 2, 2, 2,12000)
# if output == False:
#     print("Enter sqft in range (300-5000)")
# else:
#     print(output)


# In[67]:


# Prediction 1
# Input in the form : Location, BHK, Bath, Balcony, Sqft.
# prediction('Balaji Nagar', 2, 2, 1, 2000)


# In[ ]:


# Prediction 2
# Input in the form : Location, BHK, Bath, Balcony, Sqft.
# prediction('Hadapsar', 3, 2, 1, 1200)


# In[ ]:


# Prediction 3
# Input in the form : Location, BHK, Bath, Balcony, Sqft.
# prediction('Camp', 3, 3, 2, 1800)


# In[ ]:


# Prediction 4
# Input in the form : Location, BHK, Bath, Balcony, Sqft.
# prediction('Baner', 4, 3, 2, 2200)


# In[ ]:


# Prediction 4
# Input in the form : Location, BHK, Bath, Balcony, Sqft.
# prediction('Tilak Road', 5, 4, 3, 2500)

