#!/usr/bin/env python
# coding: utf-8

# # IMPORT ALL THE NECESSARY LIBARAIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV  
from sklearn.pipeline import Pipeline
from matplotlib import gridspec
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import squarify
import ast
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import warnings
import plotly.graph_objects as go
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from datetime import datetime
import squarify
get_ipython().run_line_magic('matplotlib', 'inline')


# # IMPORT DATASET AND CHECK THROUGH DATA

# In[2]:


# Import the Excel data file
excel_file_path = "C:/Users/Administrator/Desktop/bradford project/bisi/e-commernce data.xlsx"
data = pd.read_excel(excel_file_path)
# Display the data
print(data)


# In[3]:


# display the dataframe
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# # DESCRIPTIVE STATISTICS 

# In[6]:


# Select the numerical columns
numerical_columns = data.select_dtypes(include=['float', 'int'])

# Calculate descriptive statistics
descriptive_stats = numerical_columns.describe()

# Print the descriptive statistics
print(descriptive_stats)


# In[7]:


#categorical_column(s) in dataset
categorical_columns = ['category', 'price', 'event_type','brand']

for column in categorical_columns:
   column_stats = data[column].value_counts()  
   total_count = column_stats.sum()  
   unique_count = column_stats.size  
   mode = column_stats.idxmax()  
   mode_count = column_stats.max()  
   mode_percentage = mode_count / total_count * 100  
   print(f"Column: {column}")
   print(f"Total count: {total_count}")
   print(f"Unique count: {unique_count}")
   print(f"Mode: {mode}")
   print(f"Mode count: {mode_count}")
   print(f"Mode percentage: {mode_percentage:.2f}%")
   print("----------------------")


# # CHECKING FOR MISSING VALUES

# In[8]:


# Check for missing values in the DataFrame
missing_values = data.isnull().sum()
# Check for missing values in specific columns
missing_values_per_column = data.isnull().sum()
# Print the results
print("Missing Values in the DataFrame:")
print(missing_values)
print("\nMissing Values in Specific Columns:")
print(missing_values_per_column)


# In[9]:


# Imputing missing values for categorical columns
categorical_cols = ['brand','category']
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Checking for missing values again to verify the changes
missing_values_after_handling = data.isnull().sum()
print(missing_values_after_handling)


# # EXPLORATORY DATA ANALYSIS (EDA)

# In[10]:


def convert_time_to_date(utc_timestamp):    
    '''covert utc timestamp string to date yyyy-mm-dd format in datetime object
    
    Parameters: 
        utc_timestamp (str): utc timestamp string is to be converted.
    
    Returns:
        utc_date (datetime): datetime object for date in format yyyy-mm-dd
    '''
    
    utc_date = datetime.strptime(utc_timestamp[0:10], '%Y-%m-%d').date()
    return utc_date   


# In[11]:


# Due to the amount of data, datetime conversion takes a while....
data['event_date'] = data['event_time'].apply(lambda s: convert_time_to_date(s))
visitor_by_date = data[['event_date','customer_id']].drop_duplicates().groupby(['event_date'])['customer_id'].agg(['count']).sort_values(by=['event_date'])


# In[12]:


x = pd.Series(visitor_by_date.index.values)
y = visitor_by_date['count']
plt.rcParams['figure.figsize'] = (20,8)
plt.plot(x,y)
plt.show()


# In[13]:


# daily price trend
product_id = 1003461 # Enter product_id
data[data['product_id'] == product_id][['category','brand']].head(1)


# In[14]:


product_daily_price = data.loc[data['product_id'] == product_id,['event_date','price']].groupby(['event_date']).mean()
product_daily_price = data[['event_date','price']].groupby(['event_date']).mean()


# In[15]:


#how many customers visted the site
visitor = data['customer_id'].nunique()
print ("visitors: {}".format(visitor))


# In[16]:


data['brand'].value_counts()
data['event_type'].value_counts()


# In[17]:


#by category and product
top_category_n = 30
top_category = data['category'].value_counts()[:top_category_n].sort_values(ascending=False)
df = pd.DataFrame({'count':top_category, 'top_category':top_category })

squarify.plot(sizes=top_category, label=top_category.index.array, color=["red","cyan","green","orange","blue","grey"], alpha=.7  )
plt.axis('off')
plt.show()


# In[18]:


top_brand_n = 30
top_brand = data['brand'].value_counts()[:top_brand_n].sort_values(ascending=False)
df = pd.DataFrame({'count':top_brand, 'top_category':top_brand.index.array })

squarify.plot(sizes=top_brand, label=top_brand.index.array, color=["red","cyan","green","orange","blue","grey"], alpha=.7  )
plt.axis('off')
plt.show()


# In[19]:


labels = ['view', 'cart','purchase']
size = data['event_type'].value_counts()
colors = ['yellowgreen', 'lightskyblue','lightcoral']
explode = [0, 0.1,0.1]
plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Event_Type', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[20]:


#items that customer bought
purchase = data.loc[data['event_type'] == 'purchase']
purchase = purchase.dropna(axis='rows')
purchase


# In[21]:


#top buy by customer
top_sellers = purchase.groupby('brand')['brand'].agg(['count']).sort_values('count', ascending=False)
top_sellers.head(20)


# In[22]:


#users journey
user_session = 520088904
data.loc[data['customer_id'] == user_session]


# # FEATURE ENGINERRING

# In[23]:


#predict if a customer will buy a product after adding to a cart
cart_purchase_users = data.loc[data["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['customer_id'])
cart_purchase_users.dropna(how='any', inplace=True)


# In[24]:


cart_purchase_users_all_activity = data.loc[data['customer_id'].isin(cart_purchase_users['customer_id'])]


# In[25]:


#Prepare a dataframe for counting and checking activity in the session
activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()
activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})


# In[26]:


df_targets = data.loc[data["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['event_type', 'product_id','price', 'customer_id',
'user_session'])
df_targets["is_purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)
df_targets["is_purchased"] = df_targets.groupby(["user_session","product_id"])["is_purchased"].transform("max")
df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","is_purchased"])
df_targets['event_weekday'] = df_targets['event_date'].apply(lambda s: s.weekday())
df_targets.dropna(how='any', inplace=True)
df_targets["category_level1"] = df_targets["category"].str.split(".",expand=True)[0].astype('category')
df_targets["category_level2"] = df_targets["category"].str.split(".",expand=True)[1].astype('category')


# In[27]:


df_targets = df_targets.merge(activity_in_session, on='user_session', how='left')
df_targets['activity_count'] = df_targets['activity_count'].fillna(0)
df_targets.head()


# In[28]:


df_targets.info()


# In[29]:


df_targets.to_excel('training_data.xlsx')
df_targets = pd.read_excel('training_data.xlsx')


# In[30]:


#working on inbalance in trainig
is_purcahase_set = df_targets[df_targets['is_purchased']== 1]
is_purcahase_set.shape[0]


# In[31]:


not_purcahase_set = df_targets[df_targets['is_purchased']== 0]
not_purcahase_set.shape[0]


# In[32]:


n_samples = 4943  # Reduce the number of samples
is_purchase_downsampled = resample(is_purcahase_set,
                                   replace=False,
                                   n_samples=n_samples,
                                   random_state=27)
not_purchase_set_downsampled = resample(not_purcahase_set,
                                        replace=False,
                                        n_samples=n_samples,
                                        random_state=27)


# In[33]:


downsampled = pd.concat([is_purchase_downsampled, not_purchase_set_downsampled])
downsampled['is_purchased'].value_counts()


# In[38]:


features = downsampled[['brand', 'price', 'event_weekday', 'category_level1', 'category_level2', 'activity_count']]


# In[39]:


#Encode categorical variables
features.loc[:, 'brand'] = LabelEncoder().fit_transform(downsampled.loc[:, 'brand'].copy())
features.loc[:, 'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:, 'event_weekday'].copy())
features.loc[:, 'category_level1'] = LabelEncoder().fit_transform(downsampled.loc[:, 'category_level1'].copy())
features.loc[:, 'category_level2'] = LabelEncoder().fit_transform(downsampled.loc[:, 'category_level2'].copy())

is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
features.head()


# In[40]:


# Separate customers who made a purchase and customers who did not make a purchase
click_and_buy = downsampled[downsampled['is_purchased'] == 1]
click_and_not_buy = downsampled[downsampled['is_purchased'] == 0]

# Select the features for clustering (you can choose different features if needed)
click_and_buy_features = click_and_buy[['brand', 'price', 'event_weekday', 'category_level1', 'category_level2', 'activity_count']]
click_and_not_buy_features = click_and_not_buy[['brand', 'price', 'event_weekday', 'category_level1', 'category_level2', 'activity_count']]


# In[41]:


# Encode categorical variables for both groups
click_and_buy_features.loc[:, 'brand'] = LabelEncoder().fit_transform(click_and_buy.loc[:, 'brand'].copy())
click_and_buy_features.loc[:, 'event_weekday'] = LabelEncoder().fit_transform(click_and_buy.loc[:, 'event_weekday'].copy())
click_and_buy_features.loc[:, 'category_level1'] = LabelEncoder().fit_transform(click_and_buy.loc[:, 'category_level1'].copy())
click_and_buy_features.loc[:, 'category_level2'] = LabelEncoder().fit_transform(click_and_buy.loc[:, 'category_level2'].copy())

click_and_not_buy_features.loc[:, 'brand'] = LabelEncoder().fit_transform(click_and_not_buy.loc[:, 'brand'].copy())
click_and_not_buy_features.loc[:, 'event_weekday'] = LabelEncoder().fit_transform(click_and_not_buy.loc[:, 'event_weekday'].copy())
click_and_not_buy_features.loc[:, 'category_level1'] = LabelEncoder().fit_transform(click_and_not_buy.loc[:, 'category_level1'].copy())
click_and_not_buy_features.loc[:, 'category_level2'] = LabelEncoder().fit_transform(click_and_not_buy.loc[:, 'category_level2'].copy())


# In[42]:


# Perform K-means clustering for both groups
n_clusters = 3
kmeans_click_and_buy = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_click_and_buy.fit(click_and_buy_features)

kmeans_click_and_not_buy = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_click_and_not_buy.fit(click_and_not_buy_features)

# Get the cluster labels for both groups
click_and_buy_cluster_labels = kmeans_click_and_buy.labels_
click_and_not_buy_cluster_labels = kmeans_click_and_not_buy.labels_

# Add the cluster labels to the corresponding dataframes
click_and_buy['cluster_label'] = click_and_buy_cluster_labels
click_and_not_buy['cluster_label'] = click_and_not_buy_cluster_labels

# Print the count of samples in each cluster for both groups
print("Click and Buy Cluster Counts:")
print(click_and_buy['cluster_label'].value_counts())

print("Click and Not Buy Cluster Counts:")
print(click_and_not_buy['cluster_label'].value_counts())


# In[43]:


# Plot the clusters for customers who made a purchase
plt.scatter(click_and_buy_features['price'], click_and_buy_features['activity_count'], c=click_and_buy_cluster_labels)
plt.xlabel('Price')
plt.ylabel('Activity Count')
plt.title('Clusters - Customers who made a purchase')
plt.show()

# Plot the clusters for customers who did not make a purchase
plt.scatter(click_and_not_buy_features['price'], click_and_not_buy_features['activity_count'], c=click_and_not_buy_cluster_labels)
plt.xlabel('Price')
plt.ylabel('Activity Count')
plt.title('Clusters - Customers who did not make a purchase')
plt.show()


# In[44]:


#Modeling
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    is_purchased, 
                                                    test_size = 0.3, 
                                                    random_state = 0)


# In[45]:


# Define the target variable
target = downsampled['is_purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1-Score:", metrics.f1_score(y_test, y_pred))


# In[45]:


#Compute the false positive rate (fpr) and true positive rate (tpr)
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Compute the Area Under the Curve (AUC) for ROC
roc_auc = roc_auc_score(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])


# In[46]:


# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1-Score:", metrics.f1_score(y_test, y_pred))


# In[47]:


# Initialize and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the area under the ROC curve
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[48]:


model = XGBClassifier(learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[49]:


#evaluate perfoemance
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("fbeta:",metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))


# In[50]:


# Calculate predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Compute Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()


# In[51]:


# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

# Evaluate KNN performance
print("KNN Accuracy:", metrics.accuracy_score(y_test, knn_y_pred))
print("KNN Precision:", metrics.precision_score(y_test, knn_y_pred))
print("KNN Recall:", metrics.recall_score(y_test, knn_y_pred))
print("KNN F-beta:", metrics.fbeta_score(y_test, knn_y_pred, average='weighted', beta=0.5))


# In[52]:


# Calculate predicted probabilities
knn_y_pred_prob = knn_model.predict_proba(X_test)[:, 1]

# Compute false positive rate, true positive rate, and thresholds
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_y_pred_prob)

# Compute Area Under the Curve (AUC)
knn_roc_auc = auc(knn_fpr, knn_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')
plt.legend(loc='lower right')
plt.show()


# In[53]:


# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

# Evaluate Decision Tree performance
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, dt_y_pred))
print("Decision Tree Precision:", metrics.precision_score(y_test, dt_y_pred))
print("Decision Tree Recall:", metrics.recall_score(y_test, dt_y_pred))
print("Decision Tree F-beta:", metrics.fbeta_score(y_test, dt_y_pred, average='weighted', beta=0.5))


# In[54]:


# Calculate predicted probabilities
dt_y_pred_prob = dt_model.predict_proba(X_test)[:, 1]

# Compute false positive rate, true positive rate, and thresholds
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_y_pred_prob)

# Compute Area Under the Curve (AUC)
dt_roc_auc = auc(dt_fpr, dt_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()


# In[55]:


# Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# Evaluate SVM performance
print("SVM Accuracy:", metrics.accuracy_score(y_test, svm_y_pred))
print("SVM Precision:", metrics.precision_score(y_test, svm_y_pred))
print("SVM Recall:", metrics.recall_score(y_test, svm_y_pred))
print("SVM F-beta:", metrics.fbeta_score(y_test, svm_y_pred, average='weighted', beta=0.5))


# In[56]:


# Calculate predicted probabilities
svm_y_pred_prob = svm_model.decision_function(X_test)

# Compute false positive rate, true positive rate, and thresholds
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_y_pred_prob)

# Compute Area Under the Curve (AUC)
svm_roc_auc = auc(svm_fpr, svm_tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label=f'Support Vector Machine (AUC = {svm_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Support Vector Machine')
plt.legend(loc='lower right')
plt.show()


# In[59]:


# Get the true and predicted values for XGBoost
y_true = y_test
y_pred = xgb_model.predict(X_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - XGBoost')
plt.show()


# In[60]:


# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[63]:


# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy and create a confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for True and Predicted values
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title(f'Random Forest Classifier\nAccuracy: {accuracy:.2f}')
plt.show()


# In[62]:


# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy and create a confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for True and Predicted values
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title(f'Random Forest Classifier\nAccuracy: {accuracy:.2f}')
plt.show()


# In[64]:


# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate accuracy and create a confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for True and Predicted values
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title(f'Decision Tree Classifier\nAccuracy: {accuracy:.2f}')
plt.show()


# In[69]:


# Define the target variable
target = downsampled['is_purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set for Logistic Regression
y_pred_lr = lr_model.predict(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Make predictions on the test set for Random Forest
y_pred_rf = rf_model.predict(X_test)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)

# Make predictions on the test set for Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set for XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Calculate the performance metrics for each model
metrics_lr = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
}

metrics_rf = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf)
}

metrics_dt = {
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'Precision': precision_score(y_test, y_pred_dt),
    'Recall': recall_score(y_test, y_pred_dt),
    'F1-Score': f1_score(y_test, y_pred_dt)
}

metrics_xgb = {
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1-Score': f1_score(y_test, y_pred_xgb)
}

# Create a bar plot to compare the performance metrics for each algorithm
labels = list(metrics_lr.keys())
lr_values = list(metrics_lr.values())
rf_values = list(metrics_rf.values())
dt_values = list(metrics_dt.values())
xgb_values = list(metrics_xgb.values())

x = range(len(labels))

plt.bar(x, lr_values, width=0.2, label='Logistic Regression', color='b', align='center')
plt.bar([i + 0.2 for i in x], rf_values, width=0.2, label='Random Forest', color='g', align='center')
plt.bar([i + 0.4 for i in x], dt_values, width=0.2, label='Decision Tree', color='r', align='center')
plt.bar([i + 0.6 for i in x], xgb_values, width=0.2, label='XGBoost', color='purple', align='center')

plt.xticks([i + 0.3 for i in x], labels)
plt.xlabel('Performance Metrics')
plt.ylabel('Score')
plt.title('Comparison of Performance Metrics for Each Algorithm')
plt.legend()
plt.show()


# In[81]:


# Calculate the Mean Absolute Error for each model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Create a bar plot to compare the MAE values for each algorithm
algorithms = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBoost']
mae_values = [mae_lr, mae_rf, mae_dt, mae_xgb]

plt.bar(algorithms, mae_values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Algorithms')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Mean Absolute Error (MAE) for Each Algorithm')

# Rotate the x-axis labels by 45 degrees for better readability
plt.xticks(rotation=45, ha='right')

# Annotate the bars with their respective MAE values
for i in range(len(algorithms)):
    plt.text(i, mae_values[i] + 0.001, f'{mae_values[i]:.4f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.show()


# In[56]:


#feature importance
plot_importance(model, max_num_features=10, importance_type ='gain')
plt.rcParams['figure.figsize'] = (10,7)
plt.show()


# In[58]:


import matplotlib.pyplot as plt

# Create a bar chart to visualize the predicted values
unique_classes, class_counts = np.unique(y_pred, return_counts=True)

plt.bar(unique_classes, class_counts)
plt.xticks(unique_classes, ['Not Purchased', 'Purchased'])
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.title('Predicted Class Distribution')
plt.show()

