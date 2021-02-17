#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from naive_bayes import NaiveBayes
from sklearn import preprocessing
import seaborn as sb


# In[12]:


covid_data = pd.read_csv('train_data_0.csv')
test_data = pd.read_csv('train_data_9.csv')
covid_data.head(5)


# In[13]:


continues_column = ['age', 'height','weight','days_of_fever']
def normalize(data,column):
    x = data[column].values.reshape(-1,1) #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data[column] = x_scaled
    
for column in continues_column:
    normalize(covid_data,column)
    normalize(test_data,column)


# In[14]:


sb.set_style("darkgrid")
plt.scatter(covid_data[covid_data['test_result'] == True].height, covid_data[covid_data['test_result'] == True].weight, color = 'r')
plt.scatter(covid_data[covid_data['test_result'] == False].height, covid_data[covid_data['test_result'] == False].weight, color = 'b')

plt.title('Covid Data')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()


# In[15]:


print('----------------DataType----------------')
print(covid_data.dtypes)
print('----------------unique value per column----------------')
cat_col_list = list()
for column in covid_data.columns:
    if(not ((covid_data[column].dtypes == np.int64) or (covid_data[column].dtypes == np.float64))):
        print(column,"---->",covid_data[column].unique())
        cat_col_list.append(column)


# In[16]:


pearsoncorr = covid_data.corr(method='pearson')
fig, ax = plt.subplots(figsize=(10,10)) 
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=1, ax=ax)
plt.show()


# ### Data Modification 
# 1)  According to heat map graph, we can that cough, fever and sore_throat column for train data have no effect on test result.
# Hence we can drop these columns<br>
# 2) loss_of_taste and loss_of_smell are correlated to each other. Hence, We can eliminate one out of these two columns.

# In[18]:


def get_fit(df,columns,result_column,continues_column, laplace_coeff):
    x = NaiveBayes(covid_data,columns,result_column,continues_column, laplace_coeff)
    return x

def get_predictions(df, fit):
    return fit.validateSet(test_data)


def evaluate_model(training_data, test_data):  
    continues_column = ['age', 'height','weight','days_of_fever']
    column_for_result = ['age', 'height', 'weight', 'diabetic', 'asthmatic',
                         'alcoholic', 'veteran', 'has_travelled_internationally',
                         'medical_professional', 'essential_worker', 'headache',
                          'loss_of_smell', 'days_of_fever']
    fit = get_fit(covid_data,column_for_result,'test_result', continues_column, 0.8)
    predictions = get_predictions(test_data, fit)
    print('accuracy score -->',accuracy_score(predictions, test_data['test_result'])) 
    return predictions

predictions = evaluate_model(covid_data, test_data)
test_data['predictions'] = predictions


# In[19]:


plt.subplot(1, 2, 1)
plt.scatter(test_data[test_data['test_result'] == True].height, test_data[test_data['test_result'] == True].weight, color = 'r')
plt.scatter(test_data[test_data['test_result'] == False].height, test_data[test_data['test_result'] == False].weight, color = 'b')

plt.title('Correct Covid Test Data')
plt.xlabel('height')
plt.ylabel('weight')

plt.subplot(1, 2, 2)
data = test_data[test_data['predictions'] != test_data['test_result'] ]
plt.scatter(test_data.height, test_data.weight, color = 'g')
plt.scatter(data.height, data.weight, color = 'r')
plt.title('Wrongly predicted Test data in Red')
plt.xlabel('height')
plt.ylabel('weight')

plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_data['test_result'], predictions)

