#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib


#data understanding 
path = r'/Users/ohad/Downloads/SSA2 Enclosure Disks Endurance.xlsx'
df = pd.read_excel(path)
df.head(5)
df.tail(5)


# In[2]:



get_ipython().system('pip install scikit-learn==0.22')


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


for col in df.columns:
    print(col, len(df[col].unique()), df[col].unique())


# In[6]:


df.describe()


# ## Visualize data 
# 

# In[7]:


from matplotlib import pyplot as plt 
import seaborn as sns 


# In[8]:


sns.violinplot(x= 'parent_index', y = 'MAX Spare Count', data = df)


# In[9]:


for col in df.columns:
    sns.violinplot(x='Capacity', y=col, data=df)
    plt.show()


# In[10]:





# Assuming 'df' is your DataFrame and 'column_name' is the column you want to plot
for column_name in df.columns:
    plt.hist(df[column_name])
    plt.title('Histogram of {}'.format(column_name))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()


# In[11]:


df


# In[12]:


#create id for each drive 
id  = df['system_serial'].astype(str)+ df['parent_index'].astype(str)+  df['index'].astype(str)
df.insert(0, "id", id)


# In[13]:


columns_to_drop = ['id', 'system_serial', 'parent_index', 'index']
df1 = df.drop(columns=columns_to_drop)
df1


# In[14]:


df1.dtypes


# In[15]:


sns.lineplot(x = 'created', y = 'power_on_hours' , data = df1)


# In[16]:


df1.corr()


# In[17]:


sns.heatmap(df1.corr())


# In[18]:


df1['Capacity'].unique()


# In[19]:


df1['MAX Spare Count'].unique()


# In[20]:


#create dummey 

pd.get_dummies(df1['Capacity'])
pd.get_dummies(df1['MAX Spare Count'])


# In[21]:


#join the dummey to the regulare df1
df1.join(pd.get_dummies(df1['Capacity']))
df1.join(pd.get_dummies(df1['MAX Spare Count']))


# In[22]:


corrdict1 = {}
for key ,row in df1.join(pd.get_dummies(df1['MAX Spare Count'])).iterrows():
    corrdict1[key] = {int(row['MAX Spare Count']): row['MAX Spare Count']}

corrdict2 = {}
for key ,row in df1.join((pd.get_dummies(df1['Capacity']))).iterrows():
    corrdict2[key] = {int(row['Capacity']): row['Capacity']}


# In[23]:


corrdict1 = pd.DataFrame.from_dict(corrdict1).T.fillna(0)
corrdict2 = pd.DataFrame.from_dict(corrdict2).T.fillna(0)


# In[24]:


#capacity corralation 
corrdict2.corr()


# ## Data Preparation

# In[25]:


import numpy as np 


# In[26]:


for cap in df1['Capacity'].unique():
    plt.figure(figsize=(20, 6))
    sns.lineplot(x='created', y='worst_wear_leveling_count', estimator=np.median, data=df1[df1['Capacity'] == cap])
    plt.title('{} by date'.format(cap))
    plt.show()


# ##  checking dtypes

# In[27]:


df1.dtypes
df1['Capacity'] = df1['Capacity'].astype('category')
df1['MAX Spare Count'] = df1['MAX Spare Count'].astype('category')
df1.dtypes


# ## drop analsys field 

# In[28]:


df1.drop(['created'], axis = 1, inplace = True)


# In[29]:


pd.get_dummies(df1)


# In[30]:


df1.dtypes


# In[31]:


len(df1['total_read_bytes_processed'].unique())


# ## Modeling 

# In[32]:


x = df1.drop('worst_wear_leveling_count', axis = 1)
y = df1['worst_wear_leveling_count'] 


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train, x_test, y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state = 1234)


# In[35]:


print(x_train.shape , x_test.shape, y_train.shape, y_test.shape)


# In[36]:


#train 5 models of ml 


# In[37]:


from sklearn.pipeline import make_pipeline #build ml pipeline 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[38]:


# loop trough each pipline and train it seperately 
pipelines = {
    'rf' : make_pipeline(RandomForestRegressor(random_state=1234)), 
    'gb' : make_pipeline(GradientBoostingRegressor(random_state=1234)), 
    'ridge': make_pipeline(Ridge(random_state=1234)),
    'lasso': make_pipeline(Lasso(random_state=1234)),
    'enet': make_pipeline(ElasticNet(random_state=1234))
    
    
}


# In[39]:


#access into hyper parramater 

RandomForestRegressor().get_params()



# In[40]:


hypergrid = {
    
    
    'rf' : {
        'randomrorestregressor_min_samples_split' :[2,4,6], 
         'randomforestregressor_min_samples_leaf' :[1,2,3]
        
        
    }, 
    'gb' : {
        
        'gradientboostingregressor_alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
        
        
    }, 
    'ridge': {
        
        
        
        'ridge_alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
        
        
    },
    'lasso':{
    
    
    'lasso_alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
    
    
    },
    'enet': {
        
        'enet_alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
        
        
    }
    
    
    
    
}


# In[41]:


from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError


# In[42]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

# Loop through each pipeline and train it separately
pipelines = {
    'rf': make_pipeline(RandomForestRegressor(random_state=1234)),
    'gb': make_pipeline(GradientBoostingRegressor(random_state=1234)),
    'ridge': make_pipeline(Ridge(random_state=1234)),
    'lasso': make_pipeline(Lasso(random_state=1234)),
    'enet': make_pipeline(ElasticNet(random_state=1234))
}

# Access hyperparameters
hypergrid = {
    'rf': {
        'randomforestregressor__min_samples_split': [2, 4, 6],
        'randomforestregressor__min_samples_leaf': [1, 2, 3]
    },
    'gb': {
        'gradientboostingregressor__alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
    },
    'ridge': {
        'ridge__alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
    },
    'lasso': {
        'lasso__alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
    },
    'enet': {
        'elasticnet__alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 0.99]
    }
}

fit_models = {}

for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    try:
        print('Starting training for {}.'.format(algo))
        model.fit(x_train, y_train)
        fit_models[algo] = model
        print('{} has been successfully fit.'.format(algo))
    except NotFittedError as e:
        print(repr(e))


# In[43]:


y_ = fit_models['rf'].predict(x_test)


# ## Evaltion 

# In[44]:


from sklearn.metrics import r2_score , mean_absolute_error


# In[45]:


for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print('{} scores - R2:{} MAE:{}'.format(algo, r2_score(y_test, yhat), mean_absolute_error(y_test, yhat)))


# In[46]:


best_model = fit_models['rf']


# best_model

# ## Deployement

# In[47]:



# Load the CSV file
data = pd.read_csv("input_data.csv")

# Get the feature names
feature_names = ['Capacity', 'MAX Spare Count', 'Power On Hours', 'Total Read Bytes Processed', 'Used Spare Count in Core']

# Get the input values from the CSV data
input_values = data[feature_names].values

# Make the prediction using the trained model
predictions = best_model.predict(input_values)

# Display the predictions
for i, prediction in enumerate(predictions):
    print("Prediction for data point {}: {}".format(i+1, prediction))


# In[ ]:


df


# In[48]:


# total_read_bytes_processed :

column_name = 'total_read_bytes_processed'

# Extract the column values
column_values = df[column_name].values
column_values

# Calculate the mean and standard deviation
mean_trb = np.mean(column_values)
std_trb = np.std(column_values)

# Specify the outlier threshold as a number of standard deviations
std_threshold = 3 # Adjust this value as desired

# Calculate the lower and upper thresholds
#lower_threshold = mean - (std_threshold * std)
upper_threshold_trb = mean_trb + (std_threshold * std_trb)

# Find the outliers based on the thresholds
outliers_trb = df[ (column_values > upper_threshold_trb)]

# Print the outliers
print("Outliers in {}: \n{}".format(column_name, outliers_trb))

upper_threshold_trb


# In[49]:


# power_on_hours

# total_read_bytes_processed :

column_name = 'power_on_hours'

# Extract the column values
column_values = df[column_name].values
column_values

# Calculate the mean and standard deviation
mean_poh = np.mean(column_values)
std_poh = np.std(column_values)

# Specify the outlier threshold as a number of standard deviations
std_threshold = 3 # Adjust this value as desired

# Calculate the lower and upper thresholds
#lower_threshold = mean - (std_threshold * std)
upper_threshold_poh = mean_poh + (std_threshold * std_poh)

# Find the outliers based on the thresholds
outliers_poh = df[ (column_values > upper_threshold_poh)]

# Print the outliers
print("Outliers in {}: \n{}".format(column_name, outliers_poh))

upper_threshold_poh


# In[50]:


# worst_wear_leveling_count

# total_read_bytes_processed :

column_name = 'worst_wear_leveling_count'

# Extract the column values
column_values = df[column_name].values
column_values

# Calculate the mean and standard deviation
mean_wwl = np.mean(column_values)
std_wwl = np.std(column_values)

# Specify the outlier threshold as a number of standard deviations
std_threshold = 3 # Adjust this value as desired

# Calculate the lower and upper thresholds
#lower_threshold = mean - (std_threshold * std)
upper_threshold_wwl = mean_wwl + (std_threshold * std_wwl)

# Find the outliers based on the thresholds
outliers_wwl = df[ (column_values > upper_threshold_wwl)]

# Print the outliers
print("Outliers in {}: \n{}".format(column_name, outliers_wwl))

upper_threshold_wwl


# In[1]:


for i in y_:
    if upper_threshold_wwl < i:
        print(i)


# In[ ]:


import tkinter as tk
from tkinter import messagebox, StringVar
import numpy as np

# Define the feature names and choices
feature_names = ['Capacity', 'MAX Spare Count', 'Power On Hours', 'Total Read Bytes Processed', 'Used Spare Count in Core']
choices = [['3.84', '7.68'], ['1440', '2848']]

# Prepare the input data for prediction
input_values = np.zeros(len(feature_names))

def update_input_value(index, value):
    global input_values
    input_values[index] = float(value)

    


def predict():
    # Make the prediction using the trained model
    prediction = best_model.predict(input_values.reshape(1, -1))
    result_label.config(text="The predicted value is: {}".format(prediction[0]))

    # Calculate status
    power_on_hours = float(entries[2].get())
    total_read_bytes = float(entries[3].get())
    status = calculate_status(power_on_hours, total_read_bytes, prediction[0])
    status_label.config(bg=status)

    return prediction[0]

# Create the GUI window
window = tk.Tk()
window.title("Model Prediction")
window.geometry("400x350")

# Create entry fields for input
entries = []
for i, feature_name in enumerate(feature_names):
    label = tk.Label(window, text=feature_name)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")

    if i < 2:
        choice_var = StringVar(window)
        choice_var.set(choices[i][0])
        dropdown = tk.OptionMenu(window, choice_var, *choices[i], command=lambda value, index=i: update_input_value(index, value))
        dropdown.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(dropdown)
        input_values[i] = float(choices[i][0])
    else:
        entry = tk.Entry(window)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(entry)

        def callback(event, index=i):
            update_input_value(index, event.widget.get())

        entry.bind("<FocusOut>", callback)

# Create result label
result_label = tk.Label(window, text="")
result_label.grid(row=len(feature_names), column=0, columnspan=2, padx=10, pady=5)

# Create status label
status_label = tk.Label(window, width=10, height=2)
status_label.grid(row=len(feature_names)+1, column=0, columnspan=2, padx=10, pady=10)

# Create predict button
predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=len(feature_names)+2, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
window.mainloop()


# In[ ]:


import tkinter as tk
from tkinter import messagebox, StringVar
import numpy as np

# Define the feature names and choices
feature_names = ['Capacity', 'MAX Spare Count', 'Power On Hours', 'Total Read Bytes Processed', 'Used Spare Count in Core']
choices = [['3.84', '7.68'], ['1440', '2848']]

# Prepare the input data for prediction
input_values = np.zeros(len(feature_names))

def update_input_value(index, value):
    global input_values
    input_values[index] = float(value)

def calculate_status(power_on_hours, total_read_bytes, predicted_value):
    # total_read_bytes_processed
    column_name_trb = 'total_read_bytes_processed'
    column_values_trb = df[column_name_trb].values
    mean_trb = np.mean(column_values_trb)
    std_trb = np.std(column_values_trb)
    std_threshold_trb = 3
    upper_threshold_trb = mean_trb + (std_threshold_trb * std_trb)

    # power_on_hours
    column_name_poh = 'power_on_hours'
    column_values_poh = df[column_name_poh].values
    mean_poh = np.mean(column_values_poh)
    std_poh = np.std(column_values_poh)
    std_threshold_poh = 3
    upper_threshold_poh = mean_poh + (std_threshold_poh * std_poh)

    # worst_wear_leveling_count
    column_name_wwl = 'worst_wear_leveling_count'
    column_values_wwl = df[column_name_wwl].values
    mean_wwl = np.mean(column_values_wwl)
    std_wwl = np.std(column_values_wwl)
    std_threshold_wwl = 3
    upper_threshold_wwl = mean_wwl + (std_threshold_wwl * std_wwl)

    # Calculate sum based on thresholds and values
    sum = 0
    if upper_threshold_trb < total_read_bytes:
        sum += 1
    if upper_threshold_poh < power_on_hours:
        sum += 1
    if upper_threshold_wwl < predicted_value:
        sum += 1

    # Return status based on sum
    if sum == 3:
        return 'Red'
    elif sum == 2:
        return 'Yellow'
    else:
        return 'Green'

    
def predict():
    # Make the prediction using the trained model
    prediction = best_model.predict(input_values.reshape(1, -1))
    result_label.config(text="The predicted value is: {}".format(prediction[0]))

    # Calculate status
    power_on_hours = float(entries[2].get())
    total_read_bytes = float(entries[3].get())
    status = calculate_status(power_on_hours, total_read_bytes, prediction[0])
    status_label.config(bg=status)

    return prediction[0]

# Create the GUI window
window = tk.Tk()
window.title("Model Prediction")
window.geometry("400x350")

# Create entry fields for input
entries = []
for i, feature_name in enumerate(feature_names):
    label = tk.Label(window, text=feature_name)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")

    if i < 2:
        choice_var = StringVar(window)
        choice_var.set(choices[i][0])
        dropdown = tk.OptionMenu(window, choice_var, *choices[i], command=lambda value, index=i: update_input_value(index, value))
        dropdown.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(dropdown)
        input_values[i] = float(choices[i][0])
    else:
        entry = tk.Entry(window)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(entry)

        def callback(event, index=i):
            update_input_value(index, event.widget.get())

        entry.bind("<FocusOut>", callback)

# Create result label
result_label = tk.Label(window, text="")
result_label.grid(row=len(feature_names), column=0, columnspan=2, padx=10, pady=5)

# Create status label
status_label = tk.Label(window, width=10, height=2)
status_label.grid(row=len(feature_names)+1, column=0, columnspan=2, padx=10, pady=10)

# Create predict button
predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=len(feature_names)+2, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
window.mainloop()


# In[ ]:





# In[ ]:




