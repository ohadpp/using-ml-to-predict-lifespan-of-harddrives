#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib


# ## Business Understadning 
# -Forecasting the lifespan of drive 

# In[2]:


#data understanding 
path = r'/Users/ohad/Downloads/SSA2 Enclosure Disks Endurance.xlsx'
df = pd.read_excel(path)
df.head(5)
df.tail(5)


# In[3]:


df['MAX Spare Count']


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


for col in df.columns:
    print(col, len(df[col].unique()), df[col].unique())


# In[7]:


df.describe()


# In[8]:


df.dtypes


# ## Visualize data 

# In[9]:


from matplotlib import pyplot as plt 
import seaborn as sns 


# In[10]:


sns.violinplot(x= 'parent_index', y = 'MAX Spare Count', data = df)


# In[11]:


sns.violinplot(x= 'Capacity', y = 'worst_wear_leveling_count', data = df)


# In[12]:


sns.violinplot(x= 'Capacity', y = 'used_spare_count_in_core', data = df) 


# In[13]:


sns.violinplot(x= 'Capacity', y = 'total_read_bytes_processed', data = df)


# In[14]:


sns.violinplot(x= 'Capacity', y = 'power_on_hours', data = df)


# In[15]:


sns.violinplot(x= 'parent_index', y = 'power_on_hours', data = df)


# In[16]:


sns.violinplot(x= 'parent_index', y = 'used_spare_count_in_core', data = df)


# In[17]:


sns.violinplot(x= 'parent_index', y = 'worst_wear_leveling_count', data = df)


# In[18]:


sns.violinplot(x= 'parent_index', y = 'Capacity', data = df)


# In[19]:


plt.figure(figsize = (20,6))
sns.violinplot(x= 'Capacity', y = 'total_read_bytes_processed', data = df[df['Capacity']==3.84])
#sns.violinplot(x= 'parent_index', y = 'Capacity', data = df[df['parent_index']== 4] ).set_title('Parent index type by capacity')
plt.show()


# ### Review Trends 

# In[20]:


df.head() [df[''] == ]


# In[ ]:


sns.lineplot(x = 'created', y = 'total_read_bytes_processed' , data = df)


# In[ ]:


sns.lineplot(x = 'created', y = 'power_on_hours' , data = df)


# In[ ]:


sns.lineplot(x = 'created', y = 'MAX Spare Count' , data = df)


# In[ ]:


sns.lineplot(x = 'created', y = 'used_spare_count_in_core' , data = df)


# In[ ]:


sns.lineplot(x = 'created', y = 'worst_wear_leveling_count' , data = df)


# ### Correlation 

# In[ ]:


df.corr() #from line 4 the pareamaters are relevant for the corrolation


# In[ ]:


df1 = df
df1


# In[ ]:


#creating an specifci id code for each row
df1['id'] = df1['system_serial'].astype(str) + df1['parent_index'].astype(str) + df1['index'].astype(str) 


# In[ ]:





# In[ ]:


from datetime import datetime



# Assuming 'df1' is your DataFrame with a 'created' column

# Convert 'created' column to datetime type
df1['created'] = pd.to_datetime(df1['created'])

# Calculate age by subtracting current date from 'created' date
df1['age'] = (datetime.now().date() - df1['created'].dt.date).dt.days

# Print the DataFrame
df1


# In[ ]:


import numpy as np

# Assuming 'df1' is your DataFrame with a 'Capacity' column

# Create new columns for each capacity type
df1['3.84'] = np.where(df1['Capacity'] == 3.84, df1['Capacity'], 0)
df1['7.68'] = np.where(df1['Capacity'] == 7.68, df1['Capacity'], 0)


# In[ ]:


df1


# In[ ]:


df1.corr()


# In[ ]:





# In[ ]:





# In[ ]:


pd.get_dummies(df['Capacity'])


# In[ ]:


df1.join(pd.get_dummies(df['Capacity']))


# In[ ]:


df['MAX Spare Count'].unique()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:




