#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


datafile_name = "Datasets\Life_Expectancy\Life Expectancy Data.csv"


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[4]:


# Import data from the csv file without index columns

df = pd.read_csv(datafile_name,index_col=False)
df.head()


# In[5]:


print(df.isna().sum())


# In[6]:


print(df.isna().any().any())


# In[7]:


country_list =  {}
# Creating a function add country and status into a dictionary
def add_country(country, status):
    if country not in country_list:
        country_list[country] = status

for index, row in df.iterrows():
    add_country(row['Country'], row['Status'])

country_list = pd.DataFrame(list(country_list.items()), columns=['Country', 'Status'])
ountry_list = country_list.to_csv('Datasets/Life_Expectancy/country_list.csv', index=False)


# In[8]:


X = df.drop(['Life expectancy ', 'Status','Country','Year'], axis=1)

X.head()


# In[9]:


df.dropna(inplace=True)


# In[10]:


# Split the dataset

X = torch.tensor(df.drop(['Life expectancy ', 'Status','Country','Year'], axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Life expectancy '].values, dtype=torch.float32)

X, y


# In[11]:


scaler = StandardScaler()

X = torch.tensor(scaler.fit_transform(X))
y = torch.tensor(scaler.fit_transform(y.reshape(-1, 1))).flatten()


# In[12]:


X_train , X_test_train,y_train, y_test_train = train_test_split(X,y,test_size=0.2)
X_validation, X_test_val, y_validation, y_test_val = train_test_split(X_train, y_train, test_size = 0.2)


# In[13]:


X_train = X_train.clone().detach().to(torch.float32)
X_test_train = X_test_train.clone().detach().to(torch.float32)
X_validation = X_validation.clone().detach().to(torch.float32)
X_test_val = X_test_val.clone().detach().to(torch.float32)
y_train = y_train.clone().detach().to(torch.float32)
y_test_train = y_test_train.clone().detach().to(torch.float32)
y_validation = y_validation.clone().detach().to(torch.float32)
y_test_val = y_test_val.clone().detach().to(torch.float32)


# In[14]:


X_train[0]


# In[15]:


class MultilinearRegression(nn.Module):
    def __init__(self, input_size: int):
        super(MultilinearRegression, self).__init__()  # Fix the super call
        self.linear = nn.Linear(input_size, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# In[16]:


# Create instance
model = MultilinearRegression(X_train.shape[1])

model.state_dict()


# In[17]:


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# In[18]:


X_test_train = X_test_train.to(device)
y_test_train = y_test_train.to(device)
X_validation = X_validation.to(device)
y_validation = y_validation.to(device)
X_test_val = X_test_val.to(device)
y_test_val = y_test_val.to(device)
y_train = y_train.to(device)
X_train = X_train.to(device)
model = model.to(device)


# In[19]:


epochs = 1000
losses = []
val_losses = []
for i in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train.view(-1, 1))  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    losses.append(loss.item())  # Append training loss

    # Validation loss
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_pred_val = model(X_validation)
        val_loss = loss_fn(y_pred_val, y_validation.view(-1, 1))
        val_losses.append(val_loss.item())  # Append validation loss

    if i % 100 == 0:
        print(f"Epoch {i} -> Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")


# In[20]:


# Plot loss curves
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[21]:


# Test

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_pred_test = model(X_test_val)
    test_loss = loss_fn(y_pred_test, y_test_val.view(-1, 1))
    print(f"Test Loss: {test_loss.item()}")
    plt.plot(y_test_val.cpu(), label='Actual')
    plt.plot(y_pred_test.cpu(), label='Predicted')
    plt.legend()
    plt.show()
    


# In[36]:


# Draw a graph of the actual and predicted values of each feature with name of the feature
feature_names = df.drop(['Life expectancy ', 'Status','Country','Year'], axis=1)
plt.rcParams['figure.figsize'] = [10, 6]

for i in range(0, X_test_val.shape[1]):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_pred_test = model(X_test_val)
        test_loss = loss_fn(y_pred_test, y_test_val.view(-1, 1))
        print(f"Test Loss: {test_loss.item()}")    
    plt.figure()
    
    plt.plot(X_test_val.cpu()[:, i], label='Actual')
    plt.plot(y_pred_test.cpu(), label='Predicted')
    plt.title(feature_names.columns[i])
    plt.legend()
    plt.show()
    

