#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# ##  Análisis de precio de vehículos
# 
# En las siguientes líneas intentaremos predecir si una persona comprará o no un vehículo considerando las distintas variables que ofrece el data set utilizando una regresión logística.
# 
# El mismo presenta las siguientes variables:
#    * mpg = millas por galeón
#    * cylinders = cantidad de cilíndros
#    * displacement = recorrido de los pistones dentro de los cilindros
#    * horsepower = caballos de fuerza
#    * weight = peso
#    * acceleration = aceleración
#    * model year = año del modelo
#    * Price = precio en dólares

# In[2]:


data  = pd.read_csv("auto-mpg.csv")


# In[3]:


data.head()


# In[4]:


data["mpg"].max(), data["mpg"].min()


# In[5]:


data["cylinders"].max(), data["cylinders"].min()


# In[6]:


data["displacement"].max(), data["displacement"].min()


# In[7]:


data["horsepower"].max(), data["horsepower"].min()


# In[8]:


data["acceleration"].max(), data["acceleration"].min()


# In[9]:


data["model year"].max(), data["model year"].min()


# In[10]:


data.shape


# Vamos a quitar las filas que tengan algún valor NaN en ella ya que no son muchas y por ende considero que no afectará al resultado del experimento.

# In[11]:


data.isnull().sum()

data = data.dropna(axis=0)

data.shape


# -- Crearemos un nuevo data set en donde se presentarán los valores estandarizados de cada una de las variables

# In[12]:



data2 = pd.DataFrame()


data2["MPG-Z"] = (data["mpg"]-np.mean(data["mpg"]))/np.std(data["mpg"])
data2["Cylinder-Z"] = (data["cylinders"]-np.mean(data["cylinders"]))/np.std(data["cylinders"])
data2["Displacement-Z"] = (data["displacement"]-np.mean(data["displacement"]))/np.std(data["displacement"])
data2["Horsepower-Z"] = (data["horsepower"]-np.mean(data["horsepower"]))/np.std(data["horsepower"])
data2["Acceleration-Z"] = (data["acceleration"]-np.mean(data["acceleration"]))/np.std(data["acceleration"])
data2["Model_Year-Z"] = (data["model year"]-np.mean(data["model year"]))/np.std(data["model year"])


data2.head()


# Generaremos otro data set utilizando el paquete stats de scipy para presentar las probabilidades asociadas a cada valor de Z obtenido en el data set anterior

# In[13]:


data3 = pd.DataFrame()

data3["MPG-Prob"] = stats.norm.cdf(data2["MPG-Z"])
data3["Cylinder-Prob"] = stats.norm.cdf(data2["Cylinder-Z"])
data3["Displacement-Prob"] = stats.norm.cdf(data2["Displacement-Z"])
data3["Horsepower-Prob"] = stats.norm.cdf(data2["Horsepower-Z"])
data3["Acceleration-Prob"] = stats.norm.cdf(data2["Acceleration-Z"])
data3["Moder_Year-Prob"] = stats.norm.cdf(data2["Model_Year-Z"])

data3.head()


# Generamos la columna del precio indicando un valor par la media y el desvío ya que el data set base no los aporta como dato

# In[14]:


# Proponemos que el precio promedio es de U$D 7500 con un desvío de U$D 2500
Mu = 7500
sd = 2500


data2["Price"] = (data2["MPG-Z"]*sd+Mu)*0.3 + (data2["Cylinder-Z"]*sd+Mu)*0.3 + (data2["Displacement-Z"]*sd+Mu)*0.1 + (data2["Horsepower-Z"]*sd+Mu)*0.1 + (data2["Acceleration-Z"]*sd+Mu)*0.1 + (data2["Model_Year-Z"]*sd+Mu)*0.1

data3.head()


# In[16]:


data2["Price"].max(), data2["Price"].min()


# In[18]:


# Agregamos el dato del precio al data set principal

data = data.join(data2["Price"])


# In[19]:


data.head()


# In[ ]:




