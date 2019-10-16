#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# In[4]:


# Importamos los datos del archivo adjunto en el repository con el nombre "Econ Expense.csv"

data = pd.read_csv(r"D:\ToTy\Programas necesarios\Python2\DATA SETS\datasets\ecom-expense\Ecom Expense.csv")

data.head()


# In[42]:


data.shape


# In[60]:


columns_values = data.columns.values.tolist()


# * Vamos a generar una regresión con el fin de determinar cómo es la relación entre la variable dependiente "Total Spend" y las variables independientes "Gender", "City Tier", "Record", "Age" y "Monthly Income".

# In[15]:


dummy_gender = pd.get_dummies(data["Gender"], prefix="Gender").iloc[:,1:]

dummy_city = pd.get_dummies(data["City Tier"], prefix="City").iloc[:,1:]


# In[16]:


column_names = data.columns.values.tolist()
data_join_gender = data[column_names].join(dummy_gender)
column_names = data_join_gender.columns.values.tolist()
data_join_city  = data_join_gender[column_names].join(dummy_city)

data_dummy = data_join_city

data_dummy.head()
                 


# In[24]:


# Variables independientes del modelo

columnas_modelo = ["City_Tier 2","City_Tier 3","Gender_Male","Record","Age ", "Monthly Income"]

X = data_dummy[columnas_modelo]
Y = data_dummy["Total Spend"]

modelo_lineal = LinearRegression()

modelo_lineal.fit(X,Y)                                                
                                                 


# In[35]:


modelo_lineal.intercept_

#  Indica el coeficiente a de la recta de regresión


# In[44]:


list(zip(X, modelo_lineal.coef_))

# Indica los coeficientes de todas las variables


# In[46]:


estimator = SVR(kernel="linear")
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X,Y)

selector.ranking_

#Con esta función ordeno por prioridad aquellas variables que mejor se ajustan al modelo


# In[52]:


modelo_lineal.score(X,Y), np.sqrt(modelo_lineal.score(X,Y))

# Coeficiente de determinación R^2 y correlación R. Es un buen modelo dado sus altos valores.


#  Definición del modelo y predicción de valores:
#  
#   Total Spend = -308.99676615648787 + City_Tier 2 * -27.07877788712004 + City_Tier 3 * -209.1691822387599 + 
#                     Gender_Male * 269.64646351078864 + Record * 772.0952374800314 + Age  * 6.4042512334105055 +
#                     Monthly Income * 0.14719262955776458

# In[55]:


data_dummy["Prediction"] = -308.99676615648787 + data_dummy["City_Tier 2"] * -27.07877788712004 + data_dummy["City_Tier 3"] * -209.1691822387599 + data_dummy["Gender_Male"] * 269.64646351078864 + data_dummy["Record"] * 772.0952374800314 + data_dummy["Age "] * 6.4042512334105055 + data_dummy["Monthly Income"] * 0.14719262955776458

data_dummy.head()


# In[87]:


# Vamos a calcular el porcentaje de error en el modelo
  # Primero calculamos la suma de los cuadrados de los errores
SSD = sum((data_dummy["Total Spend"] - data_dummy["Prediction"])**2)

SSD


# In[88]:


RSE = np.sqrt(SSD / (len(data_dummy)- len(columns_values) - 1))

RSE


# In[89]:


Total_Spend_Mean = np.mean(data_dummy["Total Spend"])

Total_Spend_Mean


# In[111]:


error = RSE / Total_Spend_Mean

error*100

# El error es del 23,26%


# ### Al guiarnos por lo obtenido en la línea 46 podemos decidir quitar del modelo las 2 variables con mayor puntaje (y por ende menos significativas para el modelo) y observar cómo se modifican los resultados

# In[73]:


columnas_modelo_2 = ["City_Tier 2","City_Tier 3","Gender_Male","Record"]

X_2 = data_dummy[columnas_modelo_2]
Y_2 = data_dummy["Total Spend"]

modelo_lineal_2 = LinearRegression()

modelo_lineal_2.fit(X_2,Y_2)                                                
                                    


# In[74]:


modelo_lineal_2.intercept_


# In[75]:


list(zip(X_2, modelo_lineal_2.coef_))


# In[78]:


modelo_lineal_2.score(X_2,Y_2), np.sqrt(modelo_lineal_2.score(X_2,Y_2))

#Al eliminar las 2 variables notamos que las mismas explicaban las un 17.9795% del modelo, se debería considerar si es 
#conveniente o no agregarlas en el análisis.


# In[133]:


data_dummy_2 = data_dummy

data_dummy_2["Prediction_2"] = 2229.105062462111 + data_dummy_2["City_Tier 2"] * -2.5080061218057565 + data_dummy_2["City_Tier 3"] * -148.60987207206642 + data_dummy_2["Gender_Male"] * 279.38262123854065 + data_dummy_2["Record"] * 779.3046902969961

data_dummy_2.head()


# In[127]:


SSD_2 = sum((data_dummy_2["Total Spend"] - data_dummy_2["Prediction_2"])**2)
RSE_2 = np.sqrt(SSD_2 / (len(data_dummy_2)- len(columns_values) - 1))
Total_Spend_Mean_2 = np.mean(data_dummy_2["Total Spend"])
error_2 = RSE_2 / Total_Spend_Mean_2

error_2*100

