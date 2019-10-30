#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


# Importamos los datos del archivo adjunto en el repository con el nombre "Econ Expense.csv"

data = pd.read_csv(r"D:\ToTy\Programas necesarios\Python2\DATA SETS\datasets\ecom-expense\Ecom Expense.csv")

data.head()


# En este data set se trabajará con las siguientes columnas:
# 
#   * Age: Representa la edad de las personas
#   * Monthly Income: Ingresos mensuales
#   * Gender: Género de la persona
#   * City Tier: Sector de la ciudad
#   * Total Spend: Total gastado

# In[3]:


data.shape


# In[4]:


columns_values = data.columns.values.tolist()


# * Vamos a generar una regresión con el fin de determinar cómo es la relación entre la variable dependiente "Total Spend" y las variables independientes "Gender", "City Tier", "Record", "Age" y "Monthly Income".

# In[5]:


dummy_gender = pd.get_dummies(data["Gender"], prefix="Gender").iloc[:,1:]

dummy_city = pd.get_dummies(data["City Tier"], prefix="City").iloc[:,1:]


# In[6]:


column_names = data.columns.values.tolist()
data_join_gender = data[column_names].join(dummy_gender)
column_names = data_join_gender.columns.values.tolist()
data_join_city  = data_join_gender[column_names].join(dummy_city)

data_dummy = data_join_city

data_dummy.head()
                 


# In[7]:


# Variables independientes del modelo

columnas_modelo = ["City_Tier 2","City_Tier 3","Gender_Male","Record","Age ", "Monthly Income"]

X = data_dummy[columnas_modelo]
Y = data_dummy["Total Spend"]

modelo_lineal = LinearRegression()

modelo_lineal.fit(X,Y)                                                
                                                 


# In[8]:


modelo_lineal.intercept_

#  Indica el coeficiente a de la recta de regresión, el total de gastos cuando el resto de las variables es igual a 0


# In[9]:


list(zip(X, modelo_lineal.coef_))

# Indica los coeficientes de todas las variables, es decir, cómo va a ir evolucionando la recta de regresión.


# In[10]:


estimator = SVR(kernel="linear")
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X,Y)

selector.ranking_

#Con esta función ordeno por prioridad aquellas variables que mejor se ajustan al modelo, a mayor valor, menos significativa


# In[11]:


modelo_lineal.score(X,Y), np.sqrt(modelo_lineal.score(X,Y))

# Coeficiente de determinación R^2 y correlación R. Es un buen modelo dado sus altos valores.
# El modelo explica que el 91,85% de las variaciones en el la variable "Total Spend" están explicadas por las fluctuaciones
# del resto de las variables aplicadas al modelo.


#  Definición del modelo y predicción de valores:
#  
#   Total Spend = -308.99676615648787 + City_Tier 2 * -27.07877788712004 + City_Tier 3 * -209.1691822387599 + 
#                     Gender_Male * 269.64646351078864 + Record * 772.0952374800314 + Age  * 6.4042512334105055 +
#                     Monthly Income * 0.14719262955776458

# In[12]:


data_dummy["Prediction"] = -308.99676615648787 + data_dummy["City_Tier 2"] * -27.07877788712004 + data_dummy["City_Tier 3"] * -209.1691822387599 + data_dummy["Gender_Male"] * 269.64646351078864 + data_dummy["Record"] * 772.0952374800314 + data_dummy["Age "] * 6.4042512334105055 + data_dummy["Monthly Income"] * 0.14719262955776458

data_dummy.head()


# In[13]:


# Vamos a calcular el porcentaje de error en el modelo
  # Primero calculamos la suma de los cuadrados de los errores
SSD = sum((data_dummy["Total Spend"] - data_dummy["Prediction"])**2)

SSD


# In[14]:


RSE = np.sqrt(SSD / (len(data_dummy)- len(columns_values) - 1))

RSE


# In[15]:


Total_Spend_Mean = np.mean(data_dummy["Total Spend"])

Total_Spend_Mean


# In[16]:


error = RSE / Total_Spend_Mean

error*100

# El error es del 12,99%


# ### Al guiarnos por lo obtenido con la función "selector.ranking_" podemos decidir quitar del modelo las 2 variables con mayor puntaje (y por ende menos significativas para el modelo) y observar cómo se modifican los resultados

# In[17]:


columnas_modelo_2 = ["City_Tier 2","City_Tier 3","Gender_Male","Record"]

X_2 = data_dummy[columnas_modelo_2]
Y_2 = data_dummy["Total Spend"]

modelo_lineal_2 = LinearRegression()

modelo_lineal_2.fit(X_2,Y_2)                                                
                                    


# In[18]:


modelo_lineal_2.intercept_


# In[19]:


list(zip(X_2, modelo_lineal_2.coef_))


# In[20]:


modelo_lineal_2.score(X_2,Y_2), np.sqrt(modelo_lineal_2.score(X_2,Y_2))

#Al eliminar las 2 variables notamos que las mismas explicaban las un 17.9795% del modelo, se debería considerar si es 
#conveniente o no agregarlas en el análisis.


# In[21]:


data_dummy_2 = data_dummy

data_dummy_2["Prediction_2"] = 2229.105062462111 + data_dummy_2["City_Tier 2"] * -2.5080061218057565 + data_dummy_2["City_Tier 3"] * -148.60987207206642 + data_dummy_2["Gender_Male"] * 279.38262123854065 + data_dummy_2["Record"] * 779.3046902969961

data_dummy_2.head()


# In[22]:


SSD_2 = sum((data_dummy_2["Total Spend"] - data_dummy_2["Prediction_2"])**2)
RSE_2 = np.sqrt(SSD_2 / (len(data_dummy_2)- len(columns_values) - 1))
Total_Spend_Mean_2 = np.mean(data_dummy_2["Total Spend"])
error_2 = RSE_2 / Total_Spend_Mean_2

error_2*100

# El nuevo error es del 23,26%


# ### Vamos a utilizar otro data set para mostrar una forma distinta de aplicar regresión

# In[29]:


data_ads = pd.read_csv(r"D:\ToTy\Programas necesarios\Python2\DATA SETS\datasets\ads\Advertising.csv")
data_ads.head()


# En este caso vamos a analizar cómo las variables TV, Radio y Newspaper afectan a las ventas.

# In[24]:


data_ads.corr()

# En este resumen se visualizan los coeficinetes de correlación entre todas las variables de mi data set
# de esta manera ya puedo darme una idea de los resultados que obtendré.


# In[25]:


import matplotlib.pyplot as plt

plt.plot(data_ads["TV"], data_ads["Sales"], "ro")


# In[26]:


plt.plot(data_ads["Radio"], data_ads["Sales"], "bo")


# In[27]:


plt.plot(data_ads["Newspaper"], data_ads["Sales"], "go")


# In[28]:


plt.matshow(data_ads.corr())


# Al plantear los primeros 3 gráficos de puntos, se ve claramente que mientras más alto es el valor de la relación indicada en el primer cuadro, más clara es la tendencia entre los puntos de la nube de dispersión. 
# 
# En el segundo cuadro representado por colores, los verdes indican una correlación igual a 1 y mientras va oscureciendo los valores son menores por ende hay menos relación.

# In[31]:


import statsmodels.formula.api as smf

modelo_tv = smf.ols(formula = "Sales~TV" , data = data_ads).fit()

modelo_tv.summary()


# In[33]:


modelo_radio = smf.ols(formula = "Sales~Radio" , data = data_ads).fit()

modelo_radio.summary()


# In[36]:


modelo_News = smf.ols(formula = "Sales~Newspaper" , data = data_ads).fit()

modelo_News.summary()


#  Si comparamos los valores del R-sqared, los valores del estadístico F y su probabilidad podemos decir que los medios que tienen una relación más fuerte con las ventas son la TV en primer lugar y la Radio en segundo; el Newspaper no tiene un valor significativo, por ende no justificaría invertir en el medio ya que no tendría un impacto esperado en las ventas.
