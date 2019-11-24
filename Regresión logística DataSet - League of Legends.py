#!/usr/bin/env python
# coding: utf-8

# ### En las siguientes líneas investigaremos si existe relación entre las victorias de los equipos y otras variables que serán descriptas debajo como por ejemplo el campeón elegido o el jugador que participa. 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_match = pd.read_csv(r"D:\ToTy\Programas necesarios\Python2\DATA SETS\datasets\LOL\matchinfo.csv")


# In[3]:


data_match.head()


# Quitaremos la columna "Address" ya que solo indica el link de donde se obtuvieron los datos, no aporta nada al análisis deseado

# In[4]:


del data_match["Address"]


# In[5]:


data_match.shape


# In[6]:


data_match.columns.values.tolist()


# In[7]:


# Definiremos las variables cualitativas para generar así las respectivas columnas dummys a utilizar en el análisis
pd.options.display.max_columns = 10000

categorias = ["League", "blueTeamTag", "redTeamTag", "blueTop", "blueTopChamp", "blueJungle", 
              "blueJungleChamp", "blueMiddle", "blueMiddleChamp", "blueADC", "blueADCChamp", "blueSupport", 
              "blueSupportChamp", "redTop", "redTopChamp", "redJungle", "redJungleChamp", "redMiddle", 
              "redMiddleChamp", "redADC", "redADCChamp", "redSupport", "redSupportChamp"]
for category in categorias:
    cat_list = data_match[categorias]+"_"+category
    cat_dummy = pd.get_dummies(data_match[categorias])
    data_dummy = data_match.join(cat_dummy)
    


# In[8]:


data_dummy.head()


# In[9]:


data_dummy.shape


# In[10]:


pd.value_counts(data_dummy["redSupport"])


# In[11]:


data_variables = data_dummy.columns.values.tolist()

variables = [v for v in data_variables if v not in categorias]

variables


# In[12]:


data_match_dummy = data_dummy[variables]

data_match_dummy.columns.values.tolist()


# In[13]:


#Borramos 4 columnas que no utlizaremos en el análisis y la columna donde indican que gana el otro equipo para evitar futuros
#errores al implementar el modelo

data_match_dummy = data_match_dummy.drop(["Year", "Season", "Type", "rResult"], axis = 1)


# In[56]:


data_match_dummy.shape


# In[58]:


data_match_dummy_test = data_match_dummy


# In[69]:


# Como paso siguiente vamos a eliminar las columnas que tengan pocos datos para evitar problemas al momento de generar el modelo
# Aquellos que no sumen 15 repeticiones en la columna serán eliminados

data_match_dummy_test = data_match_dummy_test.loc[:, (data_match_dummy_test.sum())>15]

data_match_dummy_test.sum().sort_values()

data_match_dummy_test.shape


# In[70]:


# Vamos a predecir la variable bResult, la misma indica 1 para victoria de este equipo y 0 para victoria del equipo contrario

data_match_V = data_match_dummy_test.columns.values.tolist()

Y = data_match_dummy_test["bResult"]

Y_var = ["bResult"]

var_X = [v for v in data_match_V if v not in Y_var]

X = data_match_dummy_test[var_X]

X.head()


# In[72]:


X.shape


# In[71]:


Y.head()


# In[73]:


import statsmodels.api as sm


# In[74]:


modelo_logistico = sm.Logit(Y, X)


# In[94]:


resultado = modelo_logistico.fit(method="bfgs", max_iter=5000)


# In[95]:


resultado.summary2()


# ### VALIDACIÓN DEL MODELO

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Generamos datos para test y datos para train, es decir, datos para entrenar el modelo y para probarlo. Decido separarlo en 30%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[ ]:


# Creo el modelo para el grupo de entrenamiento

modelo2 = linear_model.LogisticRegression(solver = "lbfgs")
modelo2.fit(X_train, Y_train)


# In[ ]:




