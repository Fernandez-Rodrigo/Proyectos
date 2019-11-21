#!/usr/bin/env python
# coding: utf-8

# ### En las siguientes líneas investigaremos si existe relación entre las victorias de los equipos y otras variables que serán descriptas debajo como por ejemplo el campeón elegido o el jugador que participa. 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_match = pd.read_csv("matchinfo.csv")


# In[3]:


data_match.head(50)


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
    
data_dummy.head()


# In[8]:


data_dummy.shape


# In[9]:


data_variables = data_dummy.columns.values.tolist()

variables = [v for v in data_variables if v not in categorias]

variables


# In[10]:


data_match_dummy = data_dummy[variables]

data_match_dummy.columns.values.tolist()


# In[11]:


#Borramos 3 columnas que no utlizaremos en el análisis y la columna donde indican que gana el otro equipo para evitar futuros
#errores al implementar el modelo

del data_match_dummy["Year"]


# In[12]:


del data_match_dummy["Season"]


# In[13]:


del data_match_dummy["Type"]


# In[14]:


del data_match_dummy["rResult"]


# In[30]:


bResult_N = data_match_dummy["bResult"]+0.000001*np.random.rand()

bResult_N.head()

data_match_dummy = data_match_dummy.join(bResult_N)

Base_data_match = data_match_dummy

data_dummy_test = data_match_dummy

data_dummy_test.head()


# In[25]:


# Vamos a predecir la variable bResult, la misma indica 1 para victoria de este equipo y 0 para victoria del equipo contrario

data_match_V = data_dummy_test.columns.values.tolist()

Y = data_dummy_test["bResultN"]

var_X = [v for v in data_match_V if v not in Y]

X = data_dummy_test[var_X]

X.head()


# In[26]:


import statsmodels.api as sm


# In[27]:


modelo_logistico = sm.Logit(Y, X)


# In[28]:


resultado = modelo_logistico.fit()


# In[ ]:


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




