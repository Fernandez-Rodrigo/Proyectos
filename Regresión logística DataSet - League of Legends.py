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


# In[11]:


data_variables = data_dummy.columns.values.tolist()

variables = [v for v in data_variables if v not in categorias]

variables


# In[12]:


data_match_dummy = data_dummy[variables]

data_match_dummy.columns.values.tolist()


# In[13]:


# Borramos 4 columnas que no utlizaremos en el análisis y la columna donde indican que gana el otro equipo para evitar futuros
# errores al implementar el modelo

data_match_dummy = data_match_dummy.drop(["Year", "Season", "Type", "rResult"], axis = 1)


# In[14]:


data_match_dummy.shape


# In[15]:


data_match_dummy_test = data_match_dummy


# In[16]:


# Como paso siguiente vamos a eliminar las columnas que tengan pocos datos para evitar problemas al momento de generar el modelo
# Aquellos que no sumen 15 repeticiones en la columna serán eliminados

data_match_dummy_test = data_match_dummy_test.loc[:, (data_match_dummy_test.sum())>15]

data_match_dummy_test.sum().sort_values()

data_match_dummy_test.shape


# In[17]:


# Vamos a predecir la variable bResult, la misma indica 1 para victoria de este equipo y 0 para victoria del equipo contrario

data_match_V = data_match_dummy_test.columns.values.tolist()

Y = data_match_dummy_test["bResult"]

Y_var = ["bResult"]

var_X = [v for v in data_match_V if v not in Y_var]

X = data_match_dummy_test[var_X]

X.head()


# In[18]:


X.shape


# In[19]:


Y.head()


# ### Primero utilizaremos statsmodels como herramienta para generar el modelo

# In[20]:


import statsmodels.api as sm


# In[21]:


modelo_logistico = sm.Logit(Y, X)


# In[22]:


resultado = modelo_logistico.fit(method="bfgs", max_iter=5000)


# In[23]:


resultado.summary2()


# ### Aplicaremos ahora el análisis con scikit learn

# In[24]:


from sklearn import linear_model


# In[25]:


modelo_logistico_sk = linear_model.LogisticRegression(solver= "lbfgs", max_iter=5000)
modelo_logistico_sk.fit(X, Y)


# In[26]:


coeficientes_1 = modelo_logistico_sk.coef_
coeficientes_1


# In[27]:


import sys

coeficientes_1.sort()

coeficientes_1


# In[28]:


modelo_logistico_sk.score(X,Y)


# In[36]:


modelo_logistico_sk.intercept_


# In[29]:


pd.DataFrame(list(zip(X.columns, np.transpose(coeficientes_1))))


# ### VALIDACIÓN DEL MODELO

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


Y.mean()


# In[32]:


# Generamos datos para test y datos para train, es decir, datos para entrenar el modelo y para probarlo. Decido separarlo en 30%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[33]:


# Creo el modelo para el grupo de entrenamiento

modelo2 = linear_model.LogisticRegression(solver = "lbfgs")
modelo2.fit(X_train, Y_train)


# In[39]:


probs = modelo2.predict_proba(X_test)

probs

#Columna 1 que tan seguro estoy de lo que digo, columna 2 si gana el equipo azul o rojo si es mayor o no a 0,5


# In[41]:


prediccion = modelo2.predict(X_test)

prediccion

#Si en el anterior paso la prob es de 0,5 aca da 0 si es mayor acá da 1


# In[47]:


prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.54
prob_df["prediction"] = np.where(prob_df[0]>threshold,1,0)



pd.crosstab(prob_df.prediction, columns = "count")


# In[48]:


from sklearn import metrics


# In[51]:


metrics.accuracy_score(Y_test, prediccion)

# Mide el grado de accuracy tanto de los casos donde indicamos que gana el equipo azul y efectivamente gana y cuando este pierde
# y efectivamente pierde


# ### Validación cruzada

# In[34]:


from sklearn.model_selection import cross_val_score


# In[55]:


scores = cross_val_score(linear_model.LogisticRegression(solver= "lbfgs", max_iter = 5000), X, Y, scoring="accuracy", cv=8)

scores

scores.mean()


# In[ ]:




