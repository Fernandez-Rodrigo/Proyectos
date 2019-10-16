#!/usr/bin/env python
# coding: utf-8

# ## MODELO DE REGRESIÓN CON DATOS SIMULADOS

# In[24]:


import pandas as pd
import numpy as np
import os 


# En una primera instancia vamos a generar un modelo con datos al azar para trabajar
# * x: 100 valores distribuidos según una N(1.5 , 2.5)
# * Ye = 5 + 1.9 * X + e
# * e: estará distribuido según una disitribución normal N(0 , 0.8)

# In[87]:


x = 1.5 + 2.5 * np.random.randn(100)


# In[88]:


residuos = 0 + 0.8 * np.random.randn(100)


# In[89]:


Y_pred = 5 + 1.9 * x


# In[90]:


Y_observado = 5 + 1.9 * x + residuos


# In[91]:


x_list = x.tolist()
Y_pred_list = Y_pred.tolist()
Y_observado_list = Y_observado.tolist()

#Los transformo en listas para poder crear el data frame


# In[99]:


# Armamos el data frame de forma manual
data = pd.DataFrame(
    {
        "x" : x_list,
        "y" : Y_observado_list,
        "y_pred" : Y_pred_list
        
    }
)


# In[100]:


data.head()
# Tengo los valores de x, de y y el valor que arroja la fórmula de regresión


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


y_mean = [np.mean(Y_observado) for i in range(1, len(x_list)+1)]

# Si no agrego la parte de "for i in range..." se calcula solo 1 valor de media y no se puede graficar, por que es solo 1 valor
# de esta forma lo hago para cada valor de x que exista el valor de y_mean así se genera la recta


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot( x, Y_pred, "b")
plt.plot( x, Y_observado, "ro")
plt.plot(x,y_mean , "orange")
plt.title("Valor Actual contra Predicción")

#Dibujo la predicción y la muestra de datos


# In[92]:


#Ahora vamos a agregar al data frame la suma de los cuadrados para calcular los coeficientes de correlación y determinación
# Seguramente haya una librería que los arroje solos
np.corrcoef(x_list,Y_observado_list)
                            


# In[93]:


data["SSR"] = (data["y_pred"]-np.mean(Y_observado))**2
data["SSD"] = (data["y_pred"]- data["y"])**2
data["SST"] = (data["y"] - np.mean(Y_observado))**2


# In[94]:


data.head()


# In[95]:


SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])


# In[96]:


R2 = SSR/SST


# In[97]:


R2


# In[98]:


np.sqrt(R2)
# np.corrcoef = COEFICIENTE DE CORRELACIÓN EFECTIVAMENTE

#Ahora da diferente por que algo se rompió en el medio cuando lo ejecuté todo de nuevo


# In[ ]:




