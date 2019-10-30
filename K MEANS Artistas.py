#!/usr/bin/env python
# coding: utf-8

# * El data set presenta un grupo de artistas y ciertas características de personalidad, lo que intentaremos es demostrar si las personas que comparten un mismo arte, comparten también características similares. 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


# In[121]:


data = pd.read_csv(r"D:\ToTy\Programas necesarios\Python2\DATA SETS\datasets\analisis.csv")
data


# Resumen de columnas
# 
# * usuario (el nombre en Twitter)
# * «op» = Openness to experience – grado de apertura mental a nuevas experiencias, curiosidad, arte
# * «co» =Conscientiousness – grado de orden, prolijidad, organización
# * «ex» = Extraversion – grado de timidez, solitario o participación ante el grupo social
# * «ag» = Agreeableness – grado de empatía con los demás, temperamento
# * «ne» = Neuroticism, – grado de neuroticismo, nervioso, irritabilidad, seguridad en sí mismo.
# * Wordcount – Cantidad promedio de palabras usadas en sus tweets
#         * Categoria – Actividad laboral del usuario (actor, cantante, etc.)
#                 Donde: 
#                1- Actor/actriz
#                2- Cantante
#                3- Modelo
#                4- Tv, series
#                5- Radio
#                6- Tecnología
#                7- Deportes
#                8- Politica
#                9- Escritor

# In[7]:


print(data.groupby('categoria').size())

# De esta forma podemos visualizar cuántas personas de cada actividad se encuentran en nuestro data set


# In[10]:


sb.pairplot(data.dropna(), hue='categoria',height=4,vars=["op","ex","ag"],kind='scatter')

#Vamos a tomar 3 variables para medir en este caso, que son op, ex y ag.
# En una primera instancia no parecería que hay relación entre las variables y las categorías


# In[12]:


# Definimos las variables para generar el algorítmo

X = np.array(data[["op","ex","ag"]])
Y = np.array(data['categoria'])
X.shape


# In[20]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.grid()
plt.show()

#Tomaremos 5 como el valor de K según el gráfico obtenido


# In[21]:


kmeans = KMeans(n_clusters=5).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)


# In[25]:


labels = kmeans.predict(X)


# In[42]:


C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])


# #### Con la predicción de datos y los grupos armados puedo graficar las distintas combinaciones y analizar si existe alguna relación o conjunto.

# In[37]:


f1 = data['op'].values
f2 = data['ex'].values
plt.figure(figsize=(10,6))
plt.scatter(f1, f2, c=asignar, s=30)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()


# In[40]:


f1 = data['op'].values
f2 = data['ag'].values
plt.figure(figsize=(10,6))
plt.scatter(f1, f2, c=asignar, s=30)
plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
plt.show()


# In[41]:


f1 = data['ex'].values
f2 = data['ag'].values
plt.figure(figsize=(10,6))
plt.scatter(f1, f2, c=asignar, s=30)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()


# In[48]:


copy =  pd.DataFrame()
copy['usuario']=data['usuario'].values
copy['categoria']=data['categoria'].values
copy['label'] = labels;
cantidadGrupo =  pd.DataFrame()
cantidadGrupo['color']=colores
cantidadGrupo['cantidad']=copy.groupby('label').size()
cantidadGrupo

#Aquí observamos la cantidad de datos que contiene cada cluster.


# In[119]:


# Y ahora veremos cómo se compone cada cluster según categorías
group_referrer_index = copy['label'] ==0
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad']=group_referrals.groupby('categoria').size()
diversidadGrupo


# In[113]:


group_referrer_index = copy['label'] ==1
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad']=group_referrals.groupby('categoria').size()
diversidadGrupo


# In[107]:


group_referrer_index = copy['label'] ==2
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad']=group_referrals.groupby('categoria').size()
diversidadGrupo

# Concentra a personas de categoría 4


# In[105]:


group_referrer_index = copy['label'] ==3
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad']=group_referrals.groupby('categoria').size()
diversidadGrupo


# In[112]:


group_referrer_index = copy['label'] ==4
group_referrals = copy[group_referrer_index]

diversidadGrupo =  pd.DataFrame()
diversidadGrupo['categoria']=[0,1,2,3,4,5,6,7,8,9]
diversidadGrupo['cantidad']=group_referrals.groupby('categoria').size()
diversidadGrupo

#Concentra a más personas de las categorías 1 y 2


# In[124]:


# Ahora si encontramos los valores para nuestras 3 variables de una persona, podríamos agruparla según los clusters definidos
# durante el análisis

X_new = np.array([[50.92,57.74,15.66]])
 
new_labels = kmeans.predict(X_new)
print(new_labels)


# In[ ]:




