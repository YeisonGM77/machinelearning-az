# Importar librerías necesarias
import pandas as pd

# Cargar el dataset
ruta_archivo = "Transmilenio.csv"  # Cambia esto con la ruta real del archivo
df = pd.read_csv(ruta_archivo)

# Renombrar columnas para facilitar el trabajo
df.columns = ['Day_Group_Type', 'Fecha_transaccion', 'Hora_transaccion', 'Entrada', 'Linea',
              'Nombre_Perfil', 'Tipo_Tarifa', 'Tipo_Tarjeta', 'Dispositivo', 'Valor']

# Inspeccionar las primeras filas del dataset
print("Primeras filas del dataset:")
print(df.head())

# Verificar el tipo de dato actual de cada columna
print("\nTipos de datos antes de preprocesar:")
print(df.dtypes)

# Limpieza y conversión de 'Fecha_transaccion' a tipo datetime
df['Fecha_transaccion'] = pd.to_datetime(df['Fecha_transaccion'], format='%d/%m/%Y', errors='coerce')

# Validar valores nulos después de la conversión
print("\nValores nulos por columna después de convertir 'Fecha_transaccion':")
print(df.isnull().sum())

# Remover filas con fechas inválidas si es necesario
df = df.dropna(subset=['Fecha_transaccion'])

# Generar variables adicionales
df['Día_semana'] = df['Fecha_transaccion'].dt.dayofweek  # 0=Lunes, 6=Domingo
df['Es_festivo'] = df['Day_Group_Type'].apply(lambda x: 1 if 'festivo' in str(x).lower() else 0)
df['Día_mes'] = df['Fecha_transaccion'].dt.day  # Día del mes

# Inspeccionar las columnas generadas
print("\nDatos después del preprocesamiento:")
print(df[['Fecha_transaccion', 'Día_semana', 'Es_festivo', 'Día_mes']].head())

# Guardar el dataset preprocesado (opcional)
df.to_csv("dataset_preprocesado.csv", index=False)

# Seleccionar caracteristicas
#import pandas as pd
#import numpy as np

# Agrupación por estación y hora para obtener métricas
df_agrupado = df.groupby(['Entrada', 'Hora_transaccion']).agg({
    'Dispositivo': 'mean',  # Ingresos promedio por hora
    'Valor': 'mean',        # Promedio de recaudo por hora
    'Nombre_Perfil': 'count'  # Cantidad total de transacciones por perfil
}).reset_index()

# Agrupación adicional para obtener métricas diarias por estación
df_estaciones = df.groupby('Entrada').agg({
    'Dispositivo': 'mean',  # Ingresos promedio por estación
    'Valor': 'mean',        # Promedio de recaudo por estación
    'Nombre_Perfil': 'count',  # Cantidad total de transacciones por estación
}).reset_index()

# Normalizar las columnas numéricas (opcional, pero útil para clustering)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_estaciones_scaled = pd.DataFrame(
    scaler.fit_transform(df_estaciones[['Dispositivo', 'Valor', 'Nombre_Perfil']]),
    columns=['Dispositivo', 'Valor', 'Nombre_Perfil']
)
df_estaciones_scaled['Entrada'] = df_estaciones['Entrada']  # Conservar nombres de estaciones

#Código para Clustering con K-Means:
    
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_estaciones_scaled[['Dispositivo', 'Valor', 'Nombre_Perfil']])
    inertia.append(kmeans.inertia_)

# Visualizar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.show()

# Elegir el número de clusters basado en la gráfica (por ejemplo, 4)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_estaciones_scaled['Cluster'] = kmeans.fit_predict(df_estaciones_scaled[['Dispositivo', 'Valor', 'Nombre_Perfil']])

# Inspeccionar los resultados
print("\nEstaciones y sus clusters:")
print(df_estaciones_scaled[['Entrada', 'Cluster']])
    