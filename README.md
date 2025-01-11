# proyecto-de-IA
CODIGOS DEL PROYECTO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

datos = {
    'Fecha': pd.date_range(start='2023-01-01', periods=365),
    'Producto': np.random.choice(['A', 'B', 'C', 'D'], size=365),
    'Precio': np.random.uniform(10, 100, size=365),
    'Promoción': np.random.choice([0, 1], size=365),
    'Región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size=365),
    'Ventas': np.random.poisson(lam=20, size=365)
}
df = pd.DataFrame(datos)

ohe = OneHotEncoder()
caracteristicas_categoricas = ['Producto', 'Región']
ohe_df = pd.DataFrame(ohe.fit_transform(df[caracteristicas_categoricas]).toarray(), 
                       columns=ohe.get_feature_names_out(caracteristicas_categoricas))

df = pd.concat([df, ohe_df], axis=1).drop(columns=caracteristicas_categoricas + ['Fecha'])

X = df.drop(columns=['Ventas'])
y = df['Ventas']

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.3, random_state=42)

regresion_lineal = LinearRegression()
regresion_lineal.fit(X_entrenamiento, y_entrenamiento)
predicciones_rl = regresion_lineal.predict(X_prueba)
rmse_rl = np.sqrt(mean_squared_error(y_prueba, predicciones_rl))
r2_rl = r2_score(y_prueba, predicciones_rl)

arbol_decision = DecisionTreeRegressor(random_state=42)
arbol_decision.fit(X_entrenamiento, y_entrenamiento)
predicciones_ad = arbol_decision.predict(X_prueba)
rmse_ad = np.sqrt(mean_squared_error(y_prueba, predicciones_ad))
r2_ad = r2_score(y_prueba, predicciones_ad)

kmeans = KMeans(n_clusters=4, random_state=42)
etiquetas_kmeans = kmeans.fit_predict(X_escalado)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=etiquetas_kmeans, palette='viridis')
plt.title('Clustering K-Means con PCA')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.legend(title='Cluster')
plt.show()

print("Regresión Lineal:\nRMSE:", rmse_rl, "\nR2:", r2_rl)
print("\nÁrbol de Decisión:\nRMSE:", rmse_ad, "\nR2:", r2_ad)

importancia_caracteristicas = pd.Series(arbol_decision.feature_importances_, index=X.columns)
importancia_caracteristicas.nlargest(10).plot(kind='barh', figsize=(10, 6))
plt.title('Importancia de Características - Árbol de Decisión')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()

