# Importação das bibliotecas necessárias
import pandas as pd                  # Manipulação de dados em tabelas (DataFrame)
import numpy as np                   # Cálculos numéricos e vetoriais
import matplotlib.pyplot as plt      # Visualização de gráficos
from sklearn.cluster import KMeans   # Algoritmo de clusterização K-Means
from sklearn.preprocessing import StandardScaler  # Padronização dos dados
from sklearn.decomposition import PCA             # Redução de dimensionalidade (PCA)
from sklearn.metrics import silhouette_score      # Métrica de avaliação de clusters

"""### Dados"""

# Carrega o dataset a partir de um arquivo CSV
df = pd.read_csv("segmentacao.csv")
df.head()  # Exibe as 5 primeiras linhas do dataset

"""### Transformar Region em Numérico"""

# Converte a variável categórica "region" em variáveis dummy (numéricas)
# drop_first=True remove uma das categorias para evitar multicolinearidade
df = pd.get_dummies(df, columns=['region'], drop_first=True)
df.head()  # Exibe novamente as 5 primeiras linhas após a transformação

"""### Análise Exploratória"""

# Cria uma figura para exibir 3 gráficos lado a lado
plt.figure(figsize=(15, 5))

# Primeiro gráfico - Distribuição das idades
plt.subplot(1, 3, 1)  # Define posição do gráfico (1ª linha, 3 colunas, 1º gráfico)
plt.hist(df['age'], bins=15, color='skyblue', edgecolor='black')  # Histograma da idade
plt.title("Distribuição de Idades")  # Título
plt.xlabel("Idade")                  # Rótulo eixo X
plt.ylabel("Frequência")             # Rótulo eixo Y

# Segundo gráfico - Distribuição da renda
plt.subplot(1, 3, 2)
plt.hist(df['income'], bins=15, color='orange', edgecolor='black')
plt.title("Distribuição de Renda")
plt.xlabel("Renda Anual")
plt.ylabel("Frequência")

# Terceiro gráfico - Distribuição do índice de gastos
plt.subplot(1, 3, 3)
plt.hist(df['spending_score'], bins=15, color='green', edgecolor='black')
plt.title("Distribuição do Índice de Gastos")
plt.xlabel("Spending Score")
plt.ylabel("Frequência")

# Ajusta os gráficos para não sobreporem legendas
plt.tight_layout()
plt.show()

"""### Exclui ID"""

# Remove a coluna 'customer_id' (não é relevante para o modelo)
X = df.drop(columns=['customer_id'])
X.head()  # Mostra as primeiras linhas sem o ID

"""### Padronização"""

# Cria objeto para padronizar dados (média=0, desvio padrão=1)
scaler = StandardScaler()
# Aplica a padronização nos dados
X_scaled = scaler.fit_transform(X)
X_scaled  # Exibe a matriz padronizada

"""### PCA"""

# Cria objeto PCA para reduzir dimensões mantendo 95% da variância dos dados
pca = PCA(n_components=0.95)
# Aplica a transformação PCA nos dados escalonados
X_pca = pca.fit_transform(X_scaled)
# Exibe o número de componentes principais escolhidos automaticamente
print(f"Número de componentes principais: {X_pca.shape[1]}")
X_pca  # Exibe os dados transformados pelo PCA

"""### Número Ideal de Clusters"""

# Listas para armazenar resultados
inertia = []             # Inércia (soma das distâncias dentro dos clusters)
silhouette_scores = []   # Métrica silhouette (avalia qualidade dos clusters)
cluster_range = range(2, 10)  # Testar valores de 2 a 9 clusters

# Loop para testar diferentes valores de k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Cria modelo K-Means
    kmeans.fit(X_pca)  # Ajusta modelo aos dados
    inertia.append(kmeans.inertia_)  # Salva inércia
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))  # Salva silhouette

# Exibe os resultados de silhouette para cada k
for k, sil_score in zip(cluster_range, silhouette_scores):
    print(f"Clusters: {k}, Silhouette Score: {sil_score:.4f}")

# Gráfico do Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title("Silhouette Score")
plt.xlabel("Número de Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Gráfico do Método do Cotovelo (Elbow Method)
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.show()

"""### Gerar Clusters e Atribuir ao Cliente"""

# Define o número de clusters escolhido (k=4 neste caso)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Cria modelo com k=4
# Aplica o modelo PCA transformado e adiciona o cluster no DataFrame original
df['cluster'] = kmeans.fit_predict(X_pca)
df.head(10)  # Mostra os 10 primeiros clientes com o cluster atribuído

"""### Visualização"""

# Gráfico de dispersão mostrando os clientes nos 2 primeiros componentes principais
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis',
            alpha=0.7, edgecolors='black')  # Cada cor representa um cluster
plt.title("Segmentação de Clientes")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")  # Barra de cores indicando clusters
plt.show()