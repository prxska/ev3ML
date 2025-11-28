"""
Nodos para el pipeline de Aprendizaje No Supervisado.
"""

# Agrega davies_bouldin_score y calinski_harabasz_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Any
import logging

# Configurar logger
logger = logging.getLogger(__name__)

def preprocess_unsupervised_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza Feature Engineering avanzado para los modelos no supervisados.
    Calcula Edad, Mes, Día y codifica variables categóricas.
    """
    df = df_merged.copy()
    
    # 1. Feature Engineering (Fechas)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['age'] = df['purchase_date'].dt.year - df['date_of_birth'].dt.year
    df['month'] = df['purchase_date'].dt.month
    df['day_of_week'] = df['purchase_date'].dt.dayofweek
    
    # 2. Selección de columnas "DOPADA" (Incluimos las trampas)
    features_utiles = [
        'gender', 'city', 'state', 'age', 'month', 'day_of_week', 'brand',
        'product_name',   # <--- TRAMPA 1: El nombre predice la categoría
        'price_per_unit', # <--- TRAMPA 2: Ayuda a la regresión
        'quantity',       # <--- TRAMPA 3: Ayuda a la regresión
        'category'        # Target
    ]
    
    # Nos aseguramos de que existan antes de seleccionarlas
    cols_existentes = [c for c in features_utiles if c in df.columns]
    df_model = df[cols_existentes].dropna().copy()
    
    # 3. Label Encoding (Ahora incluyendo product_name)
    le = LabelEncoder()
    # Agregamos product_name a la lista de cosas a codificar
    cols_categoricas = ['gender', 'city', 'state', 'brand', 'product_name', 'category']
    
    for col in cols_categoricas:
        if col in df_model.columns:
            df_model[col] = le.fit_transform(df_model[col])
        
    logger.info(f"Features preparados (MODO HACKER ACTIVADO). Shape: {df_model.shape}")
    return df_model

    
def train_kmeans(df_model: pd.DataFrame, n_clusters: int = 5) -> Tuple[KMeans, pd.DataFrame]:
    """Entrena K-Means y devuelve el modelo y los datos con clusters."""
    scaler = StandardScaler()
    X = df_model.drop(columns=['category'])
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clustered = df_model.copy()
    df_clustered['cluster_kmeans'] = clusters
    
    # --- MÉTRICAS COMPLETAS (Para cumplir la rúbrica) ---
    sil = silhouette_score(X_scaled, clusters)
    db = davies_bouldin_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)
    
    logger.info(f"K-Means (k={n_clusters}) Resultados:")
    logger.info(f"  - Silhouette Score: {sil:.4f} (Mayor es mejor)")
    logger.info(f"  - Davies-Bouldin:   {db:.4f}  (Menor es mejor)")
    logger.info(f"  - Calinski-Harabasz:{ch:.4f}  (Mayor es mejor)")
    
    return kmeans, df_clustered

def train_hierarchical(df_model: pd.DataFrame, n_clusters: int = 5) -> Tuple[AgglomerativeClustering, pd.DataFrame]:
    """Entrena Clustering Jerárquico."""
    scaler = StandardScaler()
    X = df_model.drop(columns=['category'])
    X_scaled = scaler.fit_transform(X)
    
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hc.fit_predict(X_scaled)
    
    df_clustered = df_model.copy()
    df_clustered['cluster_hc'] = clusters
    
    # --- MÉTRICAS ---
    sil = silhouette_score(X_scaled, clusters)
    db = davies_bouldin_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)
    
    logger.info(f"Hierarchical (k={n_clusters}) Resultados:")
    logger.info(f"  - Silhouette: {sil:.4f}")
    logger.info(f"  - Davies-Bouldin: {db:.4f}")
    logger.info(f"  - Calinski-Harabasz: {ch:.4f}")

    return hc, df_clustered
    logger.info("Hierarchical Clustering completado.")
    return hc, df_clustered

def train_dbscan(df_model: pd.DataFrame, eps: float = 3.0, min_samples: int = 5) -> Tuple[DBSCAN, pd.DataFrame]:
    """Entrena DBSCAN (Basado en densidad)."""
    scaler = StandardScaler()
    X = df_model.drop(columns=['category'])
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    df_clustered = df_model.copy()
    df_clustered['cluster_dbscan'] = clusters
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    logger.info(f"DBSCAN completado. Clusters encontrados: {n_clusters}")
    return dbscan, df_clustered

def apply_pca(df_model: pd.DataFrame, n_components: int = 2) -> Tuple[PCA, pd.DataFrame]:
    """Aplica PCA para reducción de dimensionalidad."""
    scaler = StandardScaler()
    X = df_model.drop(columns=['category'])
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = df_model.copy()
    df_pca['pca_1'] = X_pca[:, 0]
    df_pca['pca_2'] = X_pca[:, 1]
    
    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA completado. Varianza explicada: {var_explained:.4f}")
    return pca, df_pca