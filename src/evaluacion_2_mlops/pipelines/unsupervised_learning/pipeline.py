from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_unsupervised_features,
    train_kmeans,
    train_hierarchical,
    train_dbscan,
    apply_pca
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # 1. Preprocesamiento específico para No Supervisado
        node(
            func=preprocess_unsupervised_features,
            # Usamos la 'primary_data' que viene de tu pipeline de data engineering original
            inputs="primary_data", 
            outputs="model_input_unsupervised",
            name="preprocess_unsupervised_node"
        ),
        
        # 2. Clustering: K-Means
        node(
            func=train_kmeans,
            inputs="model_input_unsupervised",
            outputs=["model_kmeans", "data_clustered_kmeans"],
            name="train_kmeans_node"
        ),
        
        # 3. Clustering: Hierarchical
        node(
            func=train_hierarchical,
            inputs="model_input_unsupervised",
            outputs=["model_hierarchical", "data_clustered_hc"],
            name="train_hierarchical_node"
        ),
        
        # 4. Clustering: DBSCAN
        node(
            func=train_dbscan,
            inputs="model_input_unsupervised",
            outputs=["model_dbscan", "data_clustered_dbscan"],
            name="train_dbscan_node"
        ),
        
        # 5. Reducción de Dimensionalidad: PCA
        node(
            func=apply_pca,
            inputs="model_input_unsupervised",
            outputs=["model_pca", "data_pca"],
            name="apply_pca_node"
        ),
    ])