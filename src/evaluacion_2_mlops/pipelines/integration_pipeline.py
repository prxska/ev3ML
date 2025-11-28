from kedro.pipeline import Pipeline, node
from ..nodes.preprocessing import create_integrated_features
from ..nodes.modeling import train_integrated_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # 1. Preparar datos (usando la salida de K-Means)
        node(
            func=create_integrated_features,
            inputs=["data_clustered_kmeans"], # Viene del pipeline 'ul'
            outputs=["X_train_int", "y_train_int", "X_test_int", "y_test_int"],
            name="create_integrated_features_node"
        ),
        
        # 2. Entrenar modelo final
        node(
            func=train_integrated_model,
            inputs=["X_train_int", "y_train_int", "X_test_int", "y_test_int"],
            outputs="model_integrated_final",
            name="train_integrated_model_node"
        )
    ])



    """
Pipeline para la integración del modelo supervisado con clusters.
"""
from kedro.pipeline import Pipeline, node
from ..nodes.preprocessing import create_integrated_features
from ..nodes.modeling import train_integrated_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # 1. Preparar datos (usando la salida de K-Means)
        node(
            func=create_integrated_features,
            inputs=["data_clustered_kmeans"], # Viene del pipeline 'ul'
            outputs=["X_train_int", "y_train_int", "X_test_int", "y_test_int"],
            name="create_integrated_features_node"
        ),
        
        # 2. Entrenar modelo final (Regresión Logística + Clusters)
        node(
            func=train_integrated_model,
            inputs=["X_train_int", "y_train_int", "X_test_int", "y_test_int"],
            outputs="model_integrated_final",
            name="train_integrated_model_node"
        )
    ])