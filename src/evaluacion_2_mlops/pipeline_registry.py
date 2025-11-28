"""Registra los pipelines del proyecto."""
from typing import Dict
from kedro.pipeline import Pipeline, node

# Importa los nodos de preprocesamiento individuales
from .nodes.preprocessing import create_primary_table, create_features

# Importa las definiciones de los pipelines
from .pipelines import classification_pipeline, regression_pipeline
from .pipelines.unsupervised_learning import pipeline as unsupervised_pipeline
from .pipelines import integration_pipeline  # <--- NUEVO IMPORT (Integración)

def register_pipelines() -> Dict[str, Pipeline]:
    """Registra los pipelines del proyecto.

    Returns:
        Un diccionario mapeando nombres de pipelines a instancias de Pipeline.
    """
    
    # --- 1. Definición de Nodos Individuales (Data Engineering) ---
    
    # Nodo que une los 3 CSVs
    data_processing_node = node(
        func=create_primary_table,
        inputs=["customer_profile", "products", "purchase_history"],
        outputs="primary_data",
        name="create_primary_table_node"
    )
    
    # Nodo que crea los features supervisados (X_train, y_train, etc.)
    feature_engineering_node = node(
        func=create_features,
        inputs="primary_data",
        outputs=[
            "X_train_class", "X_test_class", "y_train_class", "y_test_class", 
            "X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg"
        ],
        name="create_features_node"
    )

    # --- 2. Creación de Instancias de Pipelines ---
    
    # Pipelines Supervisados (Ev2)
    class_pipe = classification_pipeline.create_pipeline()
    reg_pipe = regression_pipeline.create_pipeline()
    
    # Pipeline No Supervisado (Ev3 - Clustering)
    unsup_pipe = unsupervised_pipeline.create_pipeline()
    
    # Pipeline de Integración (Ev3 - Modelo Final)
    int_pipe = integration_pipeline.create_pipeline()

    # --- 3. Ensamblaje de Pipelines Completos ---

    # Pipeline de datos (dp): Solo une los CSVs
    dp_pipeline = Pipeline([data_processing_node])
    
    # Pipeline No Supervisado (ul): Clustering y PCA
    ul_pipeline = Pipeline([
        data_processing_node,
        unsup_pipe
    ])
    
    # Pipeline de Integración (int): Clustering + Modelo Mejorado
    # Este pipeline necesita que primero corra el procesamiento y el clustering
    int_pipeline = Pipeline([
        data_processing_node,
        unsup_pipe,
        int_pipe
    ])
    
    # Pipeline de clasificación original (cl)
    cl_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        class_pipe
    ])
    
    # Pipeline de regresión original (rg)
    rg_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        reg_pipe
    ])
    
    # Pipeline por defecto (__default__): CORRE ABSOLUTAMENTE TODO
    default_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        class_pipe,
        reg_pipe,
        unsup_pipe,
        int_pipe  # <--- Agregamos la integración al flujo principal
    ])

    # --- 4. Registro ---
    return {
        "dp": dp_pipeline,
        "cl": cl_pipeline,
        "rg": rg_pipeline,
        "ul": ul_pipeline,
        "int": int_pipeline,    # <--- Pipeline específico para probar la integración
        "__default__": default_pipeline,
    }