"""Registra los pipelines del proyecto."""
from typing import Dict
from kedro.pipeline import Pipeline, node

# Importa los nodos de preprocesamiento
from .nodes.preprocessing import create_primary_table, create_features

# Importa las definiciones de los pipelines
from .pipelines import classification_pipeline, regression_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Registra los pipelines del proyecto.

    Returns:
        Un diccionario mapeando nombres de pipelines a instancias de Pipeline.
    """
    
    # --- 1. Definición de Nodos Individuales ---
    
    # Nodo que une los 3 CSVs
    data_processing_node = node(
        func=create_primary_table,
        inputs=["customer_profile", "products", "purchase_history"],
        outputs="primary_data",
        name="create_primary_table_node"
    )
    
    # Nodo que crea los features (X_train, y_train, etc.)
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
    
    # Carga los pipelines de modelado desde sus archivos
    class_pipe = classification_pipeline.create_pipeline()
    reg_pipe = regression_pipeline.create_pipeline()

    # --- 3. Ensamblaje de Pipelines Completos ---

    # Pipeline de datos (dp): Solo une los CSVs
    dp_pipeline = Pipeline([data_processing_node])
    
    # Pipeline de clasificación (cl): Une CSVs -> Crea Features -> Entrena 5 modelos
    cl_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        class_pipe  # <-- Añade el pipeline de clasificación completo
    ])
    
    # Pipeline de regresión (rg): Une CSVs -> Crea Features -> Entrena 5 modelos
    rg_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        reg_pipe # <-- Añade el pipeline de regresión completo
    ])
    
    # Pipeline por defecto (__default__): Corre TODO
    default_pipeline = Pipeline([
        data_processing_node,
        feature_engineering_node,
        class_pipe,
        reg_pipe
    ])

    # --- 4. Registro ---
    return {
        "dp": dp_pipeline,
        "cl": cl_pipeline,
        "rg": rg_pipeline,
        "__default__": default_pipeline,
    }