"""
Nodos para preprocesamiento y unión de datos.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import logging

# --- Nodo 1: Combinar y Limpiar Datos ---

def create_primary_table(
    customer_data: pd.DataFrame,
    product_data: pd.DataFrame,
    purchase_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Combina los 3 DataFrames crudos en una tabla primaria 
    y realiza la limpieza inicial.
    """

    # --- 1. Lógica de Unión (Merge) ---
    logging.info("Uniendo 'purchase' y 'customer'...")
    merged_df = pd.merge(
        purchase_data, 
        customer_data, 
        on="customer_id",  # Confirmado
        how="left"
    )
    
    logging.info("Uniendo 'merged_df' y 'product'...")
    primary_table = pd.merge(
        merged_df,
        product_data,
        on="product_id", # Confirmado
        how="left"
    )

    # --- 2. Lógica de Limpieza (Opcional pero recomendado) ---
    # Si sabes que tienes nulos en los targets, descomenta la siguiente línea:
    # primary_table = primary_table.dropna(subset=["category", "total_amount"])
    
    logging.info(f"Tabla primaria creada con {primary_table.shape[0]} filas.")
    
    return primary_table

# --- Nodo 2: Crear Features y Dividir Datos ---

def create_features(
    primary_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en features y targets para clasificación y regresión, 
    y luego los divide en sets de entrenamiento y prueba.
    """
    
    # --- ✅ Targets Definidos ---
    TARGET_CLASSIFICATION = "category" 
    TARGET_REGRESSION = "total_amount" 
    
    # --- ✅ Features a Eliminar ---
    drop_cols = [
        # Targets
        TARGET_CLASSIFICATION, 
        TARGET_REGRESSION, 
        
        # IDs
        "customer_id",
        "product_id",
        "purchase_id",
        
        # Columnas de texto/fecha inútiles para el modelo
        "first_name",
        "last_name",
        "date_of_birth",
        "email",
        "phone_number",
        "signup_date",
        "address",
        "zip_code", # Categórica con demasiados valores
        "product_name", # Redundante con category y brand
        "product_description",
        "purchase_date",
        
        # Columnas con "fugas" (leaks) que harían el problema trivial
        "price_per_unit",
        "quantity"
    ]
    
    # Revisa si las columnas existen antes de intentar dropearlas
    cols_to_drop_existing = [col for col in drop_cols if col in primary_data.columns]
    
    logging.info(f"Eliminando {len(cols_to_drop_existing)} columnas no usadas para features.")
    features_df = primary_data.drop(columns=cols_to_drop_existing)
    
    # --- Define tus targets (y) ---
    try:
        target_classification = primary_data[TARGET_CLASSIFICATION]
        target_regression = primary_data[TARGET_REGRESSION]
    except KeyError as e:
        logging.error(f"Error: La columna target {e} no se encuentra en la tabla.")
        logging.error("Asegúrate de que los CSVs se unieron correctamente.")
        raise e

    # --- Maneja variables categóricas (One-Hot Encoding) ---
    # Convierte todas las columnas de texto/objeto restantes (como gender, city, state, brand) en números.
    logging.info("Aplicando One-Hot Encoding a variables categóricas...")
    features_encoded = pd.get_dummies(features_df, drop_first=True)
    
    # --- División para Clasificación ---
    logging.info("Dividiendo datos para Clasificación...")
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        features_encoded, target_classification, test_size=0.2, random_state=42
    )
    
    # --- División para Regresión ---
    logging.info("Dividiendo datos para Regresión...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        features_encoded, target_regression, test_size=0.2, random_state=42
    )
    
    # --- Reindexa y asegura formato compatible con Parquet ---
    X_train_class = X_train_class.reset_index(drop=True)
    X_test_class = X_test_class.reset_index(drop=True)
    X_train_reg = X_train_reg.reset_index(drop=True)
    X_test_reg = X_test_reg.reset_index(drop=True)

    y_train_class = y_train_class.reset_index(drop=True).to_frame(name=TARGET_CLASSIFICATION)
    y_test_class = y_test_class.reset_index(drop=True).to_frame(name=TARGET_CLASSIFICATION)
    y_train_reg = y_train_reg.reset_index(drop=True).to_frame(name=TARGET_REGRESSION)
    y_test_reg = y_test_reg.reset_index(drop=True).to_frame(name=TARGET_REGRESSION)

    logging.info("Features creados y divididos para clasificación y regresión.")
    
    # Retorna todas las divisiones de datos
    return (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    )


    # ... (al final de preprocessing.py)

def create_integrated_features(
    data_clustered: pd.DataFrame, 
    target_col: str = "category"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepara los datos para el modelo integrado.
    Usa el DataFrame que salió del clustering (que ya tiene 'cluster_kmeans', 'age', etc.)
    """
    # Definir X e y
    # data_clustered ya viene con todo listo del pipeline anterior
    X = data_clustered.drop(columns=[target_col])
    y = data_clustered[target_col]
    
    # Split (mismo random_state para comparar peras con peras)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test