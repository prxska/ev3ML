"""
Pipeline para todos los modelos de Regresión.
"""
from kedro.pipeline import Pipeline, node
from ..nodes.modeling import (
    train_linear_regression,
    train_random_forest_reg,
    train_knn_reg,
    train_svr_reg,
    train_gb_reg,
    evaluate_regression_model  # El nodo de evaluación genérico
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        
        # --- Modelo 1: Regresión Lineal ---
        node(
            func=train_linear_regression,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="model_linear_regression",
            name="train_linear_regression_node"
        ),
        node(
            func=evaluate_regression_model,
            inputs=["model_linear_regression", "X_test_reg", "y_test_reg"],
            outputs="metrics_linear_regression",
            name="evaluate_linear_regression_node"
        ),
        
        # --- Modelo 2: Random Forest Regressor ---
        node(
            func=train_random_forest_reg,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="model_random_forest_reg",
            name="train_rf_reg_node"
        ),
        node(
            func=evaluate_regression_model,
            inputs=["model_random_forest_reg", "X_test_reg", "y_test_reg"],
            outputs="metrics_random_forest_reg",
            name="evaluate_rf_reg_node"
        ),

        # --- Modelo 3: KNN Regressor ---
        node(
            func=train_knn_reg,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="model_knn_reg",
            name="train_knn_reg_node"
        ),
        node(
            func=evaluate_regression_model,
            inputs=["model_knn_reg", "X_test_reg", "y_test_reg"],
            outputs="metrics_knn_reg",
            name="evaluate_knn_reg_node"
        ),
        
        # --- Modelo 4: SVR ---
        node(
            func=train_svr_reg,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="model_svr_reg",
            name="train_svr_reg_node"
        ),
        node(
            func=evaluate_regression_model,
            inputs=["model_svr_reg", "X_test_reg", "y_test_reg"],
            outputs="metrics_svr_reg",
            name="evaluate_svr_reg_node"
        ),
        
        # --- Modelo 5: Gradient Boosting Regressor ---
        node(
            func=train_gb_reg,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="model_gb_reg",
            name="train_gb_reg_node"
        ),
        node(
            func=evaluate_regression_model,
            inputs=["model_gb_reg", "X_test_reg", "y_test_reg"],
            outputs="metrics_gb_reg",
            name="evaluate_gb_reg_node"
        )
    ])