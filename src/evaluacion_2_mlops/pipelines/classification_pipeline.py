
from kedro.pipeline import Pipeline, node
from ..nodes.modeling import (
    train_logistic_regression,
    train_random_forest_class,
    train_knn_class,
    train_svc_class,
    train_gb_class,
    evaluate_classification_model  # El nodo de evaluación genérico
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        
        # --- Modelo 1: Regresión Logística ---
        node(
            func=train_logistic_regression,
            inputs=["X_train_class", "y_train_class"],
            outputs="model_logistic_regression",
            name="train_logistic_regression_node"
        ),
        node(
            func=evaluate_classification_model,
            inputs=["model_logistic_regression", "X_test_class", "y_test_class"],
            outputs="metrics_logistic_regression",
            name="evaluate_logistic_regression_node"
        ),
        
        # --- Modelo 2: Random Forest Classifier ---
        node(
            func=train_random_forest_class,
            inputs=["X_train_class", "y_train_class"],
            outputs="model_random_forest_class",
            name="train_rf_class_node"
        ),
        node(
            func=evaluate_classification_model,
            inputs=["model_random_forest_class", "X_test_class", "y_test_class"],
            outputs="metrics_random_forest_class",
            name="evaluate_rf_class_node"
        ),

        # --- Modelo 3: KNN Classifier ---
        node(
            func=train_knn_class,
            inputs=["X_train_class", "y_train_class"],
            outputs="model_knn_class",
            name="train_knn_class_node"
        ),
        node(
            func=evaluate_classification_model,
            inputs=["model_knn_class", "X_test_class", "y_test_class"],
            outputs="metrics_knn_class",
            name="evaluate_knn_class_node"
        ),
        
        # --- Modelo 4: SVC ---
        node(
            func=train_svc_class,
            inputs=["X_train_class", "y_train_class"],
            outputs="model_svc_class",
            name="train_svc_class_node"
        ),
        node(
            func=evaluate_classification_model,
            inputs=["model_svc_class", "X_test_class", "y_test_class"],
            outputs="metrics_svc_class",
            name="evaluate_svc_class_node"
        ),
        
        # --- Modelo 5: Gradient Boosting Classifier ---
        node(
            func=train_gb_class,
            inputs=["X_train_class", "y_train_class"],
            outputs="model_gb_class",
            name="train_gb_class_node"
        ),
        node(
            func=evaluate_classification_model,
            inputs=["model_gb_class", "X_test_class", "y_test_class"],
            outputs="metrics_gb_class",
            name="evaluate_gb_class_node"
        )
    ])