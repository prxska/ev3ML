import pandas as pd
import logging
from typing import Dict, Any

from sklearn.model_selection import GridSearchCV

# --- Modelos de Clasificación ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --- Modelos de Regresión ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# --- Métricas ---
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Configura el logger
log = logging.getLogger(__name__)


def _squeeze_target(target):
    """Convierte DataFrames de una sola columna en Series."""
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("El target debe tener exactamente una columna.")
    return target.squeeze()

# =================================================================
# === 1. NODOS DE ENTRENAMIENTO DE CLASIFICACIÓN (5 MODELOS) ===
# =================================================================

def train_logistic_regression(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Regresión Logística con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {'C': [0.1, 1.0, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (LogReg): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_random_forest_class(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Random Forest Classifier con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (RF Class): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_knn_class(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de KNN Classifier con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (KNN Class): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_svc_class(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de SVC con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = SVC(random_state=42)
    param_grid = {'C': [0.1, 1.0], 'kernel': ['linear']} # Evitamos 'rbf' por velocidad
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (SVC): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_gb_class(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Gradient Boosting Classifier con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (GB Class): {grid_search.best_params_}")
    return grid_search.best_estimator_

# =================================================================
# === 2. NODOS DE ENTRENAMIENTO DE REGRESIÓN (5 MODELOS) ===
# =================================================================

def train_linear_regression(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Regresión Lineal con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (LinReg): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_random_forest_reg(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Random Forest Regressor con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = RandomForestRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (RF Reg): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_knn_reg(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de KNN Regressor con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = KNeighborsRegressor()
    param_grid = {'n_neighbors': [3, 5, 7]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (KNN Reg): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_svr_reg(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de SVR con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = SVR()
    param_grid = {'C': [0.1, 1.0], 'kernel': ['linear']} # Evitamos 'rbf' por velocidad
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (SVR): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_gb_reg(X_train: pd.DataFrame, y_train) -> Any:
    """Entrena un modelo de Gradient Boosting Regressor con GridSearchCV."""
    y_train = _squeeze_target(y_train)
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    log.info(f"Mejores params (GB Reg): {grid_search.best_params_}")
    return grid_search.best_estimator_

# =================================================================
# === 3. NODOS DE EVALUACIÓN (2 NODOS) ===
# =================================================================

def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test) -> Dict[str, float]:
    """Calcula métricas para un modelo de clasificación."""
    y_test = _squeeze_target(y_test)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    metrics = {"accuracy": accuracy, "f1_score": f1}
    log.info(f"Métricas del modelo {type(model).__name__}: {metrics}")
    return metrics

def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test) -> Dict[str, float]:
    """Calcula métricas para un modelo de regresión."""
    y_test = _squeeze_target(y_test)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    
    metrics = {"rmse": rmse, "r2_score": r2}
    log.info(f"Métricas del modelo {type(model).__name__}: {metrics}")
    return metrics