# Arquitectura Técnica del Sistema MLOps

## 1. Visión General
El proyecto implementa una arquitectura de microservicios contenerizados para orquestar un flujo de trabajo de Machine Learning completo (ETL, Clustering, Entrenamiento).

## 2. Componentes del Stack

### A. Orquestación (Apache Airflow)
- **DAG Maestro:** `proyecto_final_mlops_v3`
- **Función:** Controla la ejecución secuencial de los pipelines de Kedro.
- **Estrategia:** Utiliza `BashOperator` para ejecutar comandos dentro del entorno compartido de Docker.

### B. Pipelines de Procesamiento (Kedro)
El código se estructura en tres pipelines modulares:
1.  **`dp` (Data Processing):** Ingesta y limpieza de datos crudos.
2.  **`ul` (Unsupervised Learning):**
    - Feature Engineering (Edad, Mes).
    - Entrenamiento de K-Means, DBSCAN, Hierarchical.
    - Generación de `cluster_id`.
3.  **`int` (Integration):**
    - Fusión de datos originales + clusters.
    - Entrenamiento del modelo final (Random Forest).

### C. Infraestructura (Docker & Docker Compose)
- **Servicios:**
    - `postgres`: Base de datos de metadatos para Airflow.
    - `airflow-webserver`: Interfaz gráfica de usuario.
    - `airflow-scheduler`: Motor de planificación.
- **Persistencia:** Se utilizan volúmenes de Docker para compartir el código y los datos entre el host y los contenedores, permitiendo desarrollo en tiempo real.

### D. Versionado de Datos (DVC)
Se utiliza DVC para rastrear los artefactos binarios grandes que no deben ir a Git:
- Datasets procesados (`.parquet`).
- Modelos serializados (`.pkl`).
- Métricas de evaluación (`.json`).