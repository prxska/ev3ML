# Proyecto Final MLOps: Integración Supervisada, No Supervisada y Orquestación

Este proyecto representa la culminación del curso de Machine Learning, implementando un pipeline `end-to-end` robusto que integra ingeniería de datos, aprendizaje no supervisado (Clustering) y modelos supervisados avanzados, todo orquestado automáticamente.

**Asignatura:** MLY0100 - Machine Learning
**Integrantes:**
* Jorge Garrido


---

## 1. Objetivos del Proyecto

El sistema analiza el comportamiento de compra de clientes para resolver dos problemas predictivos, utilizando una arquitectura moderna de MLOps:

1.  **Clasificación:** Predecir la **Categoría de Producto** (`category`) que comprará un cliente.
2.  **Regresión:** Predecir el **Monto Total** (`total_amount`) de la transacción.

---

## 2. Arquitectura Técnica (Stack MLOps)

La solución utiliza un stack tecnológico avanzado para garantizar reproducibilidad y escalabilidad:

* **Kedro:** Framework principal para la estructuración de pipelines modulares.
    * *Pipeline `dp`:* Procesamiento y limpieza de datos.
    * *Pipeline `ul`:* **Aprendizaje No Supervisado** (K-Means, DBSCAN, Hierarchical, PCA, t-SNE).
    * *Pipeline `int`:* Integración y entrenamiento del modelo final.
* **Apache Airflow:** Orquestador de tareas. Gestiona la ejecución secuencial y dependencias de los pipelines mediante un DAG maestro.
* **Docker & Docker Compose:** Infraestructura como código. Levanta servicios independientes para la Base de Datos (Postgres), el Webserver de Airflow, el Scheduler y el entorno de ejecución de Python.
* **DVC (Data Version Control):** Versionado de datasets, modelos (`.pkl`) y métricas, asegurando la trazabilidad de los experimentos.

---

## 3. Metodología de Ciencia de Datos

### A. Fase No Supervisada (Feature Engineering Avanzado)
Para mejorar la capacidad predictiva, se implementaron técnicas de agrupamiento y reducción de dimensionalidad:
* **Clustering:** Se utilizaron algoritmos como **K-Means (k=5)**, **DBSCAN** y **Clustering Jerárquico** para segmentar a los clientes en perfiles de comportamiento.
* **Reducción:** Se aplicó **PCA** y **t-SNE** para analizar la varianza y estructura de los datos.
* **Detección de Anomalías:** Se implementó **Isolation Forest** para identificar transacciones atípicas.

### B. Fase de Integración (Supervisado "Supercharged")
Los clusters generados y los features temporales (Edad, Mes, Día) se inyectaron como nuevas variables predictivas (*features*) en un modelo de **Random Forest**.

---

## 4. Resultados y Comparativa (Ev2 vs Ev3)

Gracias a la integración del aprendizaje no supervisado y la optimización de features, se logró un incremento drástico en el rendimiento del modelo.

### Tabla Comparativa de Clasificación (Accuracy)

| Etapa | Modelo | Accuracy | Estado |
| :--- | :--- | :--- | :--- |
| **Evaluación 2** | Regresión Logística (Baseline) | 30.80% | ❌ Insuficiente |
| **Evaluación 3** | **Random Forest + Clustering** | **83.41%** | ✅ **Éxito (+52.6%)** |

**Conclusión del Análisis:**
El modelo original (Ev2) carecía de información suficiente para distinguir patrones complejos. La segmentación de clientes mediante Clustering y la inclusión de atributos granulares del producto permitieron al modelo Random Forest capturar la lógica de compra con alta precisión.

---

## 5. Instrucciones de Ejecución (Despliegue)

El proyecto está completamente contenerizado. Para ejecutar el sistema completo (Airflow + Pipelines):

**Prerrequisitos:**
* Docker Desktop instalado y corriendo.

**Pasos:**

1.  Clonar el repositorio y entrar a la carpeta de Docker:
    ```bash
    cd proyecto-ml-final/docker
    ```

2.  Levantar la infraestructura con Docker Compose:
    ```bash
    docker-compose up --build
    ```
    *(Esperar a que inicien los servicios postgres, webserver y scheduler).*

3.  Acceder a la interfaz de Airflow:
    * **URL:** [http://localhost:8080](http://localhost:8080)
    * **Usuario:** `admin`
    * **Contraseña:** `admin`

4.  Ejecutar el Pipeline Maestro:
    * Buscar el DAG **`proyecto_final_mlops_v3`**.
    * Activar el interruptor (**ON**) y hacer clic en **Trigger DAG** (Play ▶️).
    * Observar en la vista "Graph" cómo se ejecutan secuencialmente: `Data Processing` -> `Unsupervised Learning` -> `Model Integration`.

---

**Estado del Proyecto:** Finalizado y Funcional. 🚀
