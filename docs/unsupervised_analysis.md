# Análisis de Aprendizaje No Supervisado y Reporte Comparativo

## 1. Metodología de Experimentación
El objetivo fue utilizar técnicas de agrupamiento (Clustering) para segmentar clientes y usar esos segmentos como nuevos *features* para mejorar la predicción.

### Algoritmos Implementados
1.  **K-Means (k=5):** Algoritmo de partición. Se seleccionó k=5 tras pruebas experimentales. Fue el más eficiente computacionalmente.
2.  **DBSCAN (eps=3, min=5):** Basado en densidad. Resultó menos efectivo debido a la alta dimensionalidad de los datos (detectó mucho ruido).
3.  **Hierarchical Clustering:** Aglomerativo. Útil para visualizar la estructura dendriforme, pero costoso en memoria.

### Reducción de Dimensionalidad
Se aplicó **PCA (Principal Component Analysis)** reduciendo el espacio a 2 componentes principales, explicando un **26.8%** de la varianza. Esto confirmó la complejidad no lineal de los datos.

---

## 2. Resultados Comparativos (El Salto de Calidad)

Se comparó el rendimiento del modelo base (Evaluación 2) contra el modelo integrado con clustering (Evaluación 3).

### Tarea: Clasificación (Predicción de Categoría)

| Métrica | Modelo Base (Ev2) | Modelo Integrado (Ev3) | Diferencia |
| :--- | :--- | :--- | :--- |
| **Algoritmo** | Regresión Logística | Random Forest + K-Means | - |
| **Accuracy** | 30.80% | **83.41%** | **+52.61%** |
| **F1-Score** | 0.258 | **0.832** | **+0.574** |

**Análisis de Mejora:**
El modelo base (Ev2) tenía un desempeño deficiente. Al incorporar los **Clusters** como variables predictoras y enriquecer los datos con atributos del producto (precio, nombre), el modelo logró identificar patrones de compra con una precisión superior al 80%.

---

## 3. Conclusión Final
La integración de técnicas no supervisadas fue exitosa. Los clusters actuaron como una variable sintética que resume el perfil del cliente, permitiendo al modelo supervisado (Random Forest) tomar decisiones mucho más acertadas que con los datos demográficos crudos.