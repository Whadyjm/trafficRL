# Sistema de Control de Tráfico Híbrido con RL (PPO)

Este proyecto implementa un sistema de control de semáforos inteligente utilizando Aprendizaje por Refuerzo (Reinforcement Learning - RL) con el algoritmo PPO (Proximal Policy Optimization). El sistema está diseñado para operar en un entorno de simulación SUMO, priorizando vehículos de emergencia (ambulancias) y peatones, mientras optimiza el flujo vehicular general.

## 🧠 Lógica del Sistema (Hybrid Control)

El "cerebro" del sistema no es solo una red neuronal; es un controlador híbrido que toma decisiones basadas en una jerarquía de prioridades estricta implementada en la clase `EntornoOptimizado`:

### 1. Prioridad Máxima: Emergencia (Ambulancia) 🚑
*   **Detección**: El sistema escanea constantemente la red buscando vehículos de tipo "ambulancia".
*   **Lógica**: Si una ambulancia es detectada en un carril controlado por el semáforo:
    1.  Identifica qué fase verde permite el paso a ese carril específico.
    2.  **Override**: Ignora cualquier decisión del modelo RL o temporizador.
    3.  **Acción Inmediata**: Fuerza el cambio a la fase verde de la ambulancia, manipulando internamente los contadores de `min_green` para evitar retrasos de seguridad estándar del semáforo.
*   **Objetivo**: Tiempo de espera cero para emergencias.

### 2. Prioridad Secundaria: Horario Peatonal Programado 🚶
*   **Condición**: Si **no** hay ambulancia presente Y **hay peatones activos** (esperando o cruzando).
*   **Lógica**: Se basa en el tiempo de ciclo de la simulación (137 segundos en total).
    *   **Ventana 1 (Segundos 23-38)**: Se fuerza la **Fase Peatonal 1** SOLO si se detecta actividad peatonal.
    *   **Ventana 2 (Segundos 122-137)**: Se fuerza la **Fase Peatonal 2** SOLO si se detecta actividad peatonal.
*   **Objetivo**: Garantizar ventanas de cruce seguras para peatones cuando son necesarias, evitando detener el tráfico vehicular innecesariamente si no hay nadie esperando.

### 3. Prioridad Terciaria: Agente Inteligente (RL - PPO) 🤖
*   **Condición**: Si no hay emergencias ni es horario peatonal reservado.
*   **Lógica**: El modelo PPO toma el control total.
*   **Input (Observación)**: Recibe un vector que incluye:
    *   Fase actual (One-hot encoding).
    *   Tiempo mínimo de verde cumplido (Binario).
    *   Densidad y cantidad de vehículos en cola por carril.
    *   Flags de presencia de ambulancia (para que aprenda a anticipar, aunque la regla 1 fuerce la acción).
*   **Output (Acción)**: Selecciona la siguiente fase verde óptima para minimizar la función de coste (Reward).

---

## 📂 Archivos Principales

### 1. `entrenamiento.py`
Script encargado de entrenar el modelo. Define la "Función de Recompensa" que guía el aprendizaje:

*   **Recompensas (+) y Castigos (-)**:
    *   **Ambulancia**: `-1000 * (tiempo_espera^2)` (Castigo extremo si espera) | `+500` (Premio si fluye).
    *   **Peatones**: `-400 * (espera^2)` (Evitar aglomeraciones) | `+150` por cada peatón que cruza.
    *   **Vehículos**: `-5 * (espera^2)` (Fluidez general) | `-50` por "Tiempos Muertos" (semáforo en verde sin autos pasando).
    *   **Seguridad**: `-2000` por colisiones | `-100` por infracciones (cruzar en rojo/jaywalking).

### 2. `despliegue.py`
Script para probar y visualizar el modelo entrenado en tiempo real.
*   **Consistencia**: Importa y redefine la misma clase `EntornoOptimizado` y `AmbulanceObservationFunction` que el entrenamiento. Esto es crucial para que el modelo cargado interprete el estado de la simulación exactamente igual que como fue entrenado.
*   **Visualización**: Ejecuta SUMO con interfaz gráfica (`use_gui=True`) y muestra métricas en consola.

---

## 🚀 Cómo Ejecutar

### Requisitos Previos
*   Python 3.x
*   SUMO (Simulation of Urban MObility) instalado y en el PATH.
*   Librerías Python: `sumo-rl`, `stable-baselines3`, `gymnasium`, `traci`, `pandas`, `matplotlib`.

### 1. Entrenamiento
Ejecuta el script para iniciar el proceso de aprendizaje. Esto creará el archivo del modelo `.zip` y guardará logs de progreso.
```bash
python entrenamiento.py
```
*Salida*: `trigal_model_ambulancia.zip` y carpeta `outputs_optimizados/`.

### 2. Despliegue (Inferencia)
Una vez entrenado (o si ya tienes el `.zip`), ejecuta el despliegue para ver al agente en acción.
```bash
python despliegue.py
```
*Nota*: Si el modelo no existe, el script fallará. Asegúrate de haber entrenado primero.
