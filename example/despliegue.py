# despliegue.py - ESTADO 11D (COMPATIBLE CON CustomSumoEnv)
import traci
from stable_baselines3 import PPO
import numpy as np
import time
import os

# === CONFIG ===
NET_CFG = "single-intersection.sumocfg"
MODEL_PATH = "modelos/modelo_ppo_latest.zip"
TLS_ID = "t"
INCOMING_LANES = ["n_t_0", "n_t_1", "w_t_0", "w_t_1"]

# Fases reales
PHASES = {
    0: "GGrr",  # Norte-Sur
    1: "rrGG"   # Oeste-Este
}

# === CARGAR MODELO ===
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Modelo no encontrado: {MODEL_PATH}")
    exit(1)

print(f"Cargando modelo: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)  # env=None no necesario si el estado es 11D
print("Modelo cargado")

# === INICIAR SUMO ===
sumoCmd = ["sumo-gui", "-c", NET_CFG, "--start", "--quit-on-end", "--delay", "100"]
traci.start(sumoCmd)
print("Simulación iniciada")

# === MÉTRICAS ===
step = 0
total_reward = 0
total_waiting = 0
prev_queue = 0
queue_history = []
start_time = time.time()

avg_waiting = 0.0

print("\n" + "="*70)
print(" DESPLIEGUE EN VIVO - AGENTE RL (11D)")
print("="*70)

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # === ESTADO 11D: IGUAL QUE EN CustomSumoEnv ===
        # 1. Autos en carriles (4)
        lane_counts = np.array([
            traci.lane.getLastStepVehicleNumber(lane) for lane in INCOMING_LANES
        ], dtype=np.float32)

        # 2. Fase actual (one-hot: 4 dimensiones)
        current_phase_str = traci.trafficlight.getRedYellowGreenState(TLS_ID)
        phase_one_hot = np.zeros(4, dtype=np.float32)
        if current_phase_str == "GGrr":
            phase_one_hot[0] = 1.0
        elif current_phase_str == "rrGG":
            phase_one_hot[1] = 1.0
        elif current_phase_str in ["yyrr", "rryy"]:
            phase_one_hot[2] = 1.0
        else:
            phase_one_hot[3] = 1.0

        # 3. Tiempo restante en fase (normalizado)
        next_switch = traci.trafficlight.getNextSwitch(TLS_ID)
        current_time = traci.simulation.getTime()
        time_left = max(0, next_switch - current_time) / 60.0  # max 60s

        # 4. Estado completo: 4 + 4 + 1 + 2 = 11
        state = np.concatenate([
            lane_counts,           # 4
            phase_one_hot,         # 4
            [time_left],           # 1
            [0.0, 0.0]             # 2 (relleno, como en sumo-rl)
        ]).astype(np.float32)

        # === PREDICCIÓN ===
        action, _ = model.predict(state, deterministic=True)
        action = int(action) % 2

        # === APLICAR CON AMARILLO ===
        target_phase = PHASES[action]
        current_state = traci.trafficlight.getRedYellowGreenState(TLS_ID)

        if current_state in ["GGrr", "rrGG"] and current_state != target_phase:
            yellow = "yyrr" if current_state == "GGrr" else "rryy"
            traci.trafficlight.setRedYellowGreenState(TLS_ID, yellow)
            for _ in range(3):
                traci.simulationStep()
                time.sleep(0.1)
            traci.trafficlight.setRedYellowGreenState(TLS_ID, target_phase)
        else:
            traci.trafficlight.setRedYellowGreenState(TLS_ID, target_phase)

        # === MÉTRICAS ===
        current_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES)
        diff_queue = current_queue - prev_queue
        reward = -abs(diff_queue)
        total_reward += reward
        total_waiting += current_queue
        prev_queue = current_queue
        queue_history.append(current_queue)

        avg_waiting = total_waiting / step

        # === CONSOLA ===
        elapsed = time.time() - start_time
        print(f"\rPaso {step:4d} | Cola: {current_queue:2.0f} | "
              f"Prom: {avg_waiting:4.2f}s | Acción: {'N-S' if action==0 else 'O-E'} | "
              f"T: {elapsed:3.0f}s", end="")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nDetenido.")
except Exception as e:
    print(f"\nError: {e}")
finally:
    traci.close()
    print("\n\n" + "="*70)
    print(" SIMULACIÓN TERMINADA")
    print(f"   → Cola promedio: {avg_waiting:.2f} autos")
    print(f"   → Tiempo de espera: {avg_waiting:.2f}s")
    print("="*70)
    print(" ¡DESPLIEGUE EXITOSO!")