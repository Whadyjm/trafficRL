# entrenamiento.py
# CONTROL TOTAL: elige cualquier fase, duraciones, prioridad a esperas largas y flujo

import os
import sumo_rl
from sumo_rl.environment.traffic_signal import TrafficSignal
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURACIÓN ===
NET_FILE = "trigal.net.xml"
ROUTE_FILE = "trigal.rou.xml"
MODEL_PATH = "trigal_model.zip"
OUTPUT_DIR = "outputs"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 10000
SIM_SECONDS = 3600

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("INICIANDO ENTRENAMIENTO CON CONTROL AVANZADO")
print("-" * 60)

# === CUSTOM TRAFFIC SIGNAL (elige cualquier fase) ===
class CustomTrafficSignal(TrafficSignal):
    def __init__(self, env, ts_id, simulation, program_id, delta_time, min_green, max_green, yellow_time, phases):
        super().__init__(env, ts_id, simulation, program_id, delta_time, min_green, max_green, yellow_time, phases)
        self.action_space = gym.spaces.Discrete(len(self.phases))  # Acción = fase a activar

    def act(self, action):
        if action != self.phase:
            self.set_phase(action)  # Cambia directamente a la fase elegida
        # Si es la misma, mantiene → extiende duración

# === REWARD AVANZADO (usa atributos internos) ===
def reward_avanzado(ts):
    wait_time = getattr(ts, '_waiting_time', 0)
    stopped = getattr(ts, '_total_queued', 0)
    pressure = getattr(ts, '_pressure', 0)
    lane_waits = list(getattr(ts, '_lane_waiting_time', {}).values())
    max_wait = max(lane_waits) if lane_waits else 0

    return - (0.4 * wait_time + 0.2 * stopped + 0.3 * max_wait + 0.1 * abs(pressure))

# === CALLBACK ===
class ProgressLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.data = []

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            info = self.locals.get('infos', [{}])[0]
            reward = self.locals.get('rewards', [0])[0]
            waiting = info.get('system_total_waiting_time', 0)
            stopped = info.get('system_total_stopped', 0)

            self.data.append({
                'timestep': self.num_timesteps,
                'reward': reward,
                'waiting_time': waiting,
                'stopped_vehicles': stopped
            })
            print(f"Step {self.num_timesteps:5d} | R: {reward:8.1f} | Wait: {waiting:6.0f}s | Stop: {stopped:3.0f}")
        return True

    def save_and_plot(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        df.to_csv(LOG_CSV, index=False)
        print(f"CSV guardado: {LOG_CSV}")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        ax1.plot(df['timestep'], df['reward'], 'g-', linewidth=2)
        ax1.set_title('Reward Avanzado')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df['timestep'], df['waiting_time'], 'r-', linewidth=2)
        ax2.set_title('Tiempo de Espera Total')
        ax2.set_ylabel('Tiempo (s)')
        ax2.grid(True, alpha=0.3)

        ax3.plot(df['timestep'], df['stopped_vehicles'], 'b-', linewidth=2)
        ax3.set_title('Vehículos Parados')
        ax3.set_xlabel('Timesteps')
        ax3.set_ylabel('N° Vehículos')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# === ENTORNO PERSONALIZADO ===
class CustomSumoEnvironment(sumo_rl.SumoEnvironment):
    def _get_traffic_signal(self, ts_id, tl_logic, phases, program_id):
        return CustomTrafficSignal(self, ts_id, self.simulation, program_id, self.delta_time, self.min_green, self.max_green, self.yellow_time, phases)

env = CustomSumoEnvironment(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=os.path.join(OUTPUT_DIR, "trigal_train"),
    use_gui=False,
    num_seconds=SIM_SECONDS,
    single_agent=True,
    min_green=10,
    max_green=60,
    yellow_time=3,
    delta_time=10,
    reward_fn=reward_avanzado
)

# === MODELO ===
callback = ProgressLoggerCallback()
if os.path.exists(MODEL_PATH):
    print(f"CARGANDO MODELO: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("CREANDO NUEVO MODELO")
    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)

# === ENTRENAR ===
print(f"\nINICIANDO ENTRENAMIENTO...")
model.learn(total_timesteps=TIMESTEPS, callback=callback)
print("ENTRENAMIENTO TERMINADO")

# === GUARDAR ===
model.save(MODEL_PATH)
print(f"Modelo guardado: {MODEL_PATH}")
callback.save_and_plot()

env.close()
print("\n¡COMPLETO! El modelo ahora controla todo.")