# entrenamiento.py - REANUDABLE, VERSIONADO, CON CSV Y GRÁFICOS
import gymnasium as gym
from sumo_rl.environment.env import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import traci
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
NET_FILE = 'single-intersection.net.xml'
ROUTE_FILE = 'single-intersection.rou.xml'
OUT_DIR = 'outputs'
TLS_ID = 't'

# === MODEL & CSV CON TIMESTAMP ===
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
MODEL_DIR = "modelos"
MODEL_PATH = os.path.join(MODEL_DIR, f"modelo_ppo_{TIMESTAMP}.zip")
LATEST_MODEL = os.path.join(MODEL_DIR, "modelo_ppo_latest.zip")

# === CSV CON TIMESTAMP ===
CSV_EPISODES = os.path.join(OUT_DIR, f"rl_output_episodios_{TIMESTAMP}.csv")
CSV_STEPS = os.path.join(OUT_DIR, f"rl_output_pasos_{TIMESTAMP}.csv")

# === CREAR CARPETAS ===
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === ENTORNO PERSONALIZADO ===
class CustomSumoEnv(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tls_id = TLS_ID
        self.in_lanes = []
        self.prev_queue = 0
        self.step_data = []
        self.episode_data = []
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_queue_sum = 0

    def _get_in_lanes(self):
        if not self.in_lanes and traci.isLoaded():
            try:
                links = traci.trafficlight.getControlledLinks(self.tls_id)
                lanes = set()
                for link in links:
                    if link and link[0] and link[0][0]:
                        lanes.add(link[0][0])
                self.in_lanes = list(lanes)
                if self.in_lanes:
                    print(f"Carriles detectados: {self.in_lanes}")
                else:
                    print("ADVERTENCIA: No se detectaron carriles. Usando fallback.")
                    self.in_lanes = ["n_t_0", "n_t_1", "w_t_0", "w_t_1"]
            except:
                self.in_lanes = ["n_t_0", "n_t_1", "w_t_0", "w_t_1"]
        return self.in_lanes

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        in_lanes = self._get_in_lanes()
        current_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in in_lanes)

        diff_queue = current_queue - self.prev_queue
        reward = -abs(diff_queue)
        self.prev_queue = current_queue

        info['system_total_waiting_time'] = current_queue
        info['queue_length'] = current_queue

        self.step_data.append({
            'episode': self.current_episode,
            'step': self.episode_steps,
            'reward': reward,
            'queue': current_queue,
            'action': action
        })

        self.episode_steps += 1
        self.episode_queue_sum += current_queue

        if done:
            avg_queue = self.episode_queue_sum / max(1, self.episode_steps)
            total_reward = sum(r['reward'] for r in self.step_data[-self.episode_steps:])
            ep_row = {
                'episode': self.current_episode,
                'reward': total_reward,
                'waiting_time': avg_queue,
                'steps': self.episode_steps,
                'avg_queue': avg_queue
            }
            self.episode_data.append(ep_row)
            print(f"Episodio {self.current_episode} | Reward: {total_reward:.1f} | Cola: {avg_queue:.1f}")

            self._save_csvs()

            self.episode_steps = 0
            self.episode_queue_sum = 0

        return obs, reward, terminated, truncated, info

    def _save_csvs(self):
        pd.DataFrame(self.episode_data).to_csv(CSV_EPISODES, index=False)
        pd.DataFrame(self.step_data).to_csv(CSV_STEPS, index=False)

    def reset(self, seed=None, options=None):
        self.current_episode += 1
        obs, info = super().reset(seed=seed, options=options)
        self.in_lanes = []
        self.prev_queue = 0
        self.episode_steps = 0
        self.episode_queue_sum = 0
        print(f"\n--- Episodio {self.current_episode} iniciado ---")
        return obs, info

    def close(self):
        super().close()
        self._save_csvs()
        print(f"CSVs guardados:\n  → {CSV_EPISODES}\n  → {CSV_STEPS}")

# === CREAR ENTORNO ===
env = CustomSumoEnv(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    single_agent=True,
    use_gui=False,
    out_csv_name=None,
    num_seconds=2000,
    delta_time=10,
    min_green=10,
    max_green=60,
    yellow_time=3,
)
env = DummyVecEnv([lambda: env])

# === CARGAR O CREAR MODELO ===
if os.path.exists(LATEST_MODEL):
    print(f"Cargando modelo anterior: {LATEST_MODEL}")
    model = PPO.load(LATEST_MODEL, env=env)
    print(f"Reanudando desde timestep: {model.num_timesteps}")
else:
    print("Creando modelo nuevo desde cero...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        seed=42
    )

# === ENTRENAR ===
print("\n" + "="*60)
print(f"ENTRENAMIENTO INICIADO ({TIMESTAMP})")
print("="*60)

model.learn(total_timesteps=500_000, reset_num_timesteps=False)

# === GUARDAR ===
model.save(MODEL_PATH)
model.save(LATEST_MODEL)  # Siempre actualiza el "último"
print(f"Modelo guardado:\n  → {MODEL_PATH}\n  → {LATEST_MODEL}")

# === GRAFICAR ===
try:
    df = pd.read_csv(CSV_EPISODES)
    print("\nÚLTIMOS 5 EPISODIOS:")
    print(df[['episode', 'reward', 'waiting_time']].tail())

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['episode'], df['reward'], 'o-', color='green', linewidth=2, markersize=4)
    plt.title("Reward Total por Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df['episode'], df['waiting_time'], 's-', color='red', linewidth=2, markersize=4)
    plt.title("Cola Promedio (Waiting Time)")
    plt.xlabel("Episodio")
    plt.ylabel("Autos esperando")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Progreso del Entrenamiento - {TIMESTAMP}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("No se pudo graficar (CSV vacío):", e)

env.close()
print("\nENTRENAMIENTO COMPLETADO")