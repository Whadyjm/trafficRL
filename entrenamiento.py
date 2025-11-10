# entrenamiento.py
# Entrenamiento optimizado: 3600 segundos = duración real del tráfico
#2000 timesteps - 1 min
#10000 timesteps - 4 min
#50000 timesteps - 14 min
import os
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# === FECHA Y HORA ACTUAL ===
ahora = datetime.now()
fecha_hora = ahora.strftime("%Y-%m-%d %H:%M:%S")

# === CONFIGURACIÓN ===
NET_FILE = "trigal.net.xml"
ROUTE_FILE = "trigal_flujo_bajo.rou.xml"
MODEL_PATH = "trigal_model.zip"
OUTPUT_DIR = "outputs"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 50000        
SIM_SECONDS = 3600      

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("INICIANDO ENTRENAMIENTO OPTIMIZADO (3600s)")
print(f"Simulación: {SIM_SECONDS} segundos | Timesteps: {TIMESTEPS}")
print("-" * 60)

# === CALLBACK PARA REGISTRO ===
class ProgressLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.data = []

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            info = self.locals.get('infos', [{}])[0]
            reward = self.locals.get('rewards', [0])[0]
            waiting = info.get('system_total_waiting_time', 0)
            mean_wait = info.get('system_mean_waiting_time', 0)

            self.data.append({
                'timestep': self.num_timesteps,
                'reward': reward,
                'waiting_time': waiting,
                'mean_waiting_time': mean_wait
            })
            print(f"Step {self.num_timesteps:5d} | R: {reward:6.2f} | Wait: {waiting:6.0f}s")
        return True

    def save_and_plot(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        df.to_csv(LOG_CSV, index=False)
        print(f"CSV guardado: {LOG_CSV}")

        # Gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        ax1.plot(df['timestep'], df['reward'], 'g-', linewidth=2)
        ax1.set_title('Reward durante el Entrenamiento')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df['timestep'], df['waiting_time'], 'r-', linewidth=2, label='Total')
        ax2.plot(df['timestep'], df['mean_waiting_time'], 'orange', linewidth=1.5, label='Promedio')
        ax2.set_title('Tiempo de Espera en la Intersección')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Tiempo (s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# === ENTORNO ===
env = sumo_rl.SumoEnvironment(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=os.path.join(OUTPUT_DIR, "trigal_train"),
    use_gui=False,
    num_seconds=SIM_SECONDS,    # 3600s = 1 hora real
    single_agent=True,
    min_green=10,
    delta_time=10
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
print(f"\nINICIANDO ENTRENAMIENTO... [{fecha_hora}]")
model.learn(total_timesteps=TIMESTEPS, callback=callback)
print("ENTRENAMIENTO TERMINADO")

# === GUARDAR Y MOSTRAR ===
model.save(MODEL_PATH)
print(f"Modelo guardado: {MODEL_PATH}")
callback.save_and_plot()

env.close()
print(f"\n¡ENTRENAMIENTO COMPLETADO A LAS {datetime.now().strftime('%H:%M:%S')}!")