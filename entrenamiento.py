# entrenamiento.py
# Entrenamiento optimizado + TIEMPO ESTIMADO Y REAL
# 2000 timesteps → ~1 min | 10000 timesteps → ~4 min

import os
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time

# === FECHA Y HORA ACTUAL ===
ahora = datetime.now()
fecha_hora_inicio = ahora.strftime("%Y-%m-%d %H:%M:%S")

# === CONFIGURACIÓN ===
NET_FILE = "trigal.net.xml"
ROUTE_FILE = "trigal_flujo_bajo.rou.xml"
MODEL_PATH = "trigal_model.zip"
OUTPUT_DIR = "outputs"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 500_000        
SIM_SECONDS = 3600      

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("INICIANDO ENTRENAMIENTO OPTIMIZADO")
print(f"Simulación: {SIM_SECONDS}s | Timesteps: {TIMESTEPS:,}")
print(f"Inicio: {fecha_hora_inicio}")
print("-" * 70)

# === CALLBACK CON TIEMPO Y PROGRESO ===
class ProgressLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.data = []
        self.start_time = None
        self.last_print = 0
        self.timesteps_per_sec = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("Entrenamiento INICIADO")
        print("Estimando tiempo... (primeros 1000 timesteps)")

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            elapsed = time.time() - self.start_time
            self.timesteps_per_sec = self.num_timesteps / elapsed

            # Estimación al inicio
            if self.num_timesteps == 1000:
                estimated_min = (self.total_timesteps / self.timesteps_per_sec) / 60
                print(f"Velocidad: {self.timesteps_per_sec:.1f} timesteps/s")
                print(f"Tiempo estimado: ~{estimated_min:.1f} minutos")

            # Progreso cada 10%
            progress = self.num_timesteps / self.total_timesteps
            if progress >= 0.1 and abs(progress - self.last_print) >= 0.1:
                self.last_print = progress
                bar = "█" * int(20 * progress) + "░" * (20 - int(20 * progress))
                print(f"Progreso: [{bar}] {progress:.1%} ({self.num_timesteps:,}/{self.total_timesteps:,})")

            # Registro
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

        return True

    def save_and_plot(self):
        if not self.data:
            return
        
        df = pd.DataFrame(self.data)
        df.to_csv(LOG_CSV, index=False)
        print(f"\nCSV guardado: {LOG_CSV}")

        # Gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(df['timestep'], df['reward'], 'g-', linewidth=2, alpha=0.8)
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
    #out_csv_name=os.path.join(OUTPUT_DIR, "trigal_train"),
    use_gui=False,
    num_seconds=SIM_SECONDS,
    single_agent=True,
    min_green=10,
    delta_time=10
)

# === MODELO ===
callback = ProgressLoggerCallback(total_timesteps=TIMESTEPS)
if os.path.exists(MODEL_PATH):
    print(f"CARGANDO MODELO: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("CREANDO NUEVO MODELO")
    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)

# === ENTRENAR ===
print(f"\nINICIANDO ENTRENAMIENTO... [{fecha_hora_inicio}]")
start_time = time.time()
model.learn(total_timesteps=TIMESTEPS, callback=callback)
end_time = time.time()

# === TIEMPO REAL TRANSCURRIDO ===
elapsed_time = end_time - start_time
elapsed_min = elapsed_time / 60
final_speed = TIMESTEPS / elapsed_time

# === RESUMEN FINAL ===
print("\n" + "=" * 70)
print("ENTRENAMIENTO TERMINADO")
print(f"Inicio:     {fecha_hora_inicio}")
print(f"Fin:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duración:   {elapsed_min:.2f} minutos ({elapsed_time:.1f} segundos)")
print(f"Velocidad:  {final_speed:.1f} timesteps/s")
print(f"Timesteps:  {TIMESTEPS:,}")
print(f"Episodios:  ~{TIMESTEPS / (SIM_SECONDS / 10):.0f} (1 paso = 10s)")
print("-" * 70)

# === GUARDAR ===
model.save(MODEL_PATH)
print(f"Modelo guardado: {MODEL_PATH}")
callback.save_and_plot()

env.close()
print(f"\n¡ENTRENAMIENTO COMPLETADO A LAS {datetime.now().strftime('%H:%M:%S')}!")
print("=" * 70)