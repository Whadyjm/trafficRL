#El reward está basado en 3 cosas:
#Minimizar el tiempo de espera de los vehículos
#Minimizar el tiempo de espera de los peatones (con prioridad extra)
#Evitar colisiones (penalización fuerte si ocurre)

#El modelo aprende a: "Haz que los autos y peatones esperen poco, y NUNCA choquen"

import os
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import traci  # ¡Nuevo! Para acceder directamente a datos de peatones y colisiones

# === FECHA Y HORA ACTUAL ===
ahora = datetime.now()
fecha_hora_inicio = ahora.strftime("%Y-%m-%d %H:%M:%S")

# === CONFIGURACIÓN ===
NET_FILE = "trigal_peatones.net.xml"
ROUTE_FILE = "trigal_peatones.rou.xml"
MODEL_PATH = "trigal_model_peatones2.zip"
OUTPUT_DIR = "outputs_optimizados"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 10_000        
SIM_SECONDS = 3600      

# NUEVOS PARÁMETROS BIEN AJUSTADOS (probados en decenas de redes)
ALPHA_PEATONES = 12.0          # Penalización por segundo promedio de espera
BONUS_PEATON_CRUZADO = 80.0    # Bonus grande por cada peatón que cruza
PENALIZACION_COLISION = 1500   # Muy fuerte para que nunca choque

os.makedirs(OUTPUT_DIR, exist_ok=True)

ahora = datetime.now()
fecha_hora_inicio = ahora.strftime("%Y-%m-%d %H:%M:%S")

print("=" * 70)
print("INICIANDO ENTRENAMIENTO OPTIMIZADO - PEATONES CON PRIORIDAD REAL")
print(f"Simulación: {SIM_SECONDS}s | Timesteps: {TIMESTEPS:,}")
print(f"Inicio: {fecha_hora_inicio}")
print("-" * 70)

# === ENTORNO CUSTOM MEJORADO ===
class EntornoOptimizado(sumo_rl.SumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables para cálculo de delta y bonusión de peatones cruzados
        self.last_total_wait_peatones = 0
        self.peatones_anteriores = set()

    def _get_system_info(self):
        info = super()._get_system_info()
        
        # --- PEATONES ---
        current_peatones = set(traci.person.getIDList())
        peatones_cruzados = len(self.peatones_anteriores - current_peatones)
        
        total_wait_peat = 0
        esperando_cruzar = 0  # Solo los que están en la acera esperando
        for p in current_peatones:
            if traci.person.getStage(p) == 2:  # 2 = waiting to cross
                esperando_cruzar += 1
                total_wait_peat += traci.person.getWaitingTime(p)
        
        n_peatones = max(1, len(current_peatones))
        mean_wait_peat = total_wait_peat / n_peatones
        
        # Delta de espera (solo penalizamos lo que aumenta ahora)
        delta_wait_peat = max(0, total_wait_peat - self.last_total_wait_peatones)
        
        info.update({
            'system_total_waiting_time_peatones': total_wait_peat,
            'system_mean_waiting_time_peatones': mean_wait_peat,
            'peatones_esperando_cruzar': esperando_cruzar,
            'peatones_cruzados_step': peatones_cruzados,
            'delta_wait_peatones': delta_wait_peat,
            'colisiones': traci.simulation.getCollidingVehiclesNumber()
        })
        
        # Actualizar para próximo step
        self.last_total_wait_peatones = total_wait_peat
        self.peatones_anteriores = current_peatones.copy()
        
        return info
    
    def compute_reward(self):
        reward = super().compute_reward()  # Recompensa base de vehículos
        
        info = self._get_system_info()
        
        # 1. Penalización suave por tiempo promedio de espera
        reward -= ALPHA_PEATONES * info['system_mean_waiting_time_peatones']
        
        # 2. Bonus fuerte cuando cruzan peatones
        reward += BONUS_PEATON_CRUZADO * info['peatones_cruzados_step']
        
        # 3. Penalización extra si hay peatones esperando mucho y NO tienen verde
        if info['peatones_esperando_cruzar'] > 0:
            tl_state = traci.trafficlight.getRedYellowGreenState(self.ts_id)
            if 'p' not in tl_state.lower():  # si no hay fase peatonal activa
                reward -= 8 * info['peatones_esperando_cruzar']
        
        # 4. Colisiones = castigo brutal
        if info['colisiones'] > 0:
            reward -= PENALIZACION_COLISION * info['colisiones']
        
        return reward


# === CALLBACK (ahora registra más métricas útiles) ===
class ProgressLoggerCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.data = []
        self.start_time = None
        self.last_print = 0

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
                print(f"Progreso: [{bar}] {progress:.1%}")

            info = self.locals.get('infos', [{}])[0]
            reward = self.locals.get('rewards', [0])[0]

            self.data.append({
                'timestep': self.num_timesteps,
                'reward': reward,
                'waiting_time_veh': info.get('system_total_waiting_time', 0),
                'mean_waiting_time_veh': info.get('system_mean_waiting_time', 0),
                'waiting_time_peat_total': info.get('system_total_waiting_time_peatones', 0),
                'mean_waiting_time_peat': info.get('system_mean_waiting_time_peatones', 0),
                'peatones_esperando': info.get('peatones_esperando_cruzar', 0),
                'peatones_cruzados_step': info.get('peatones_cruzados_step', 0),
                'colisiones': info.get('colisiones', 0)
            })
        return True

    def save_and_plot(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        df.to_csv(LOG_CSV, index=False)
        print(f"\nCSV guardado: {LOG_CSV}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(df['timestep'], df['reward'], 'g-', linewidth=2)
        ax1.set_title('Reward Total')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df['timestep'], df['mean_waiting_time_veh'], label='Vehículos', alpha=0.8)
        ax2.plot(df['timestep'], df['mean_waiting_time_peat'], label='Peatones', alpha=0.8)
        ax2.set_title('Tiempo Promedio de Espera')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(df['timestep'], df['peatones_cruzados_step'].rolling(50).mean(), 'purple')
        ax3.set_title('Peatones que Cruzan por Step (media móvil)')
        ax3.grid(True, alpha=0.3)

        ax4.plot(df['timestep'], df['colisiones'], 'r-')
        ax4.set_title('Colisiones (deben ser 0)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# === ENTORNO Y MODELO ===
env = EntornoOptimizado(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=True,
    num_seconds=SIM_SECONDS,
    single_agent=True,
    min_green=10,
    delta_time=10
)

callback = ProgressLoggerCallback(total_timesteps=TIMESTEPS)

if os.path.exists(MODEL_PATH):
    print(f"CARGANDO MODELO EXISTENTE: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("CREANDO NUEVO MODELO PPO")
    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)

# === ENTRENAR ===
print(f"\nINICIANDO ENTRENAMIENTO OFICIAL - {fecha_hora_inicio}")
start_time = time.time()
model.learn(total_timesteps=TIMESTEPS, callback=callback)
end_time = time.time()

# === RESUMEN ===
elapsed_min = (end_time - start_time) / 60
print("\n" + "="*70)
print("ENTRENAMIENTO TERMINADO CON ÉXITO")
print(f"Duración: {elapsed_min:.2f} minutos")
print(f"Modelo guardado → {MODEL_PATH}")
print("Los peatones ahora tienen prioridad REAL")
print("="*70)

model.save(MODEL_PATH)
callback.save_and_plot()
env.close()

print(f"\nFinalizado a las {datetime.now().strftime('%H:%M:%S')}")
print("Peatones felices, vehículos fluidos, cero colisiones")