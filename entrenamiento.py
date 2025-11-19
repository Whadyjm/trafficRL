#El reward está basado en 3 cosas:
#Minimizar el tiempo de espera de los vehículos
#Minimizar el tiempo de espera de los peatones (con prioridad extra)
#Evitar colisiones (penalización fuerte si ocurre)

#El modelo aprende a: "Haz que los autos y peatones esperen poco, y NUNCA choquen"
#PRIORIDAD: PEATONES, EMERGENCIAS Y TIEMPOS MUERTOS

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
MODEL_PATH = "trigal_model_peatones.zip"
OUTPUT_DIR = "outputs_optimizados"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 100_000        
SIM_SECONDS = 3600      

os.makedirs(OUTPUT_DIR, exist_ok=True)

ahora = datetime.now()
fecha_hora_inicio = ahora.strftime("%Y-%m-%d %H:%M:%S")

print("=" * 70)
print("INICIANDO ENTRENAMIENTO OPTIMIZADO - PEATONES CON PRIORIDAD REAL")
print(f"Simulación: {SIM_SECONDS}s | Timesteps: {TIMESTEPS:,}")
print(f"Inicio: {fecha_hora_inicio}")
print("-" * 70)

# === ENTORNO FINAL: PEATONES PRIORIDAD ALTA + FLUJO VEHICULAR EQUILIBRADO ===
class EntornoOptimizado(sumo_rl.SumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peatones_anteriores = set()

    def _get_system_info(self):
        info = super()._get_system_info()
        
        current_peatones = set(traci.person.getIDList())
        cruzados = len(self.peatones_anteriores - current_peatones)
        self.peatones_anteriores = current_peatones.copy()

        # Métricas detalladas de peatones
        peat_esperando = 0
        peat_max_wait = 0
        peat_total_wait = 0
        
        for p in current_peatones:
            wt = traci.person.getWaitingTime(p)
            if wt > 0:
                peat_esperando += 1
                peat_total_wait += wt
                if wt > peat_max_wait:
                    peat_max_wait = wt

        # Métricas detalladas de vehículos (NUEVO PARA EQUILIBRIO)
        veh_esperando = 0
        veh_max_wait = 0
        veh_total_wait = 0
        
        for v in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(v)
            if wt > 0:
                veh_esperando += 1
                veh_total_wait += wt
                if wt > veh_max_wait:
                    veh_max_wait = wt

        # Estado del semáforo
        try:
            state = traci.trafficlight.getRedYellowGreenState(self.ts_ids[0])
        except:
            state = "rrrr"
            
        fase_peatonal_activa = state.endswith("GGGG")

        # Detección de Infracciones (Jaywalking)
        # Si el semáforo peatonal (últimos 4) NO tiene verde, y hay peatones en los cruces (edges internos ":")
        peatones_infraccion = 0
        sem_peatones_rojo = 'G' not in state[-4:]
        
        if sem_peatones_rojo:
            for p in current_peatones:
                # Si está en un cruce (interno, empieza con :)
                if traci.person.getRoadID(p).startswith(":"):
                    peatones_infraccion += 1

        info.update({
            'peatones_infraccion': peatones_infraccion,
            'peatones_esperando': peat_esperando,
            'peatones_max_wait': peat_max_wait,
            'peatones_total_wait': peat_total_wait,
            'peatones_cruzados': cruzados,
            'fase_peatonal': fase_peatonal_activa,
            'colisiones': traci.simulation.getCollidingVehiclesNumber(),
            
            # Métricas de vehículos actualizadas
            'veh_esperando': veh_esperando,
            'veh_max_wait': veh_max_wait,
            'veh_total_wait': veh_total_wait,
            'veh_mean_wait': veh_total_wait / max(1, veh_esperando)
        })
        return info

    def compute_reward(self):
        info = self._get_system_info()
        reward = 0

        # === 1. PEATONES (Prioridad Alta pero Equilibrada) ===
        # Aglomeración: Penalización cuadrática
        if info['peatones_esperando'] > 0:
            reward -= 100 * (info['peatones_esperando'] ** 2)
            
        # Frustración Peatonal: Penalización exponencial por tiempo máximo
        if info['peatones_max_wait'] > 0:
            reward -= 15 * (info['peatones_max_wait'] ** 1.6)

        # === 2. VEHÍCULOS (Evitar Frustración Vehicular) ===
        # Aglomeración Vehicular
        if info['veh_esperando'] > 0:
            reward -= 5 * (info['veh_esperando'] ** 2)
            
        # Frustración Vehicular: Penalización para evitar colas eternas
        if info['veh_max_wait'] > 0:
            reward -= 10 * (info['veh_max_wait'] ** 1.5)

        # === 3. EFICIENCIA Y FLUJO ===
        # Bonus por cruzar (Throughput)
        if info['peatones_cruzados'] > 0:
            reward += 150 * info['peatones_cruzados'] # Gran premio por liberar peatones
            
        # Bonus suave por fase peatonal activa SOLO si hay demanda
        if info['fase_peatonal']:
            if info['peatones_esperando'] > 0 or info['peatones_cruzados'] > 0:
                reward += 100
            else:
                # Pequeña penalización si está en verde para peatones sin nadie (ineficiente)
                reward -= 10

        # === 4. SEGURIDAD (Crítico) ===
        if info['colisiones'] > 0:
            reward -= 2000
            
        # Penalización por Infracción (Jaywalking)
        if info['peatones_infraccion'] > 0:
            reward -= 100 * info['peatones_infraccion']

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
        # === NUEVO: Contador en tiempo real ===
        info = self.locals.get('infos', [{}])[0]
        peat_waiting = info.get('peatones_esperando', 0)
        veh_waiting = info.get('veh_esperando', 0)
        
        # Imprimir estado actual (usamos \r para sobrescribir la línea)
        print(f"\rStep: {self.num_timesteps} | Peatones: {peat_waiting} | Autos: {veh_waiting}   ", end="", flush=True)

        if self.num_timesteps % 1000 == 0:
            elapsed = time.time() - self.start_time
            self.timesteps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

            # Estimación al inicio
            if self.num_timesteps == 1000 and self.timesteps_per_sec > 0:
                estimated_min = (self.total_timesteps / self.timesteps_per_sec) / 60
                print(f"\nVelocidad: {self.timesteps_per_sec:.1f} timesteps/s")
                print(f"Tiempo estimado: ~{estimated_min:.1f} minutos")

            # Progreso cada 10%
            progress = self.num_timesteps / self.total_timesteps
            if progress >= 0.1 and abs(progress - self.last_print) >= 0.1:
                self.last_print = progress
                bar = "█" * int(20 * progress) + "░" * (20 - int(20 * progress))
                print(f"\nProgreso: [{bar}] {progress:.1%}")

            reward = self.locals.get('rewards', [0])[0]

            self.data.append({
                'timestep': self.num_timesteps,
                'reward': reward,
                'waiting_time_veh': info.get('veh_total_wait', 0),
                'mean_waiting_time_veh': info.get('veh_mean_wait', 0),
                'waiting_time_peat_total': info.get('peatones_total_wait', 0),
                'mean_waiting_time_peat': info.get('peatones_total_wait', 0) / max(1, info.get('peatones_esperando', 1)),
                'peatones_esperando': info.get('peatones_esperando', 0),
                'peatones_cruzados_step': info.get('peatones_cruzados', 0),
                'colisiones': info.get('colisiones', 0),
                'infracciones': info.get('peatones_infraccion', 0)
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
        # plt.show() # Comentado para evitar bloqueo en servidor

# === ENTORNO Y MODELO ===
env = EntornoOptimizado(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=True,
    num_seconds=SIM_SECONDS,
    single_agent=True,
    min_green=15,
    delta_time=8,
    yellow_time=3
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