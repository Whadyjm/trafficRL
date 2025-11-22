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
import numpy as np
from gymnasium import spaces

# === FECHA Y HORA ACTUAL ===
ahora = datetime.now()
fecha_hora_inicio = ahora.strftime("%Y-%m-%d %H:%M:%S")

# === CONFIGURACIÓN ===
NET_FILE = "trigal_peatones.net.xml"
ROUTE_FILE = "trigal_peatones.rou.xml"
MODEL_PATH = "trigal_model_ambulancia.zip"
OUTPUT_DIR = "outputs_optimizados"
LOG_CSV = os.path.join(OUTPUT_DIR, "progreso_entrenamiento.csv")
TIMESTEPS = 500_000        
SIM_SECONDS = 3600      

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("INICIANDO ENTRENAMIENTO OPTIMIZADO - PEATONES CON PRIORIDAD REAL")
print(f"Simulación: {SIM_SECONDS}s | Timesteps: {TIMESTEPS:,}")
print(f"Inicio: {fecha_hora_inicio}")
print("-" * 70)

class AmbulanceObservationFunction(sumo_rl.ObservationFunction):
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, ts=None):
        if ts is None:
            ts = self.ts
            
        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
        min_green = [1 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 0]
        density = ts.get_lanes_density()
        queue = ts.get_lanes_queue()
        
        # Detectar ambulancia
        ambulance_approaching = 0
        ambulance_waiting = 0
        
        vehicles = traci.vehicle.getIDList()
        for v in vehicles:
            if traci.vehicle.getTypeID(v) == "ambulancia":
                lane_id = traci.vehicle.getLaneID(v)
                if lane_id in ts.lanes:
                    ambulance_approaching = 1
                    if traci.vehicle.getWaitingTime(v) > 0:
                        ambulance_waiting = 1
        
        observation = np.array(phase_id + min_green + density + queue + [ambulance_approaching, ambulance_waiting], dtype=np.float32)
        return observation

    def observation_space(self):
        ts = self.ts
        dim = ts.num_green_phases + 1 + 2 * len(ts.lanes) + 2
        return spaces.Box(low=0, high=1, shape=(dim,), dtype=np.float32)

class EntornoOptimizado(sumo_rl.SumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peatones_anteriores = set()

    def _get_ambulance_action(self, ts):
        """
        Retorna la acción (índice de fase verde) que favorece a la ambulancia.
        Si no hay ambulancia o no necesita paso, retorna None.
        """
        ambulance_lane = None
        vehicles = traci.vehicle.getIDList()
        for v in vehicles:
            if traci.vehicle.getTypeID(v) == "ambulancia":
                ambulance_lane = traci.vehicle.getLaneID(v)
                break # Asumimos una ambulancia prioritaria a la vez por simplicidad

        if not ambulance_lane:
            return None

        # Verificar si la ambulancia está en un carril controlado por este semáforo
        if ambulance_lane not in ts.lanes:
            return None

        # Mapear carril a índice de enlace del semáforo
        # controlled_links = [ [(lane, via, link), ...], ... ] correspondiente a cada char del state
        controlled_links = traci.trafficlight.getControlledLinks(ts.id)
        target_link_indices = []
        
        for i, links in enumerate(controlled_links):
            for link in links:
                if link[0] == ambulance_lane:
                    target_link_indices.append(i)
        
        if not target_link_indices:
            return None

        # Buscar qué fase verde activa estos enlaces
        best_action = None
        
        for action in range(ts.num_green_phases):
            # Intentamos acceder a la fase verde directamente si existe la propiedad
            try:
                # sumo_rl abstrae las fases verdes. Necesitamos encontrar cuál es.
                # ts.green_phases es una lista de objetos Phase (o dicts)
                phase_state = ts.green_phases[action].state
            except AttributeError:
                continue

            # Verificar si esta fase da paso a la ambulancia
            gives_way = False
            for idx in target_link_indices:
                if idx < len(phase_state):
                    if phase_state[idx].lower() == 'g':
                        gives_way = True
                        break
            
            if gives_way:
                best_action = action
                break # Encontramos una fase que sirve
        
        return best_action

    def step(self, action):
        # Override de acción si hay ambulancia
        if self.single_agent:
            ts = self.traffic_signals[self.ts_ids[0]]
            
            # 1. PRIORIDAD MÁXIMA: AMBULANCIA
            ambulance_action = self._get_ambulance_action(ts)
            if ambulance_action is not None:
                action = ambulance_action
                # HACK: Forzar cambio inmediato ignorando min_green
                # Hacemos creer al sistema que ya pasó el tiempo mínimo
                ts.time_since_last_phase_change = ts.min_green + ts.yellow_time + 1
            
            # 2. PRIORIDAD SECUNDARIA: HORARIO PEATONAL (Si no hay ambulancia)
            else:
                # Ciclo total es 137s según .net.xml
                # Fase Peatonal 1 (Phase 2): Inicio 23s, Duración 15s -> [23, 38)
                # Fase Peatonal 2 (Phase 9): Inicio 122s, Duración 15s -> [122, 137)
                
                # Mapeo de Acciones (Solo fases verdes):
                # Action 0 -> Phase 0
                # Action 1 -> Phase 2 (Peatonal 1)
                # Action 2 -> Phase 3
                # Action 3 -> Phase 5
                # Action 4 -> Phase 7
                # Action 5 -> Phase 9 (Peatonal 2)
                
                sim_time = traci.simulation.getTime()
                cycle_time = sim_time % 137
                
                if 23 <= cycle_time < 38:
                    action = 1 # Forzar Fase Peatonal 1
                elif 122 <= cycle_time < 137:
                    action = 5 # Forzar Fase Peatonal 2
        
        return super().step(action)

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
        ambulance_waiting_time = 0
        ambulance_in_system = 0
        
        for v in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(v)
            if wt > 0:
                veh_esperando += 1
                veh_total_wait += wt
                if wt > veh_max_wait:
                    veh_max_wait = wt
            
            if traci.vehicle.getTypeID(v) == "ambulancia":
                ambulance_in_system = 1
                if wt > 0:
                    ambulance_waiting_time = wt

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

        # Detección de Tiempos Muertos Vehiculares (Luz verde sin autos)
        veh_dead_time = 0
        # Obtenemos los links controlados por el semáforo (índice 0)
        # controlled_links es una lista de listas de tuplas (lane, via, link) para cada índice del estado
        controlled_links = traci.trafficlight.getControlledLinks(self.ts_ids[0])
        
        # Recorremos el estado del semáforo (excluyendo los últimos 4 que son peatones)
        for i, char in enumerate(state[:-4]):
            if char.lower() == 'g': # Si está en verde (mayúscula o minúscula)
                # Verificamos los carriles asociados a este índice
                if i < len(controlled_links):
                    links = controlled_links[i]
                    for link in links:
                        lane_id = link[0] # El primer elemento es el ID del carril de entrada
                        if lane_id:
                            # Contamos vehículos en este carril
                            if traci.lane.getLastStepVehicleNumber(lane_id) == 0:
                                veh_dead_time += 1

        info.update({
            'peatones_infraccion': peatones_infraccion,
            'veh_dead_time': veh_dead_time, # Nueva métrica
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
            'veh_mean_wait': veh_total_wait / max(1, veh_esperando),
            
            # Ambulancia
            'ambulance_in_system': ambulance_in_system,
            'ambulance_waiting_time': ambulance_waiting_time
        })
        return info

    def compute_reward(self):
        info = self._get_system_info()
        reward = 0

        # === 0. AMBULANCIA (PRIORIDAD MÁXIMA) ===
        if info['ambulance_in_system'] > 0:
            # Si hay ambulancia, el objetivo principal es que NO espere
            if info['ambulance_waiting_time'] > 0:
                reward -= 1000 * (info['ambulance_waiting_time'] ** 2) # Penalización masiva
            else:
                reward += 500 # Premio por mantenerla en movimiento

        # === 1. PEATONES (Prioridad Alta pero Equilibrada) ===
        # Aglomeración: Penalización cuadrática
        if info['peatones_esperando'] > 0:
            reward -= 400 * (info['peatones_esperando'] ** 2)
            
        # Frustración Peatonal: Penalización exponencial por tiempo máximo
        if info['peatones_max_wait'] > 0:
            reward -= 15 * (info['peatones_max_wait'] ** 1.8)

        # === 2. VEHÍCULOS (Evitar Frustración Vehicular) ===
        # Aglomeración Vehicular
        if info['veh_esperando'] > 0:
            reward -= 5 * (info['veh_esperando'] ** 2)
            
        # Frustración Vehicular: Penalización para evitar colas eternas
        if info['veh_max_wait'] > 0:
            reward -= 10 * (info['veh_max_wait'])

        # Penalización por Tiempos Muertos Vehiculares (NUEVO)
        if info['veh_dead_time'] > 0:
            reward -= 50 * info['veh_dead_time'] # Penalización por cada carril verde vacío

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
        amb_waiting = info.get('ambulance_waiting_time', 0)
        
        # Imprimir estado actual (usamos \r para sobrescribir la línea)
        amb_str = f" | AMBULANCIA ESPERANDO: {amb_waiting}s" if amb_waiting > 0 else ""
        print(f"\rStep: {self.num_timesteps} | Peatones: {peat_waiting} | Autos: {veh_waiting}{amb_str}   ", end="", flush=True)

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
                'infracciones': info.get('peatones_infraccion', 0),
                'ambulance_waiting': info.get('ambulance_waiting_time', 0)
            })
        return True

    def save_and_plot(self):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        df.to_csv(LOG_CSV, index=False)
        print(f"\nCSV guardado: {LOG_CSV}")

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. Recompensa
        ax1.plot(df['timestep'], df['reward'], label='Recompensa', color='blue', alpha=0.6)
        ax1.set_title('Evolución de la Recompensa')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Reward')
        ax1.grid(True)

        # 2. Tiempos de Espera (Vehículos vs Peatones)
        ax2.plot(df['timestep'], df['waiting_time_veh'], label='Vehículos (Total)', color='red', alpha=0.5)
        ax2.plot(df['timestep'], df['waiting_time_peat_total'], label='Peatones (Total)', color='green', alpha=0.5)
        ax2.set_title('Tiempo Total de Espera (Global)')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Segundos')
        ax2.legend()
        ax2.grid(True)

        # 3. Peatones Esperando vs Cruzados
        ax3.plot(df['timestep'], df['peatones_esperando'], label='Esperando', color='orange', alpha=0.6)
        ax3.plot(df['timestep'], df['peatones_cruzados_step'].cumsum(), label='Cruzados (Acumulado)', color='purple')
        ax3.set_title('Flujo de Peatones')
        ax3.set_xlabel('Timesteps')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Infracciones y Colisiones
        ax4.plot(df['timestep'], df['infracciones'].cumsum(), label='Infracciones (Acum)', color='brown')
        ax4.plot(df['timestep'], df['colisiones'].cumsum(), label='Colisiones (Acum)', color='black')
        ax4.set_title('Seguridad e Infracciones')
        ax4.set_xlabel('Timesteps')
        ax4.legend()
        ax4.grid(True)

        # 5. Tiempo de Espera Vehicular (Detallado)
        ax5.plot(df['timestep'], df['mean_waiting_time_veh'], label='Promedio por Vehículo', color='teal')
        # ax5.plot(df['timestep'], df['max_waiting_time_veh'], label='Máximo (Peor caso)', color='red', linestyle='--') # Si tuviéramos max guardado
        ax5.set_title('Tiempo de Espera Vehicular Promedio')
        ax5.set_xlabel('Timesteps')
        ax5.set_ylabel('Segundos')
        ax5.legend()
        ax5.grid(True)
        
        # 6. Ambulancia (Si hay datos)
        if df['ambulance_waiting'].sum() > 0:
             ax6.plot(df['timestep'], df['ambulance_waiting'], label='Tiempo Espera Ambulancia', color='red')
             ax6.set_title('Prioridad Ambulancia')
             ax6.set_xlabel('Timesteps')
             ax6.set_ylabel('Segundos')
             ax6.legend()
             ax6.grid(True)
        else:
            ax6.text(0.5, 0.5, 'Sin datos de ambulancia aún', horizontalalignment='center', verticalalignment='center')
            ax6.set_title('Prioridad Ambulancia')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "progreso_entrenamiento.png"))
        print("Gráficas guardadas.")

# === MAIN ===
if __name__ == "__main__":
    # Configurar entorno
    env = EntornoOptimizado(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        #out_csv_name=os.path.join(OUTPUT_DIR, "trigal_train"),
        use_gui=True,
        num_seconds=SIM_SECONDS,
        single_agent=True,
        min_green=10,
        delta_time=10,
        observation_class=AmbulanceObservationFunction
    )

    # Modelo PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01
    )

    # Callback
    logger = ProgressLoggerCallback(total_timesteps=TIMESTEPS)

    # Entrenar
    try:
        model.learn(total_timesteps=TIMESTEPS, callback=logger)
        model.save(MODEL_PATH)
        print(f"\nModelo guardado en {MODEL_PATH}")
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por usuario. Guardando modelo parcial...")
        model.save(MODEL_PATH + "_interrumpido")
    finally:
        logger.save_and_plot()
        env.close()