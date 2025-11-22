# despliegue_corregido.py
import os
from stable_baselines3 import PPO
import sumo_rl
import traci
import numpy as np
from gymnasium import spaces

# === OBSERVATION CLASS (DEBE SER IDÉNTICA AL ENTRENAMIENTO) ===
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

# === COPIA EXACTAMENTE TU CLASE DE ENTORNO DEL ENTRENAMIENTO ===
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
        # El reward no se usa en despliegue, pero la función debe existir
        return 0


# === DESPLIEGUE CORRECTO ===
os.makedirs("outputs", exist_ok=True)

# Usa EXACTAMENTE tu clase personalizada
env = EntornoOptimizado(
    net_file="trigal_peatones.net.xml",
    route_file="trigal_peatones.rou.xml",
    out_csv_name="outputs/trigal_test_rl",
    use_gui=True,
    num_seconds=3600,
    single_agent=True,
    min_green=10,
    delta_time=10,
    observation_class=AmbulanceObservationFunction # ¡NUEVO!
)

model_path = "trigal_model_ambulancia.zip"
if not os.path.exists(model_path):
    print(f"¡ALERTA! No se encontró el modelo {model_path}. Asegúrate de haber entrenado primero.")
    # Fallback para pruebas si no existe, aunque fallará al cargar
else:
    print(f"Cargando modelo: {model_path}")

model = PPO.load(model_path, env=env)

obs, _ = env.reset()
done = False

print("Iniciando simulación con modelo RL (usando entorno correcto)...")
step = 0
while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    step += 1
    
    amb_msg = ""
    if info.get('ambulance_in_system', 0) > 0:
        amb_msg = " [AMBULANCIA EN CAMINO!]"
        if info.get('ambulance_waiting_time', 0) > 0:
            amb_msg += " [ESPERANDO!]"
            
    if step % 50 == 0 or amb_msg:
        print(f"Step {step} - Fase actual: {info.get('step')}s{amb_msg}")

env.close()
print("Simulación terminada correctamente. Resultados en outputs/trigal_test_rl*.csv")