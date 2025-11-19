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
            state = ""
        fase_peatonal_activa = state.endswith("GGGG")

        info.update({
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