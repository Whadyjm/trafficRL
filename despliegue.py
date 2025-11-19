# despliegue_corregido.py
import os
from stable_baselines3 import PPO
import sumo_rl
import traci

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
            'veh_mean_wait': veh_total_wait / max(1, veh_esperando)
        })
        return info

    def compute_reward(self):
        info = self._get_system_info()
        reward = 0

        # === 1. PEATONES (Prioridad Alta pero Equilibrada) ===
        # Aglomeración: Penalización cuadrática
        if info['peatones_esperando'] > 0:
            reward -= 40 * (info['peatones_esperando'] ** 2)
            
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
                reward += 50
            else:
                # Pequeña penalización si está en verde para peatones sin nadie (ineficiente)
                reward -= 10

        # === 4. SEGURIDAD (Crítico) ===
        if info['colisiones'] > 0:
            reward -= 2000
            
        return reward


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
    delta_time=10
)

model = PPO.load("trigal_model_peatones.zip", env=env)  # ¡OJO! Pásale el env aquí también

obs, _ = env.reset()
done = False

print("Iniciando simulación con modelo RL (usando entorno correcto)...")
step = 0
while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    step += 1
    if step % 50 == 0:
        print(f"Step {step} - Fase actual: {info.get('step')}s")

env.close()
print("Simulación terminada correctamente. Resultados en outputs/trigal_test_rl*.csv")