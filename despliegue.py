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
        esperando = 0
        max_wait = 0
        total_wait = 0
        
        for p in current_peatones:
            wt = traci.person.getWaitingTime(p)
            if wt > 0:
                esperando += 1
                total_wait += wt
                if wt > max_wait:
                    max_wait = wt

        # Estado del semáforo
        try:
            state = traci.trafficlight.getRedYellowGreenState(self.ts_ids[0])
        except:
            state = ""
        fase_peatonal_activa = 'p' in state.lower() or 'P' in state

        info.update({
            'peatones_esperando': esperando,
            'peatones_max_wait': max_wait,
            'peatones_total_wait': total_wait,
            'peatones_cruzados': cruzados,
            'fase_peatonal': fase_peatonal_activa,
            'colisiones': traci.simulation.getCollidingVehiclesNumber(),
            'veh_waiting': info.get('system_total_waiting_time', 0),
            'veh_mean_wait': info.get('system_mean_waiting_time', 0)
        })
        return info

    def compute_reward(self):
        info = self._get_system_info()
        reward = 0

        # 1. EVITAR AGLOMERAMIENTO
        if info['peatones_esperando'] > 0:
            reward -= 50 * (info['peatones_esperando'] ** 2)

        # 2. EVITAR DESESPERACIÓN
        if info['peatones_max_wait'] > 0:
            reward -= 20 * (info['peatones_max_wait'] ** 1.5)

        # 3. PRIORIDAD ABSOLUTA
        if info['fase_peatonal']:
            reward += 200
            if info['peatones_cruzados'] > 0:
                reward += 100 * info['peatones_cruzados']

        # 4. Penalización por colisiones
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