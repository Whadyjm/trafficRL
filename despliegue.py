# despliegue_corregido.py
import os
from stable_baselines3 import PPO
import sumo_rl
import traci

# === COPIA EXACTAMENTE TU CLASE DE ENTORNO DEL ENTRENAMIENTO ===
class EntornoOptimizado(sumo_rl.SumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_total_wait_peatones = 0
        self.peatones_anteriores = set()

    def _get_system_info(self):
        info = super()._get_system_info()
        
        current_peatones = set(traci.person.getIDList())
        peatones_cruzados = len(self.peatones_anteriores - current_peatones)
        
        total_wait_peat = 0
        esperando_cruzar = 0
        for p in current_peatones:
            if traci.person.getStage(p) == 2:
                esperando_cruzar += 1
                total_wait_peat += traci.person.getWaitingTime(p)
        
        n_peatones = max(1, len(current_peatones))
        mean_wait_peat = total_wait_peat / n_peatones
        
        info.update({
            'system_total_waiting_time_peatones': total_wait_peat,
            'system_mean_waiting_time_peatones': mean_wait_peat,
            'peatones_esperando_cruzar': esperando_cruzar,
            'peatones_cruzados_step': peatones_cruzados,
            'colisiones': traci.simulation.getCollidingVehiclesNumber()
        })
        
        self.last_total_wait_peatones = total_wait_peat
        self.peatones_anteriores = current_peatones.copy()
        
        return info

    # IMPORTANTE: aunque no uses reward en test, compute_reward debe existir
    def compute_reward(self):
        # Puedes dejar uno simple o el mismo del entrenamiento
        reward = super().compute_reward()
        info = self._get_system_info()
        reward -= 12.0 * info['system_mean_waiting_time_peatones']
        reward += 80.0 * info['peatones_cruzados_step']
        if info['colisiones'] > 0:
            reward -= 1500 * info['colisiones']
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

model = PPO.load("trigal_model_peatones2.zip", env=env)  # ¡OJO! Pásale el env aquí también

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