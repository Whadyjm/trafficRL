# despliegue.py
import sumo_rl
from stable_baselines3 import PPO
import os

os.makedirs("outputs", exist_ok=True)

env = sumo_rl.SumoEnvironment(
    net_file="trigal.net.xml",
    route_file="trigal_flujo_bajo.rou.xml",
    out_csv_name="outputs/trigal_test",
    use_gui=True,          # Activa la GUI para ver el semáforo inteligente
    num_seconds=3600,      # 1 hora de simulación
    single_agent=True,
)

model = PPO.load("trigal_model")
 
obs, _ = env.reset()
done = False

print("Iniciando simulación con control RL...")
while not done:
    action, _ = model.predict(obs, deterministic=True)  # Usa deterministic=True para mejor control
    obs, reward, done, truncated, info = env.step(action)

env.close()
print("Simulación terminada. Revisa 'outputs/trigal_test*.csv' para resultados.")