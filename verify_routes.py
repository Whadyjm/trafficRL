
import os
import traci
import sumo_rl
from gymnasium import spaces
import numpy as np

# Define minimal observation class needed for environment initialization
class AmbulanceObservationFunction(sumo_rl.ObservationFunction):
    def __init__(self, ts):
        self.ts = ts
    def __call__(self, ts=None):
        return np.zeros(10, dtype=np.float32) # Dummy observation
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

# Define minimal environment class
class EntornoOptimizado(sumo_rl.SumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    print("Verifying route file...")
    try:
        env = EntornoOptimizado(
            net_file="trigal_peatones.net.xml",
            route_file="trigal_peatones.rou.xml",
            use_gui=False,
            num_seconds=100,
            single_agent=True,
            min_green=10,
            delta_time=10,
            observation_class=AmbulanceObservationFunction
        )
        env.reset()
        print("Environment reset successful.")
        for _ in range(50):
            env.step(0)
        print("Ran 50 steps successfully.")
        env.close()
        print("Verification passed!")
    except Exception as e:
        print(f"Verification failed: {e}")
        exit(1)
