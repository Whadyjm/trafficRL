import traci
import time

# === CONFIGURACIÓN DE SUMO ===
sumoBinary = "sumo-gui"
sumoConfig = "single-intersection.sumocfg"
sumoCmd = [
    sumoBinary,
    "-c", sumoConfig,
    "--start",
    "--quit-on-end",
    "--no-warnings", "true"  # Opcional: menos spam
]

# === INICIAR CONEXIÓN CON SUMO ===
traci.start(sumoCmd, port=8873)
print("Simulación iniciada. Controlando semáforo 't'...")

step = 0
max_steps = 1000

# === IDs reales de tu red ===
TLS_ID = "t"
NORTH_LANES = ["n_t_0", "n_t_1"]
WEST_LANES = ["w_t_0", "w_t_1"]

# === Fases del semáforo ===
PHASES = {
    'green_north': "GGrr",  # Norte-Sur: Verde | Oeste-Este: Rojo
    'yellow_north': "yyrr", # Transición desde Norte
    'green_west': "rrGG",   # Oeste-Este: Verde | Norte-Sur: Rojo
    'yellow_west': "rryy"   # Transición desde Oeste
}

# Umbral para cambiar (evita cambios frecuentes)
THRESHOLD = 5.0  # Ajusta: más alto = menos cambios

# Factor para equilibrar espera (espera en seg se divide por esto)
WAIT_FACTOR = 10.0  # Ej: cada 10 seg de espera cuenta como 1 "auto extra"

# === BUCLE PRINCIPAL ===
try:
    # Inicia con verde en Norte por defecto
    current_phase = 'green_north'
    traci.trafficlight.setRedYellowGreenState(TLS_ID, PHASES[current_phase])
    
    while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # --- 1. Obtener estado: autos y tiempo de espera total ---
        north_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in NORTH_LANES)
        west_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in WEST_LANES)
        
        north_waiting = sum(traci.lane.getWaitingTime(lane) for lane in NORTH_LANES)
        west_waiting = sum(traci.lane.getWaitingTime(lane) for lane in WEST_LANES)
        
        # --- 2. Calcular prioridad (autos + espera ponderada) ---
        priority_north = north_vehicles + (north_waiting / WAIT_FACTOR)
        priority_west = west_vehicles + (west_waiting / WAIT_FACTOR)
        
        print(f"Paso {step:4d} | Norte: {north_vehicles:2d} autos, {north_waiting:.1f} seg espera | "
              f"Oeste: {west_vehicles:2d} autos, {west_waiting:.1f} seg espera | "
              f"Prioridad N: {priority_north:.1f}, O: {priority_west:.1f}", end=" | ")

        # --- 3. Lógica de control mejorada ---
        if 'north' in current_phase:
            if priority_west > priority_north + THRESHOLD:
                # Cambiar a Oeste: amarillo Norte (3 seg), luego verde Oeste
                print("→ Cambiando a Oeste-Este (amarillo primero)")
                traci.trafficlight.setRedYellowGreenState(TLS_ID, PHASES['yellow_north'])
                for _ in range(3):  # 3 segundos de amarillo
                    traci.simulationStep()
                    step += 1
                traci.trafficlight.setRedYellowGreenState(TLS_ID, PHASES['green_west'])
                current_phase = 'green_west'
            else:
                print("→ Manteniendo Norte-Sur")
                
        elif 'west' in current_phase:
            if priority_north > priority_west + THRESHOLD:
                # Cambiar a Norte: amarillo Oeste (3 seg), luego verde Norte
                print("→ Cambiando a Norte-Sur (amarillo primero)")
                traci.trafficlight.setRedYellowGreenState(TLS_ID, PHASES['yellow_west'])
                for _ in range(3):  # 3 segundos de amarillo
                    traci.simulationStep()
                    step += 1
                traci.trafficlight.setRedYellowGreenState(TLS_ID, PHASES['green_north'])
                current_phase = 'green_north'
            else:
                print("→ Manteniendo Oeste-Este")
                
        time.sleep(0.1)

except traci.TraCIException as e:
    print(f"\nError de TraCI: {e}")
finally:
    traci.close()
    print("\nSimulación terminada.")