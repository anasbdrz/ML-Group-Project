import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from dc_motor_env import DCMotorEnv
import numpy as np

# --- PID Controller Class ---
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# --- Shared Parameters ---
omega_target = 50.0  # rad/s
dt = 0.01
T = 10.0
steps = int(T / dt)

# --- DQN Controller Simulation ---
env = DCMotorEnv()
obs, _ = env.reset()
model = DQN.load("dqn_dc_motor")
done = False
time_dqn, speed_dqn = [], []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time_dqn.append(env.t)
    speed_dqn.append(obs[0])

# --- PID Controller Simulation (matches your original script) ---
# DC motor parameters (must match env!)
J = 0.01
b = 0.01
K = 0.05
R = 0.5
L = 0.05

# Initial motor state
theta = 0.0
omega = 0.0
i = 0.0
t = 0.0

# PID controller (your original gains)
pid = PIDController(Kp=1, Ki=2, Kd=0.05)

time_pid = []
speed_pid = []

for step in range(steps):
    V = pid.compute(omega_target, omega, dt)

    # DC motor dynamics (Euler)
    dtheta = omega
    domega = (K * i - b * omega) / J
    di = (V - R * i - K * omega) / L

    theta += dtheta * dt
    omega += domega * dt
    i += di * dt
    t += dt

    time_pid.append(t)
    speed_pid.append(omega)

from sklearn.metrics import mean_squared_error

def compute_metrics(time, speed, target):
    speed = np.array(speed)
    time = np.array(time)
    
    # --- Rise Time ---
    lower_bound = 0.1 * target
    upper_bound = 0.9 * target

    try:
        rise_start = time[np.where(speed >= lower_bound)[0][0]]
        rise_end = time[np.where(speed >= upper_bound)[0][0]]
        rise_time = rise_end - rise_start
    except IndexError:
        rise_time = float('nan')

    # --- Overshoot ---
    max_speed = np.max(speed)
    overshoot = ((max_speed - target) / target) * 100 if max_speed > target else 0.0

    # --- MSE ---
    mse = mean_squared_error([target] * len(speed), speed)

    return rise_time, overshoot, mse

# Compute metrics
metrics_dqn = compute_metrics(time_dqn, speed_dqn, omega_target)
metrics_pid = compute_metrics(time_pid, speed_pid, omega_target)

# Display
print("\n📊 PERFORMANCE METRICS")
print("--------------------------------------------------")
print(f"{'Controller':<15} | {'Rise Time':>10} | {'Overshoot (%)':>15} | {'MSE':>10}")
print("--------------------------------------------------")
print(f"{'DQN':<15} | {metrics_dqn[0]:10.3f} | {metrics_dqn[1]:15.2f} | {metrics_dqn[2]:10.4f}")
print(f"{'PID':<15} | {metrics_pid[0]:10.3f} | {metrics_pid[1]:15.2f} | {metrics_pid[2]:10.4f}")
print("--------------------------------------------------")

# --- Plotting Setup ---
# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(time_dqn, speed_dqn, label='DQN Controller')
plt.plot(time_pid, speed_pid, label='PID Controller (original)')
plt.axhline(omega_target, color='r', linestyle='--', label='Target Speed')
plt.xlabel("Time (s)")
plt.ylabel("Speed (rad/s)")
plt.title("DC Motor Speed: DQN vs PID (Original Script)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



