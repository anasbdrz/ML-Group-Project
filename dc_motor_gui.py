import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. DCMotorEnv class
# -------------------------------
class DCMotorEnv(gym.Env):
    def __init__(self):
        super(DCMotorEnv, self).__init__()
        self.J = 0.01
        self.b = 0.01
        self.K = 0.05
        self.R = 0.5
        self.L = 0.05
        self.dt = 0.02  
        self.t = 0.0
        self.max_time = 10.0  
        self.voltages = [round(i * 0.5, 1) for i in range(25)]
        self.action_space = spaces.Discrete(len(self.voltages))
        high = np.array([100.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.omega_target = 50.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.omega = 0.0
        self.i = 0.0
        self.t = 0.0
        self.prev_omega = 0.0
        self.prev_action = 0.0
        return np.array([self.omega], dtype=np.float32), {}

    def step(self, action):
        V = self.voltages[action]
        domega_dt = (self.K * self.i - self.b * self.omega) / self.J
        di_dt = (V - self.R * self.i - self.K * self.omega) / self.L
        self.omega += domega_dt * self.dt
        self.i += di_dt * self.dt
        self.t += self.dt
        state = np.array([self.omega], dtype=np.float32)
        error = abs(self.omega_target - self.omega)
        reward = -error
        terminated = self.t >= self.max_time
        truncated = False
        return state, reward, terminated, truncated, {}

# -------------------------------
# 2. PID Controller
# -------------------------------
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

# -------------------------------
# 3. Load trained DQN
# -------------------------------
model = DQN.load("dqn_dc_motor")

# -------------------------------
# 4. Simulation functions
# -------------------------------
def run_dqn(target_speed):
    env = DCMotorEnv()
    env.omega_target = target_speed
    obs, _ = env.reset()
    time_dqn, speed_dqn = [], []
    steps = int(env.max_time / env.dt)  # ✅ Fixed steps
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        time_dqn.append(env.t)
        speed_dqn.append(obs[0])
    return time_dqn, speed_dqn

def run_pid(target_speed):
    J, b, K, R, L = 0.01, 0.01, 0.05, 0.5, 0.05
    dt, T = 0.02, 10.0
    steps = int(T / dt)
    omega, i, t = 0.0, 0.0, 0.0
    pid = PIDController(1, 2, 0.05)
    time_pid, speed_pid = [], []
    for _ in range(steps):
        V = pid.compute(target_speed, omega, dt)
        domega = (K * i - b * omega) / J
        di = (V - R * i - K * omega) / L
        omega += domega * dt
        i += di * dt
        t += dt
        time_pid.append(t)
        speed_pid.append(omega)
    return time_pid, speed_pid

# -------------------------------
# 5. Performance metrics
# -------------------------------
def compute_metrics(time, speed, target):
    speed = np.array(speed)
    time = np.array(time)

    lower_bound = 0.1 * target
    upper_bound = 0.9 * target

    try:
        rise_start = time[np.where(speed >= lower_bound)[0][0]]
        rise_end = time[np.where(speed >= upper_bound)[0][0]]
        rise_time = rise_end - rise_start
    except IndexError:
        rise_time = float('nan')

    max_speed = np.max(speed)
    overshoot = ((max_speed - target) / target) * 100 if max_speed > target else 0.0

    mse = mean_squared_error([target] * len(speed), speed)

    within_band = np.abs(speed - target) <= 0.05 * target
    settling_time = float('nan')
    if np.any(within_band):
        last_outside_idx = np.where(~within_band)[0]
        if len(last_outside_idx) > 0:
            settling_time = time[last_outside_idx[-1]]
        else:
            settling_time = time[-1]
    return rise_time, overshoot, mse, settling_time

# -------------------------------
# 6. GUI code
# -------------------------------
def run_simulation():
    controller = controller_var.get()
    target = float(target_var.get())

    if controller == "DQN":
        t, s = run_dqn(target)
    else:
        t, s = run_pid(target)

    metrics = compute_metrics(t, s, target)
    rise_time, overshoot, mse, settling_time = metrics

    plt.figure(figsize=(8, 4))
    plt.plot(t, s, label=f"{controller} Output")
    plt.axhline(target, color='r', linestyle='--', label='Target Speed')
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (rad/s)")
    plt.title(f"DC Motor Speed Control - {controller}")
    plt.legend()
    plt.grid(True)
    plt.show()

    msg = (
        f"📊 PERFORMANCE METRICS ({controller})\n"
        f"----------------------------------------\n"
        f"Rise Time: {rise_time:.3f} s\n"
        f"Overshoot: {overshoot:.2f} %\n"
        f"MSE: {mse:.4f}\n"
        f"Settling Time (approx.): {settling_time:.3f} s"
    )
    print(msg)
    messagebox.showinfo("Performance Metrics", msg)

root = tk.Tk()
root.title("DC Motor Control GUI")

tk.Label(root, text="Select Controller:").pack()
controller_var = tk.StringVar(value="DQN")
ttk.Combobox(root, textvariable=controller_var, values=["DQN", "PID"]).pack()

tk.Label(root, text="Target Speed (rad/s):").pack()
target_var = tk.StringVar(value="50")
tk.Entry(root, textvariable=target_var).pack()

tk.Button(root, text="Run Simulation", command=run_simulation).pack()

root.mainloop()
