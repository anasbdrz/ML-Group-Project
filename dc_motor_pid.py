import numpy as np
import matplotlib.pyplot as plt

# DC motor parameters
J = 0.01     # kg.m^2
b = 0.01      # N.m.s
K = 0.05     # V/rad/s
R = 0.5      # ohm
L = 0.05      # H
# PID Controller class
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

# Simulation parameters
dt = 0.01  # time step (s)
T = 5.0    # total simulation time (s)
steps = int(T / dt)

# Initial motor state
theta = 0.0  # angle (not used here)
omega = 0.0  # angular velocity
i = 0.0      # current
state_history = []

# Target speed
omega_target = 50  # rad/s

# Initialize PID controller
pid = PIDController(Kp=1, Ki=2, Kd=0.05)

# Simulation loop
for step in range(steps):
    t = step * dt
    V = pid.compute(omega_target, omega, dt)

    # DC motor dynamics (Euler method)
    dtheta = omega
    domega = (K * i - b * omega) / J
    di = (V - R * i - K * omega) / L

    theta += dtheta * dt
    omega += domega * dt
    i += di * dt

    state_history.append([t, omega, V])

# Convert to array
state_history = np.array(state_history)

# Plot speed
plt.figure(figsize=(12,6))
plt.plot(state_history[:,0], state_history[:,1], label="Motor Speed ω(t)")
plt.axhline(omega_target, color='r', linestyle='--', label='Target Speed')
plt.title("DC Motor Speed Control using PID")
plt.xlabel("Time (s)")
plt.ylabel("Speed (rad/s)")
plt.grid(True)
plt.legend()
plt.show()
