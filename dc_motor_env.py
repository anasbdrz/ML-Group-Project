import gymnasium as gym
from gymnasium import spaces

import numpy as np

class DCMotorEnv(gym.Env):
    def __init__(self):
        super(DCMotorEnv, self).__init__()

        # DC motor parameters
        self.J = 0.01
        self.b =  0.01
        self.K = 0.05
        self.R = 0.5
        self.L = 0.05

        self.dt = 0.02
        self.t = 0.0
        self.max_time = 10.0  # maximum simulation time

        # Discrete action space: voltages
        self.voltages = [round(i * 0.5, 1) for i in range(0, 25)]  # 0.0V to 12.0V in 0.5V steps
        # Action space: choose one of the voltages
        self.action_space = spaces.Discrete(len(self.voltages))

        # Observation: [omega] (speed only for now)
        high = np.array([100.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.omega_target = 50.0  # target speed
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.omega = 0.0
        self.i = 0.0
        self.t = 0.0
        self.prev_omega = 0.0  # ⬅️ Initialize for reward calculation
        self.prev_action = 0.0
        return np.array([self.omega], dtype=np.float32), {}


    def step(self, action):
        V = self.voltages[action]

        # DC motor dynamics
        domega_dt = (self.K * self.i - self.b * self.omega) / self.J
        di_dt = (V - self.R * self.i - self.K * self.omega) / self.L

        self.omega += domega_dt * self.dt
        self.i += di_dt * self.dt
        self.t += self.dt

        state = np.array([self.omega], dtype=np.float32)

        # Reward calculation
        # Use previous omega and action for reward calculation
        # Inside your step() function or wherever you compute reward
        error = abs(self.omega_target - self.omega)
        delta_omega = self.omega - self.prev_omega
        delta_action = abs(action - self.prev_action)
        
        # Reward components
        error_penalty = -error  # Encourage minimizing speed error
        stability_penalty = -0.05 * (delta_omega ** 2)  # Penalize large changes in omega
        effort_penalty = -0.05 * (delta_action ** 2)  # Penalize large control effort

        # Combine all components
        reward = error_penalty + stability_penalty + effort_penalty

        # Optional: give bonus for staying close to target
        if error < 0.5:
            reward += 3.0  # bonus for accuracy
        elif error < 1.0:
            reward += 1.0
        
        # Optional: penalize aggressive control near limits
        voltage = self.voltages[action]
        if abs(voltage) > max(self.voltages) * 0.9:
            reward -= 0.5


        # Update previous values
        self.prev_omega = self.omega
        self.prev_action = action


        terminated = self.t >= self.max_time
        truncated = False

        return state, reward, terminated, truncated, {}


    def render(self,  mode='human'):
        print(f"Time: {self.t:.2f}s, Speed: {self.omega:.2f} rad/s")
    
    


