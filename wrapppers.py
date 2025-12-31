import gymnasium as gym
import numpy as np
from gymnasium import Env


class HumanoidPDWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, kp=10.0, kd=0.1):
        super().__init__(env)
        self.kp = kp  # Stiffness (Proportional gain)
        self.kd = kd  # Damping (Derivative gain)

        # Lazy initialization - capture joint limits after first action call
        self._joint_limits_low = None
        self._joint_limits_high = None

    def action(self, action):
        """
        1. Receive 'Target Angle' from Agent
        2. Calculate necessary Torque using PD formula
        3. Send Torque to Physics Engine
        """

        # Access MuJoCo data
        data = self.env.unwrapped.data  # type: ignore
        model = self.env.unwrapped.model  # type: ignore

        # Lazy initialization - capture joint limits from MuJoCo model
        # This happens on the first action call, right after env.reset()
        if self._joint_limits_low is None:
            # Get joint limits for actuated joints (skip free joint at index 0)
            # jnt_range gives us [low, high] for each joint
            joint_ranges = model.jnt_range[1:]  # Skip free joint (root)
            self._joint_limits_low = joint_ranges[:, 0].copy()
            self._joint_limits_high = joint_ranges[:, 1].copy()

        # --- Step 1: Interpret the Agent's Action ---
        # Map action from [-1, 1] to actual joint limits [low, high]
        # action = -1 -> joint_low, action = 0 -> middle, action = 1 -> joint_high
        target_q = self._joint_limits_low + (action + 1.0) * 0.5 * (
            self._joint_limits_high - self._joint_limits_low
        )

        # --- Step 2: Get Current Physics State (q and q_dot) ---
        # Note the indices! We skip the root body (first 7 pos, first 6 vel)
        current_q = data.qpos[7:]
        current_qdot = data.qvel[6:]

        # --- Step 3: The PD Equation ---
        # Torque = Kp * (Target - Current) - Kd * (Velocity)
        # This is the "Spring - Damper" logic
        error = target_q - current_q
        torque = (self.kp * error) - (self.kd * current_qdot)

        # --- Step 4: Safety Clipping ---
        # Ensure we don't send values larger than the motors can physically handle
        lb, ub = self.env.action_space.low, self.env.action_space.high  # type: ignore
        torque = np.clip(torque, lb, ub)

        return torque
