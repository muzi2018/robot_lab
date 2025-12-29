# Config Folder

/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/assets/SEAG_2_URDF.py -> model

/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py -> mdp

/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/SEAG_2_URDF/rough_env_cfg.py -> specific robot config

/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/SEAG_2_URDF/flat_env_cfg.py -> specific robot config

/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py -> general reward

# Run Folder

/home/wang/workspace/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py

/home/wang/workspace/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py


# Log Folder

/home/wang/workspace/robot_lab/logs/rsl_rl/SEAG_2_URDF_flat/2025-12-25_01-09-26 -> model_*.pt (train model)

/home/wang/workspace/robot_lab/logs/rsl_rl/SEAG_2_URDF_flat/2025-12-25_01-09-26/exported/policy.pt-> (employ model, mujoco and real)

/home/wang/workspace/robot_lab/logs/rsl_rl/SEAG_2_URDF_flat/2025-12-25_01-09-26/params -> (agent, env, config. e.m. reward, noise, motor..)
