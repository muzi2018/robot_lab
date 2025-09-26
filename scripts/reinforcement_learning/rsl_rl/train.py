# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# TensorBoard arguments
parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
parser.add_argument("--tb_port", type=int, default=6006, help="TensorBoard port.")
parser.add_argument("--tb_launch", action="store_true", default=False, help="Auto-launch TensorBoard server.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import threading
import subprocess
import time
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("[WARNING] TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class TensorBoardManager:
    """Manages TensorBoard logging and server."""
    
    def __init__(self, log_dir: str, port: int = 6006, auto_launch: bool = False):
        self.log_dir = log_dir
        self.port = port
        self.auto_launch = auto_launch
        self.writer = None
        self.tb_process = None
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
            print(f"[INFO] TensorBoard logging to: {log_dir}")
            
            if self.auto_launch:
                self.launch_tensorboard()
    
    def launch_tensorboard(self):
        """Launch TensorBoard server in background."""
        try:
            # Check if TensorBoard is already running on this port
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            
            if result != 0:  # Port is free
                print(f"[INFO] Launching TensorBoard on port {self.port}")
                self.tb_process = subprocess.Popen([
                    'tensorboard', '--logdir', self.log_dir, 
                    '--port', str(self.port), '--reload_interval', '30'
                ])
                print(f"[INFO] TensorBoard URL: http://localhost:{self.port}")
                
                # Wait a bit for TensorBoard to start
                time.sleep(2)
            else:
                print(f"[INFO] TensorBoard already running on port {self.port}")
                
        except Exception as e:
            print(f"[WARNING] Could not launch TensorBoard: {e}")
    
    def log_scalar(self, tag: str, value, step: int):
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalars."""
        if self.writer:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        """Log image."""
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def flush(self):
        """Flush logs."""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer and server."""
        if self.writer:
            self.writer.close()
        
        if self.tb_process:
            try:
                self.tb_process.terminate()
                print("[INFO] TensorBoard server stopped")
            except:
                pass


class EnhancedOnPolicyRunner(OnPolicyRunner):
    """Extended OnPolicyRunner with TensorBoard integration."""
    
    def __init__(self, env, train_cfg, log_dir, device='cpu', tb_manager=None):
        super().__init__(env, train_cfg, log_dir, device)
        self.tb_manager = tb_manager
        self.training_step = 0
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Enhanced learning with TensorBoard logging."""
        # Store original log method if it exists
        original_log = getattr(self.alg, 'log', None) if hasattr(self, 'alg') else None
        
        # Override algorithm's log method to add TensorBoard logging
        if self.tb_manager and original_log:
            def enhanced_log(locs, width=80, pad=35):
                # Call original logging
                original_log(locs, width, pad)
                
                # Add TensorBoard logging
                if isinstance(locs, dict):
                    for key, value in locs.items():
                        if isinstance(value, (int, float)):
                            self.tb_manager.log_scalar(f'Training/{key}', value, self.training_step)
                        elif isinstance(value, dict):
                            self.tb_manager.log_scalars(f'Training/{key}', value, self.training_step)
                
                self.training_step += 1
                
                # Flush every 10 steps
                if self.training_step % 10 == 0:
                    self.tb_manager.flush()
            
            # Replace the log method
            if hasattr(self.alg, 'log'):
                self.alg.log = enhanced_log
        
        # Call parent learn method
        result = super().learn(num_learning_iterations, init_at_random_ep_len)
        
        return result


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Initialize TensorBoard manager
    tb_manager = None
    if args_cli.tensorboard and TENSORBOARD_AVAILABLE:
        tb_dir = os.path.join(log_dir, "tensorboard")
        tb_manager = TensorBoardManager(tb_dir, args_cli.tb_port, args_cli.tb_launch)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl (enhanced version with TensorBoard)
    if tb_manager:
        runner = EnhancedOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, 
                                       device=agent_cfg.device, tb_manager=tb_manager)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Log hyperparameters to TensorBoard
    if tb_manager:
        # Create a simple dict of key hyperparameters
        hparams = {
            'num_envs': env_cfg.scene.num_envs,
            'max_iterations': agent_cfg.max_iterations,
            'seed': env_cfg.seed,
            'task': args_cli.task or 'unknown'
        }
        
        # Add learning rate if available
        if hasattr(agent_cfg.algorithm, 'learning_rate'):
            hparams['learning_rate'] = agent_cfg.algorithm.learning_rate
        
        # Log as text for now (since hparams logging can be complex)
        hparam_text = '\n'.join([f'{k}: {v}' for k, v in hparams.items()])
        tb_manager.writer.add_text('Hyperparameters', hparam_text, 0)

    try:
        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        
    finally:
        # Clean up
        if tb_manager:
            print("[INFO] Closing TensorBoard...")
            tb_manager.close()
        
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()