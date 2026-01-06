# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import SEAG_2_URDFRoughEnvCfg


@configclass
class SEAG_2_URDFFlatEnvCfg(SEAG_2_URDFRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.base_height_l2.weight = -20
        self.rewards.base_height_l2.params["target_height"] = 0.80
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.flat_orientation_l2.weight = -5
        self.rewards.joint_torques_l2.weight = -2.0e-4 # -2.0e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]

        self.rewards.joint_power.weight = -2.e-3
        self.rewards.joint_power.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]

        self.rewards.lin_vel_z_l2.weight = -2
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.joint_vel_l2.weight = -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]
        self.rewards.joint_acc_l2.weight = -1.e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]
        self.rewards.body_lin_acc_l2.weight = -1.0e-4

        self.rewards.action_rate_l2.weight = -1

        self.rewards.joint_pos_limits.weight = -1
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = [".*_hip_roll.*", ".*_hip_yaw.*", ".*_hip_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*",".*_ankle_roll.*"]
        
        self.rewards.applied_torque_limits.weight = 0.0

        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 1.5


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "SEAG_2_URDFFlatEnvCfg":
            self.disable_zero_weight_rewards()


    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=20.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.15,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
    #     },
    # )