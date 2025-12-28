# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets.utils.usd_converter import (  # noqa: F401
    spawn_from_lazy_usd,
    urdf_to_usd,
)

# parameter_wwj
SEAG_2_URDF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # for urdf
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path="/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/assets/descriptions/SEAG_2_URDF/urdf/SEAG_2_URDF_bend.urdf",
            merge_joints=True,
            fix_base=False,
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=3.0,
            max_angular_velocity=3.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # ... other configuration parameters ...
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_Joint": -0.0,
            ".*_hip_roll_Joint": -0.0,
            ".*_hip_yaw_Joint": 0.0,
            ".*_knee_pitch_Joint": 0.0,
            ".*_ankle_pitch_Joint": -0.0,
            ".*_ankle_roll_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_Joint",
                ".*_hip_roll_Joint",
                ".*_hip_yaw_Joint",
                ".*_knee_pitch_Joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_Joint": 193,
                ".*_hip_roll_Joint": 99,
                ".*_hip_yaw_Joint": 99,
                ".*_knee_pitch_Joint": 193,
            },
            velocity_limit_sim={
                ".*_hip_pitch_Joint": 122,
                ".*_hip_roll_Joint": 245,
                ".*_hip_yaw_Joint": 146,
                ".*_knee_pitch_Joint": 122,
            },
            stiffness={
                ".*_hip_pitch_Joint": 400.0,
                ".*_hip_roll_Joint": 180.0,
                ".*_hip_yaw_Joint": 200.0,
                ".*_knee_pitch_Joint": 400.0,
            },
            damping={
                ".*_hip_pitch_Joint": 10.0,
                ".*_hip_roll_Joint": 4.0,
                ".*_hip_yaw_Joint": 4.0,
                ".*_knee_pitch_Joint": 10.0,
            },
            armature={
                ".*_hip_pitch_Joint": 0.0646,
                ".*_hip_roll_Joint": 0.013,
                ".*_hip_yaw_Joint": 0.019,
                ".*_knee_pitch_Joint": 0.0646,
            },
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_Joint", ".*_ankle_roll_Joint"],
            effort_limit_sim={
                ".*_ankle_pitch_Joint": 99,
                ".*_ankle_roll_Joint": 99,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_Joint": 146,
                ".*_ankle_roll_Joint": 146,
            },
            stiffness={
                ".*_ankle_pitch_Joint": 160,
                ".*_ankle_roll_Joint": 50,
            },
            damping={
                ".*_ankle_pitch_Joint": 3,
                ".*_ankle_roll_Joint": 1,
            },
            armature={
                ".*_ankle_pitch_Joint": 0.019,
                ".*_ankle_roll_Joint": 0.019,
            },
        ),
    },
)
