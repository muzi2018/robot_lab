# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.utils.usd_converter import (  # noqa: F401
    mjcf_to_usd,
    spawn_from_lazy_usd,
    urdf_to_usd,
    xacro_to_usd,
)


SEAG_2_URDF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # for urdf
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"/home/wang/workspace/robot_lab/source/robot_lab/robot_lab/assets/descriptions/SEAG_2_URDF/urdf/SEAG_2_URDF_bend.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0 # parameter_wwj
        ),
        # # for xacro
        # func=spawn_from_lazy_usd,
        # usd_path=xacro_to_usd(  # type: ignore
        #     file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/your_robot/your_robot.xacro",
        #     merge_joints=True,
        #     fix_base=False,
        # ),
        # # for mjcf
        # func=spawn_from_lazy_usd,
        # usd_path=mjcf_to_usd(  # type: ignore
        #     file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/your_robot/your_robot.xml",
        #     import_sites=True,
        #     fix_base=False,
        # ),
        # ... other configuration parameters ...
    ),
    # ... other configuration parameters ...
    init_state=ArticulationCfg.InitialStateCfg( #parameter_wwj
        pos=(0.0, 0.0, 0.8),
        joint_pos={ # 12 DOF
            ".*_hip_pitch_Joint": -0.4,
            ".*_hip_roll_Joint": -0.0,
            ".*_hip_yaw_Joint": 0.0,
            ".*_knee_pitch_Joint": 0.8,
            ".*_ankle_pitch_Joint": -0.4,
            ".*_ankle_roll_Joint": 0.0,
            
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_Joint",
                ".*_hip_roll_Joint",
                ".*_hip_yaw_Joint",
                ".*_knee_pitch_Joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_pitch_Joint": 200.0,
                ".*_hip_roll_Joint": 150.0,
                ".*_hip_yaw_Joint": 150.0,
                ".*_knee_pitch_Joint": 200.0,
            },
            damping={
                ".*_hip_pitch_Joint": 5.0,
                ".*_hip_roll_Joint": 5.0,
                ".*_hip_yaw_Joint": 5.0,
                ".*_knee_pitch_Joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_pitch_Joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_Joint", ".*_ankle_roll_Joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

