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


URDF_0924_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # for urdf
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"/home/wang/robot_lab/source/robot_lab/robot_lab/assets/descriptions/urdf0924/urdf/urdf0924.urdf",
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.83),
        joint_pos={ # 27 DOF
            ".*_hip_pitch_joint": -0.4,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.8,
            ".*_ankle_pitch_joint": -0.4,
            ".*_ankle_roll_joint": 0.0,
            
            "torso_joint": 0.0,  # torso joints
            ".*_shoulder_pitch_joint": -0.3,
            "right_shoulder_roll_joint": -0.2, #(right -0.2, left 0.2)
            "left_shoulder_roll_joint": 0.2,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.1,
            
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 150.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                "torso_joint": 200.0,
            },
            damping={
                "torso_joint": 5.0,
            },
            armature={
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_knee_joint",".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)

