# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for HJE quadruped robot.
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

##
# Configuration
##


HJE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/../robot_lab/assets/descriptions/hje/urdf/hje.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/../robot_lab/assets/descriptions/hje/usd/hje.usd",
            merge_joints=True,
            fix_base=False,
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*_HIP_JOINT": 0.0,
            ".*_THIGH_JOINT": 0.8,
            ".*_CALF_JOINT": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*_HIP_JOINT"],
            effort_limit=106.0,
            saturation_effort=106.0,
            velocity_limit=23.34,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "thighs_calfs": DCMotorCfg(
            joint_names_expr=[".*_THIGH_JOINT", ".*_CALF_JOINT"],
            effort_limit=150.0,
            saturation_effort=150.0,
            velocity_limit=14.65,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
