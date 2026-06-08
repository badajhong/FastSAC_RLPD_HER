import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.assets.rigid_object import RigidObjectCfg
import torch

ASSET_PATH = os.path.dirname(__file__)

class CustomArticulation(Articulation):
    def _create_buffers(self):
        super()._create_buffers()
        self._custom_friction = torch.zeros((self.num_instances,), device=self.device)
        self._custom_damping = torch.zeros((self.num_instances,), device=self.device)
        assert len(self.joint_names) == 1, "DoorArticulation should have exactly one joint."
        self.custom_joint_id = 0
        self.custom_torques = torch.zeros((self.num_instances,), device=self.device)

    def _initialize_impl(self):
        super()._initialize_impl()

        # set joint stiffness and damping to 0
        joint_attrs_zero = torch.zeros((self.num_instances, self.num_joints), device=self.device)
        self.write_joint_stiffness_to_sim(joint_attrs_zero)
        self.write_joint_damping_to_sim(joint_attrs_zero)
        self.write_joint_friction_coefficient_to_sim(joint_attrs_zero)

        # set actuator stiffness and damping to 0
        for actuator in self.actuators.values():
            actuator.stiffness.fill_(0.0)
            actuator.damping.fill_(0.0)

    def write_data_to_sim(self):
        j_vel = self.data.joint_vel[:, self.custom_joint_id]
        j_friction = -torch.sign(j_vel) * (j_vel.abs() > 0.01) * self._custom_friction
        j_damping = -j_vel * self._custom_damping
        self.custom_torques[:] = j_friction + j_damping

        self.set_joint_effort_target(self.custom_torques.unsqueeze(-1), joint_ids=[self.custom_joint_id])
        super().write_data_to_sim()

class WHEELArticulation(Articulation):
    def _create_buffers(self):
        super()._create_buffers()
        self._custom_friction = torch.zeros((self.num_instances,), device=self.device)
        self._custom_damping = torch.zeros((self.num_instances,), device=self.device)
        self.custom_joint_id = 0
        self.custom_torques = torch.zeros((self.num_instances,), device=self.device)

    def _initialize_impl(self):
        super()._initialize_impl()

        # set joint stiffness and damping to 0
        joint_attrs_zero = torch.zeros((self.num_instances, self.num_joints), device=self.device)
        self.write_joint_stiffness_to_sim(joint_attrs_zero)
        self.write_joint_damping_to_sim(joint_attrs_zero)
        self.write_joint_friction_coefficient_to_sim(joint_attrs_zero)

        # set actuator stiffness and damping to 0
        for actuator in self.actuators.values():
            actuator.stiffness.fill_(0.0)
            actuator.damping.fill_(0.0)

    def write_data_to_sim(self):
        j_vel = self.data.joint_vel[:, self.custom_joint_id]
        j_friction = -torch.sign(j_vel) * (j_vel.abs() > 0.01) * self._custom_friction
        j_damping = -j_vel * self._custom_damping
        self.custom_torques[:] = j_friction + j_damping

        self.set_joint_effort_target(self.custom_torques.unsqueeze(-1), joint_ids=[self.custom_joint_id])
        super().write_data_to_sim()

MYBOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/mybox",
    spawn=sim_utils.UsdFileCfg(
        scale=(1.0, 1.0, 1.0),
        usd_path=f"{ASSET_PATH}/objects/mybox/mybox.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=5.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
)

SKATEBOARD_CFG = ArticulationCfg(
    class_type=WHEELArticulation,
    prim_path="{ENV_REGEX_NS}/skateboard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/skateboard/skateboard.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        )
    ),
    actuators={
        "front_left_wheel_joint": IdealPDActuatorCfg(
            joint_names_expr="front_left_wheel_joint",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "front_right_wheel_joint": IdealPDActuatorCfg(
            joint_names_expr="front_right_wheel_joint",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_left_wheel_joint": IdealPDActuatorCfg(
            joint_names_expr="rear_left_wheel_joint",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_right_wheel_joint": IdealPDActuatorCfg(
            joint_names_expr="rear_right_wheel_joint",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
    },
)

CART_CFG = ArticulationCfg(
    class_type=WHEELArticulation,
    prim_path="{ENV_REGEX_NS}/cart",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/cart/cart.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        )
    ),
    actuators={
        "front_left_wheel_joint_0": IdealPDActuatorCfg(
            joint_names_expr="front_left_wheel_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "front_right_wheel_joint_0": IdealPDActuatorCfg(
            joint_names_expr="front_right_wheel_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_left_wheel_joint_0": IdealPDActuatorCfg(
            joint_names_expr="rear_left_wheel_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_right_wheel_joint_0": IdealPDActuatorCfg(
            joint_names_expr="rear_right_wheel_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "front_left_wheel_joint_1": IdealPDActuatorCfg(
            joint_names_expr="front_left_wheel_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "front_right_wheel_joint_1": IdealPDActuatorCfg(
            joint_names_expr="front_right_wheel_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_left_wheel_joint_1": IdealPDActuatorCfg(
            joint_names_expr="rear_left_wheel_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "rear_right_wheel_joint_1": IdealPDActuatorCfg(
            joint_names_expr="rear_right_wheel_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
    },
)


WHEELCHAIR_CFG = ArticulationCfg(
    class_type=WHEELArticulation,
    prim_path="{ENV_REGEX_NS}/wheelchair",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/wheelchair/wheelchair.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            enabled_self_collisions=False,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        )
    ),
    actuators={
        "base_swivel_joint": IdealPDActuatorCfg(
            joint_names_expr="base_swivel_joint",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "caster_0_joint_0": IdealPDActuatorCfg(
            joint_names_expr="caster_0_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "caster_1_joint_0": IdealPDActuatorCfg(
            joint_names_expr="caster_1_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "caster_2_joint_0": IdealPDActuatorCfg(
            joint_names_expr="caster_2_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "caster_3_joint_0": IdealPDActuatorCfg(
            joint_names_expr="caster_3_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "caster_4_joint_0": IdealPDActuatorCfg(
            joint_names_expr="caster_4_joint_0",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "caster_0_joint_1": IdealPDActuatorCfg(
            joint_names_expr="caster_0_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "caster_1_joint_1": IdealPDActuatorCfg(
            joint_names_expr="caster_1_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=2000.0,
        ),
        "caster_2_joint_1": IdealPDActuatorCfg(
            joint_names_expr="caster_2_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "caster_3_joint_1": IdealPDActuatorCfg(
            joint_names_expr="caster_3_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
        "caster_4_joint_1": IdealPDActuatorCfg(
            joint_names_expr="caster_4_joint_1",
            # will be randomized
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=200.0,
        ),
    },
)

STAIRSLOPE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/stairslope",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/stairslope/stairslope.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=200.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
