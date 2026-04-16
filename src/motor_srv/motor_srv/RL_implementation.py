import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Define the ActorCritic class to match the saved checkpoint
class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, activation='elu'):
        super(ActorCritic, self).__init__()
        # Activation function
        if activation == 'elu':
            self.activation = torch.nn.ELU()
       
        # Actor network
        actor_layers = []
        prev_dim = obs_dim
        for dim in actor_hidden_dims:
            actor_layers.append(torch.nn.Linear(prev_dim, dim))
            actor_layers.append(self.activation)
            prev_dim = dim
        actor_layers.append(torch.nn.Linear(prev_dim, action_dim))
        self.actor = torch.nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        prev_dim = obs_dim
        for dim in critic_hidden_dims:
            critic_layers.append(torch.nn.Linear(prev_dim, dim))
            critic_layers.append(self.activation)
            prev_dim = dim
        critic_layers.append(torch.nn.Linear(prev_dim, 1))
        self.critic = torch.nn.Sequential(*critic_layers)

    def forward(self, obs):
        action_logits = self.actor(obs)
        value = self.critic(obs)
        return action_logits, value


class MotorSRVNode(Node):
    def __init__(self):
        super().__init__('RL_implementation')

        self.initial_gravity = None           # 用于记录第一次经过旋转和归一化后的重力方向
        self.gravity_calib_matrix = None      # 校准旋转矩阵

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Imu, '/camera/gyro/sample', self.gyro_callback, sensor_qos)
        self.create_subscription(Imu, '/camera/accel/sample', self.accel_callback, sensor_qos)
        self.create_subscription(Twist, '/motor_commands', self.command_callback, sensor_qos)
        self.create_subscription(Float32MultiArray, '/dynamixel_status', self.dynamixel_callback, sensor_qos)
        self.create_subscription(Float32MultiArray, '/arduino/height', self.height_callback, sensor_qos)

        self.publisher = self.create_publisher(Float32MultiArray, '/robot/motor_commands', 10)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Move tensors to the same device
        self.base_ang_vel = torch.zeros(3, device=self.device)
        self.projected_gravity = torch.zeros(3, device=self.device)
        self.commands = torch.zeros(3, device=self.device)
        self.height = torch.zeros(1, device=self.device)
        self.dof_pos = torch.zeros(8, device=self.device)  # Relative positions
        self.last_dof_pos = torch.zeros(8, device=self.device)  # Store last position
        self.dof_vel = torch.zeros(8, device=self.device)  # Velocity computed from position differences
        self.actions = torch.zeros(8, device=self.device)
        self.initial_gravity = None
        self.last_valid_dof_pos = torch.zeros(8, device=self.device)
        self.last_valid_dof_vel = torch.zeros(8, device=self.device)
        # Flag to track if positions are initialized and non-zero
        self.positions_initialized = False

        # Load the policy model and configuration file
        self.policy = self.load_policy(
            "src/mingsong_turtle_try/src/motor_srv/models/model_700_01m.pt",
            obs_dim=33,  # Observation dimension (34 for height changable model, 33 for height non-changable model.)
            action_dim=8,  # Action dimension
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )

        self.default_dof_pos = torch.tensor(
            [1.5, -1.3, 1.3, -1.5, 0.15, -0.15, 0.15, -0.15], dtype=torch.float32
        ).to(self.device)

        self.create_timer(0.1, self.compute_and_publish)

    def load_policy(self, checkpoint_path, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, activation):
        """Load only the actor (student) policy network from a checkpoint file."""
        # Instantiate the full ActorCritic network (we only need the actor part later)
        policy_net = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        ).to(self.device)
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        # Filter out only the actor parameters.
        # This keeps only keys that start with "actor." and removes the prefix so they match the actor's own keys.
        actor_state_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        # Load the filtered state dict into the actor network
        policy_net.actor.load_state_dict(actor_state_dict, strict=False)
        policy_net.actor.eval()  # Set the actor network to evaluation mode.
        self.get_logger().info("Policy actor loaded successfully.")
        return policy_net.actor  # Return only the actor network for inference.


    def encoder_to_rad(self, encoder_value):
    # Maps: encoder=2048 -> rad=0, 0 -> +pi, 4096 -> -pi
        return -(np.pi / 2048.0) * (encoder_value - 2048)


    def rad_to_encoder(self, rad_value):
    # Inverse of the above; ensures 0->2048, +pi->0, -pi->4096
        return int(
            2048 - (2048.0 * rad_value / np.pi)
        )

    def compute_rotation_matrix(self, v_from, v_to):
        """
        计算将单位向量 v_from 旋转到 v_to 的旋转矩阵，使用 Rodrigues 公式。
        """
        # 标准化两个向量
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        
        # 计算旋转轴（叉乘）
        v = np.cross(v_from, v_to)
        c = np.dot(v_from, v_to)
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-8:
            # 如果 v_from 和 v_to 已经平行，则返回单位矩阵
            return np.eye(3)
        
        # 构造叉乘的反对称矩阵
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        # Rodrigues 公式
        R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (norm_v ** 2))
        return R

    def rotate_vector(self, vector, rotation_matrix):
        return np.dot(rotation_matrix, vector)

    def accel_callback(self, msg):
        # 1) 提取原始加速度数据
        raw_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # 2) 使用已有的旋转操作进行转换（根据IMU的安装方向）
        R_mount = np.array([
            [0,  0, -1],
            [-1, 0,  0],
            [0,  1,  0]
        ])
        rotated_accel = self.rotate_vector(raw_accel, R_mount)

        # 3) 归一化操作，使得向量长度为1
        norm = np.linalg.norm(rotated_accel)
        if norm > 0:
            rotated_accel = rotated_accel / norm

        # 4) 第一次接收到数据时，记录归一化后的重力方向，并计算校准旋转矩阵，
        #    使得它能将初始重力旋转为理想重力 (0, 0, -1)
        if self.initial_gravity is None:
            self.initial_gravity = rotated_accel.copy()
            target_gravity = np.array([0, 0, -1])
            self.gravity_calib_matrix = self.compute_rotation_matrix(self.initial_gravity, target_gravity)
            self.get_logger().info("Accelerometer calibration rotation matrix computed.")

        # 5) 对新接收到的数据先经过已有的旋转和归一化后，再乘以校准旋转矩阵
        calibrated_accel = np.dot(self.gravity_calib_matrix, rotated_accel)
        norm = np.linalg.norm(calibrated_accel)
        if norm > 0:
            calibrated_accel = calibrated_accel / norm

        # 将校准后的重力数据转换为 torch tensor
        self.projected_gravity = torch.tensor(calibrated_accel, dtype=torch.float32, device=self.device)
    # def rotate_vector(self, vector, rotation_matrix):
    #     return np.dot(rotation_matrix, vector)

    def gyro_callback(self, msg):
        raw_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        R = np.array([
            [0,  0,  -1],
            [-1, 0,  0],
            [0,  1,  0]
        ])
        rotated_ang_vel = self.rotate_vector(raw_ang_vel, R)

        # 第一次接收到陀螺仪数据时，记录初始读数作为偏置
        if not hasattr(self, 'initial_gyro'):
            self.initial_gyro = rotated_ang_vel
            self.get_logger().info("Gyro initial calibration set.")
        
        # 校正：当前读数减去初始读数，使得理想状态为 0
        calibrated_ang_vel = rotated_ang_vel - self.initial_gyro
        self.base_ang_vel = torch.tensor(calibrated_ang_vel, dtype=torch.float32, device=self.device)
    # def gyro_callback(self, msg):
    #         raw_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    #         R = np.array([
    #             [0,  0,  -1],   
    #             [-1, 0,  0],
    #             [0, 1,  0]
    #         ])
    #         rotated_ang_vel = self.rotate_vector(raw_ang_vel, R)
    #         self.base_ang_vel = torch.tensor(rotated_ang_vel, dtype=torch.float32, device=self.device)

    # def accel_callback(self, msg):
    #     raw_accel = np.array([
    #         msg.linear_acceleration.x,
    #         msg.linear_acceleration.y,
    #         msg.linear_acceleration.z
    #     ])
    #     R = np.array([
    #         [0,  0,  -1],
    #         [-1, 0,  0],
    #         [0, 1,  0]
    #     ])
    #     rotated_accel = self.rotate_vector(raw_accel, R)

    #     # 如果初始重力还没有设置（为 None），则进行校准
    #     if self.initial_gravity is None:
    #         self.initial_gravity = rotated_accel.copy()
    #         self.accel_offset = np.array([0, 0, -1]) - self.initial_gravity
    #         self.get_logger().info("Accelerometer initial calibration set.")

    #     calibrated_accel = rotated_accel + self.accel_offset

    #     norm = np.linalg.norm(calibrated_accel)
    #     if norm > 0:
    #         calibrated_accel /= norm

    #     self.projected_gravity = torch.tensor(calibrated_accel, dtype=torch.float32, device=self.device)

    def command_callback(self, msg):
    # Ensure we extract all four values correctly
        self.commands = torch.tensor(
            [msg.linear.x*2, msg.linear.y*2, msg.linear.z],  # Added msg.angular.z
            dtype=torch.float32,
            device=self.device
        )

    def height_callback(self, msg):
# Ensure we extract all four values correctly
        self.height = torch.tensor(
            [msg.data],  
            dtype=torch.float32,
            device=self.device
        )


    def dynamixel_callback(self, msg):
        target_motors = [0, 3, 6, 9, 12, 15, 18, 21]  # Motors we need
        raw_positions = []
        raw_velocities = []
        try:
            for motor_index in target_motors:
                 # Find motor_id index
                position = msg.data[motor_index + 1]  # Read position
                speed = msg.data[motor_index + 2]  # Read speed

                raw_positions.append(position)
                raw_velocities.append(-speed)
                print(raw_positions)
                # print(raw_velocities)
            # Convert positions to radians
            new_dof_pos = torch.tensor([self.encoder_to_rad(pos) for pos in raw_positions], device=self.device)
            new_dof_vel = torch.tensor([(vel * 0.229 * (2 * torch.pi / 60)) for vel in raw_velocities], device=self.device)

            # Ensure position differences are within a reasonable range
            delta_pos = new_dof_pos - self.default_dof_pos
            max_deviation = 1  # Acceptable range limit, adjust if needed

            # Apply filtering for outlier positions
            for i in range(len(delta_pos)):
                if abs(delta_pos[i]) > max_deviation:
                    self.get_logger().warn(f"Outlier detected on Motor {target_motors[i]} Position: {delta_pos[i]} (Replaced with last valid)")
                    new_dof_pos[i] = self.dof_pos[i]  # Replace with last valid position
                else:
                    self.dof_pos[i] = new_dof_pos[i]  # Update valid position

            # Apply filtering for outlier velocities
            max_velocity = 6.0  # Theoretical max velocity in rad/s
            min_velocity = -6.0

            for i in range(len(new_dof_vel)):
                if abs(new_dof_vel[i] * 0.05) > 0.3:  # Check scaled value
                    self.get_logger().warn(f"Outlier detected on Motor {target_motors[i]} Velocity: {new_dof_vel[i]} (Replaced with last valid)")
                    new_dof_vel[i] = self.dof_vel[i]  # Replace with last valid velocity
                else:
                    self.dof_vel[i] = new_dof_vel[i]  # Update valid velocity

            if not self.positions_initialized and torch.any(self.dof_pos != 0):
                self.positions_initialized = True
                self.get_logger().info("Positions initialized and non-zero. RL model will now be deployed.")

        except Exception as e:
            self.get_logger().error(f"Error in dynamixel_callback: {str(e)}")

    def compute_and_publish(self):
        if not self.positions_initialized:
            self.get_logger().info("Waiting for positions to be initialized and non-zero...")
            return

        obs_buf = torch.cat([
            self.base_ang_vel * 0.25,  # (3,)
            self.projected_gravity,  # (3,)
            self.commands,  # (3) -> Needs to be (4)
            (self.dof_pos - self.default_dof_pos),  # (8,)
            self.dof_vel * 0.05,  # (8,)
            self.actions,  # (8,)
            #self.height,
        ], axis=-1).to(self.device)

        #self.get_logger().info(f'Observation Buffer: {obs_buf.tolist()}')

        with torch.no_grad():
            self.actions = self.policy(obs_buf.unsqueeze(0)).squeeze(0)
        #print(self.actions)
        # Apply scaling and offset to actions
        adjusted_actions = (self.actions * 0.25) + self.default_dof_pos

        # Convert actions to encoder values
        encoder_commands = [float(self.rad_to_encoder(action)) for action in adjusted_actions.cpu().tolist()]
        msg = Float32MultiArray()
        msg.data = encoder_commands

        self.publisher.publish(msg)
        #self.get_logger().info(f'Published actions (encoder values): {adjusted_actions}')


def main(args=None):
    rclpy.init(args=args)
    node = MotorSRVNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()