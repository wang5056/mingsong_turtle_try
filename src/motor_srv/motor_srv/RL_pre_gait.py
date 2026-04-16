import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String, Float32
import torch
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import time
import math
from rclpy.executors import MultiThreadedExecutor

# Define the ActorCritic class to match the saved checkpoint
class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, activation='elu'):
        super(ActorCritic, self).__init__()
        if activation == 'elu':
            self.activation = torch.nn.ELU()
       
        actor_layers = []
        prev_dim = obs_dim
        for dim in actor_hidden_dims:
            actor_layers.append(torch.nn.Linear(prev_dim, dim))
            actor_layers.append(self.activation)
            prev_dim = dim
        actor_layers.append(torch.nn.Linear(prev_dim, action_dim))
        self.actor = torch.nn.Sequential(*actor_layers)

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

        self.initial_gravity = None
        self.gravity_calib_matrix = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        delay_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )

        self.create_subscription(Imu, '/camera/gyro/sample', self.gyro_callback, sensor_qos)
        self.create_subscription(Imu, '/camera/accel/sample', self.accel_callback, sensor_qos)
        self.create_subscription(Twist, '/motor_commands', self.command_callback, sensor_qos)
        self.create_subscription(Float32MultiArray, '/dynamixel_status', self.dynamixel_callback, sensor_qos)
        self.create_subscription(Float32MultiArray, '/arduino/height', self.height_callback, sensor_qos)
        self.create_subscription(String, '/gait_mode', self.mode_callback, sensor_qos)
        self.create_subscription(Float32, '/gait_delay', self.gait_delay_callback, delay_qos)
        self.create_subscription(String, '/gait_turn', self.gait_turn_callback, sensor_qos)

        self.publisher = self.create_publisher(Float32MultiArray, '/robot/motor_commands', 10)
        self.gait_publisher = self.create_publisher(Float32MultiArray, '/robot/motor_commands', 10)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.base_ang_vel = torch.zeros(3, device=self.device)
        self.projected_gravity = torch.zeros(3, device=self.device)
        self.commands = torch.zeros(3, device=self.device)
        self.height = torch.zeros(1, device=self.device)
        self.dof_pos = torch.zeros(8, device=self.device)
        self.last_dof_pos = torch.zeros(8, device=self.device)
        self.dof_vel = torch.zeros(8, device=self.device)
        self.actions = torch.zeros(8, device=self.device)
        self.initial_gravity = None
        self.last_valid_dof_pos = torch.zeros(8, device=self.device)
        self.last_valid_dof_vel = torch.zeros(8, device=self.device)
        self.positions_initialized = False

        self.policy = self.load_policy(
            "src/mingsong_turtle_try/src/motor_srv/models/model_700_01m.pt",
            obs_dim=33,
            action_dim=8,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )

        self.default_dof_pos = torch.tensor(
            [1.5, -1.3, 1.3, -1.5, 0.15, -0.15, 0.15, -0.15], dtype=torch.float32
        ).to(self.device)

        self.rl_timer = None
        self.gait_timer = None
        self.gait_ready_time = 0.0
        self.mode = 'RL'
        self.gait_turn_mode = 'normal'  # 'normal', 'left', 'right'
        self.start_rl_timer()

        self.get_logger().info("Starting in RL mode. Press 'r' for RL, 'f' for GAIT (drag_square).")

        # Gait 参数 (使用度)
        self.gait_cycle = 5
        self.gait_total_phases = int(self.gait_cycle * 200)
        self.gait_phase_index = 0
        self.gait_step_length = 100  # 度
        self.gait_step_height = 50  # 度
        self.gait_initial_angle_a = -20  # 度
        self.gait_initial_angle_b = -60  # 度
        self.gait_move_delay = 0.0081
        self.gait_running = False
        self.gait_ready_executed = False
        self.last_gait_execution_time = 0.0  # 记录上一次步态执行时间

    def start_rl_timer(self):
        """Start or restart the timer for RL mode."""
        if self.rl_timer is not None:
            self.rl_timer.cancel()
        self.rl_timer = self.create_timer(0.1, self.rl_compute_and_publish)
        self.get_logger().info("RL mode timer started with 0.1s interval.")

    def stop_rl_timer(self):
        """Stop the timer for RL mode."""
        if self.rl_timer is not None:
            self.rl_timer.cancel()
            self.rl_timer = None
            self.get_logger().info("RL mode timer stopped.")

    def start_gait_timer(self):
        """Start the timer for GAIT mode with fixed frequency."""
        if self.gait_timer is not None:
            self.gait_timer.cancel()
        self.gait_timer = self.create_timer(0.001, self.gait_compute_and_publish)  # 固定 1000 Hz
        self.get_logger().info("GAIT mode timer started with fixed 0.001s interval.")

    def stop_gait_timer(self):
        """Stop the timer for GAIT mode."""
        if self.gait_timer is not None:
            self.gait_timer.cancel()
            self.gait_timer = None
            self.get_logger().info("GAIT mode timer stopped.")

    def gait_delay_callback(self, msg):
        if self.mode == 'GAIT':
            new_delay = clamp(msg.data, 0.001, 0.1)
            if abs(new_delay - self.gait_move_delay) > 1e-6:
                self.gait_move_delay = new_delay
                self.get_logger().info(f"Gait step delay updated to {self.gait_move_delay:.4f}s (effective frequency: {1/self.gait_move_delay:.2f} Hz)")

    def gait_turn_callback(self, msg):
        if self.mode == 'GAIT':
            if msg.data in ['normal', 'left', 'right']:
                self.gait_turn_mode = msg.data
                self.get_logger().info(f"Gait turn mode updated to {self.gait_turn_mode}")

    def load_policy(self, checkpoint_path, obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims, activation):
        policy_net = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        ).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        actor_state_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        policy_net.actor.load_state_dict(actor_state_dict, strict=False)
        policy_net.actor.eval()
        self.get_logger().info("Policy actor loaded successfully.")
        return policy_net.actor

    def encoder_to_rad(self, encoder_value):
        return -(np.pi / 2048.0) * (encoder_value - 2048)

    def rad_to_encoder(self, rad_value):
        return int(2048 - (2048.0 * rad_value / np.pi))

    def compute_rotation_matrix(self, v_from, v_to):
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        v = np.cross(v_from, v_to)
        c = np.dot(v_from, v_to)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return np.eye(3)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (norm_v ** 2))
        return R

    def rotate_vector(self, vector, rotation_matrix):
        return np.dot(rotation_matrix, vector)

    def accel_callback(self, msg):
        raw_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        R_mount = np.array([
            [0,  0, -1],
            [-1, 0,  0],
            [0, 1,  0]
        ])
        rotated_accel = self.rotate_vector(raw_accel, R_mount)
        norm = np.linalg.norm(rotated_accel)
        if norm > 0:
            rotated_accel = rotated_accel / norm
        if self.initial_gravity is None:
            self.initial_gravity = rotated_accel.copy()
            target_gravity = np.array([0, 0, -1])
            self.gravity_calib_matrix = self.compute_rotation_matrix(self.initial_gravity, target_gravity)
            self.get_logger().info("Accelerometer calibration rotation matrix computed.")
        calibrated_accel = np.dot(self.gravity_calib_matrix, rotated_accel)
        norm = np.linalg.norm(calibrated_accel)
        if norm > 0:
            calibrated_accel = calibrated_accel / norm
        self.projected_gravity = torch.tensor(calibrated_accel, dtype=torch.float32, device=self.device)

    def gyro_callback(self, msg):
        raw_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        R = np.array([
            [0,  0, -1],
            [-1, 0,  0],
            [0, 1,  0]
        ])
        rotated_ang_vel = self.rotate_vector(raw_ang_vel, R)
        if not hasattr(self, 'initial_gyro'):
            self.initial_gyro = rotated_ang_vel
            self.get_logger().info("Gyro initial calibration set.")
        calibrated_ang_vel = rotated_ang_vel - self.initial_gyro
        self.base_ang_vel = torch.tensor(calibrated_ang_vel, dtype=torch.float32, device=self.device)

    def command_callback(self, msg):
        self.commands = torch.tensor(
            [msg.linear.x*2, msg.linear.y*2, msg.linear.z], # msg.angular.z
            dtype=torch.float32,
            device=self.device
        )

    def height_callback(self, msg):
        self.height = torch.tensor(
            [msg.data],
            dtype=torch.float32,
            device=self.device
        )

    def dynamixel_callback(self, msg):
        target_motors = [0, 3, 6, 9, 12, 15, 18, 21]
        raw_positions = []
        raw_velocities = []
        try:
            for motor_index in target_motors:
                position = msg.data[motor_index + 1]
                speed = msg.data[motor_index + 2]
                raw_positions.append(position)
                raw_velocities.append(-speed)
            new_dof_pos = torch.tensor([self.encoder_to_rad(pos) for pos in raw_positions], device=self.device)
            new_dof_vel = torch.tensor([(vel * 0.229 * (2 * torch.pi / 60)) for vel in raw_velocities], device=self.device)
            delta_pos = new_dof_pos - self.default_dof_pos
            max_deviation = 1
            for i in range(len(delta_pos)):
                if abs(delta_pos[i]) > max_deviation:
                    self.get_logger().warn(f"Outlier detected on Motor {target_motors[i]} Position: {delta_pos[i]} (Replaced with last valid)")
                    new_dof_pos[i] = self.dof_pos[i]
                else:
                    self.dof_pos[i] = new_dof_pos[i]
            max_velocity = 6.0
            min_velocity = -6.0
            for i in range(len(new_dof_vel)):
                if abs(new_dof_vel[i] * 0.05) > 0.3:
                    self.get_logger().warn(f"Outlier detected on Motor {target_motors[i]} Velocity: {new_dof_vel[i]} (Replaced with last valid)")
                    new_dof_vel[i] = self.dof_vel[i]
                else:
                    self.dof_vel[i] = new_dof_vel[i]
            if not self.positions_initialized and torch.any(self.dof_pos != 0):
                self.positions_initialized = True
                self.get_logger().info("Positions initialized and non-zero. RL model will now be deployed.")
        except Exception as e:
            self.get_logger().error(f"Error in dynamixel_callback: {str(e)}")

    def mode_callback(self, msg):
        new_mode = msg.data
        if new_mode in ['RL', 'GAIT']:
            if self.mode != new_mode:
                self.get_logger().info(f"Attempting to switch from {self.mode} to {new_mode}")
                self.mode = new_mode
                self.gait_phase_index = 0
                self.gait_ready_executed = False
                self.last_gait_execution_time = 0.0
                self.gait_turn_mode = 'normal'  # Reset turn mode on mode change
                if new_mode == 'RL':
                    self.stop_gait_timer()
                    self.start_rl_timer()
                else:
                    self.stop_rl_timer()
                    self.gait_move_delay = 0.0081  # Reset delay
                    amplitude = 45
                    omega = 1.0
                    phase = 0
                    angle_b = -10 + amplitude * math.cos(omega * phase + math.pi)
                    AOT = 90
                    encoder_commands = [
                        float(self.rad_to_encoder(np.deg2rad(40))),
                        float(self.rad_to_encoder(np.deg2rad(10))),
                        float(self.rad_to_encoder(np.deg2rad(-10))),
                        float(self.rad_to_encoder(np.deg2rad(-40))),
                        float(self.rad_to_encoder(np.deg2rad(-angle_b))),
                        float(self.rad_to_encoder(np.deg2rad(-90))),
                        float(self.rad_to_encoder(np.deg2rad(90))),
                        float(self.rad_to_encoder(np.deg2rad(angle_b))),
                        float(self.rad_to_encoder(np.deg2rad(AOT))),
                        float(self.rad_to_encoder(np.deg2rad(90))),
                        float(self.rad_to_encoder(np.deg2rad(-90))),
                        float(self.rad_to_encoder(np.deg2rad(-AOT)))
                    ]
                    msg = Float32MultiArray()
                    msg.data = encoder_commands
                    self.get_logger().info(f"Ready drag positions: {encoder_commands}")
                    self.gait_publisher.publish(msg)
                    self.gait_ready_time = time.time() + 1.0
                    self.start_gait_timer()
                self.get_logger().info(f"Switched to {new_mode} mode.")

    def rl_compute_and_publish(self):
        if not self.positions_initialized:
            self.get_logger().info("Waiting for positions to be initialized and non-zero...")
            return

        self.get_logger().info("Current mode: RL")
        obs_buf = torch.cat([
            self.base_ang_vel * 0.25,
            self.projected_gravity,
            self.commands,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * 0.05,
            self.actions,
        ], axis=-1).to(self.device)
        with torch.no_grad():
            self.actions = self.policy(obs_buf.unsqueeze(0)).squeeze(0)
        adjusted_actions = (self.actions * 0.25) + self.default_dof_pos
        encoder_commands = [float(self.rad_to_encoder(action)) for action in adjusted_actions.cpu().tolist()]
        self.get_logger().info(f"RL Encoder commands (motors {', '.join(map(str, [0, 3, 6, 9, 12, 15, 18, 21]))}): {encoder_commands}")
        msg = Float32MultiArray()
        msg.data = encoder_commands
        self.publisher.publish(msg)

    def gait_compute_and_publish(self):
        if not self.positions_initialized:
            self.get_logger().info("Waiting for positions to be initialized and non-zero...")
            return

        current_time = time.time()
        if not self.gait_ready_executed:
            if current_time >= self.gait_ready_time:
                self.gait_ready_executed = True
                self.last_gait_execution_time = current_time
            return

        # 检查是否达到目标步态延迟
        if current_time - self.last_gait_execution_time < self.gait_move_delay:
            return  # 未达到延迟时间，跳过执行

        self.last_gait_execution_time = current_time

        # 每10次记录一次日志
        if self.gait_phase_index % 10 == 0:
            elapsed_time = current_time - (self.last_gait_execution_time - self.gait_move_delay)
            self.get_logger().info(f"Executing GAIT step, elapsed time: {elapsed_time:.4f}s, target delay: {self.gait_move_delay:.4f}s")

        # Execute drag_square gait
        theta = (self.gait_phase_index / self.gait_total_phases) * (self.gait_cycle * 2 * math.pi)
        norm_phase = (theta % (2 * math.pi)) / (2 * math.pi)
        if norm_phase < 0.25:
            x_offset = self.gait_step_height
            y_offset = self.gait_step_length * (norm_phase / 0.25)
        elif norm_phase < 0.5:
            x_offset = self.gait_step_height - 2 * self.gait_step_height * ((norm_phase - 0.25) / 0.25)
            y_offset = self.gait_step_length
        elif norm_phase < 0.75:
            x_offset = -self.gait_step_height
            y_offset = self.gait_step_length * (1 - (norm_phase - 0.5) / 0.25)
        else:
            x_offset = -self.gait_step_height + 2 * self.gait_step_height * ((norm_phase - 0.75) / 0.25)
            y_offset = 0
        angle_a = self.gait_initial_angle_a - x_offset
        angle_b = self.gait_initial_angle_b + y_offset
        AOT = 90 if norm_phase < 0.5 else 0
        angles = [
            -angle_a,  # 0: Front left shoulder
            10,       # 3: Back left shoulder
            -10,      # 9: Back right shoulder
            angle_a,  # 6: Front right shoulder
            -angle_b, # 1: Front left knee
            -90,      # 4: Back left knee
            90,       # 10: Back right knee
            angle_b,  # 7: Front right knee
            -AOT,      # 2: Front left pitch
            90,       # 5: Back left pitch
            -90,      # 11: Back right pitch
            AOT,     # 8: Front right pitch
        ]
        # Apply turn mode overrides
        if self.gait_turn_mode == 'left':  # Disable left front limb
            angles[0] = 40  # Front left shoulder
            angles[4] = -55  # Front left knee (use right knee angle for symmetry)
            angles[8] = -10  # Front left pitch
        elif self.gait_turn_mode == 'right':  # Disable right front limb
            angles[3] = -40   # Front right shoulder
            angles[7] = 55  # Front right knee (use left knee angle for symmetry)
            angles[11] = 10  # Front right pitch

        encoder_commands = [float(self.rad_to_encoder(np.deg2rad(deg))) for deg in angles]
        motor_labels = [
            "Front left shoulder (0)", "Back left shoulder (1)", "Back right shoulder (2)", "Front right shoulder (3)",
            "Front left knee (4)", "Back left knee (5)", "Back right knee (6)", "Front right knee (7)",
            "Front left pitch (8)", "Back left pitch (9)", "Back right pitch (10)", "Front right pitch (11)"
        ]
        if self.gait_phase_index % 10 == 0:
            log_message = "\n".join(
                f"Motor {label}: Angle={angles[i]:.2f} deg, Encoder={encoder_commands[i]:.2f}"
                for i, label in enumerate(motor_labels)
            )
            self.get_logger().info(f"GAIT Encoder commands (turn mode: {self.gait_turn_mode}):\n{log_message}")
        msg = Float32MultiArray()
        msg.data = encoder_commands
        self.gait_publisher.publish(msg)
        self.gait_phase_index = (self.gait_phase_index + 1) % self.gait_total_phases

def clamp(value, min_value, max_value):
    """Clamps a value within the specified range."""
    return max(min_value, min(value, max_value))

def main(args=None):
    rclpy.init(args=args)
    node = MotorSRVNode()
    executor = MultiThreadedExecutor(num_threads=8)
    rclpy.spin(node, executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()