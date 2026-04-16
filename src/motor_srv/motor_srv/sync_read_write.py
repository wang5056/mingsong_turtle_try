import rclpy
import time 
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from dynamixel_sdk import *
from dynamixel_sdk.registerDict import X_Series
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class DynamixelControlNode(Node):
    def __init__(self):
        super().__init__('dynamixel_control_node')

        sensor_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10
                )

        # Dynamixel settings
        self.DEVICENAME = '/dev/ttyUSB0'
        self.BAUDRATE = 4000000
        self.PROTOCOL_VERSION = 2.0
        self.DXL_IDS = list([0,1,2,3,4,5,6,7,8,9,10,11])  # Motor IDs from 0 to 11

        # Default positions
        self.goal_positions = [1205,2890,1205,2890,1956,2147,1956,2147,2048,2048,2048,2048]
        self.target_positions = [1205,2890,1205,2890,1956,2147,1956,2147,2048,2048,2048,2048]
        # self.goal_positions = [2503,1934,2162,1593,2674,3072,1024,1422,2192,1024,3072,1934]
        # self.target_positions = [2503,1934,2162,1593,2674,3072,1024,1422,2192,1024,3072,1934]

        self.motor_commands_available = False
        self.ready_to_move = True

        # Initialize communication with Dynamixel motors
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            self.get_logger().error("Failed to open port")
            rclpy.shutdown()
            return

        if not self.port_handler.setBaudRate(self.BAUDRATE):
            self.get_logger().error("Failed to set baudrate")
            rclpy.shutdown()
            return

        self.get_logger().info("Port opened and baudrate set")

        # Initialize SyncWrite
        self.goal_position_write = GroupSyncWrite(self.port_handler, self.packet_handler, 
                                                  X_Series["ADDR_GOAL_POSITION"], X_Series["LEN_GOAL_POSITION"])

        self.enable_torque()
        self.set_motor_gains(kp=800, kd=15)

        # SyncRead for position, velocity and current
        self.position_read = GroupSyncRead(self.port_handler, self.packet_handler, 
                                           X_Series["ADDR_PRESENT_POSITION"], X_Series["LEN_PRESENT_POSITION"])
        self.velocity_read = GroupSyncRead(self.port_handler, self.packet_handler, 
                                           X_Series["ADDR_PRESENT_VELOCITY"], X_Series["LEN_PRESENT_VELOCITY"])
        self.current_read = GroupSyncRead(self.port_handler, self.packet_handler,
                                         X_Series["ADDR_PRESENT_CURRENT"], X_Series["LEN_PRESENT_CURRENT"])

        for dxl_id in self.DXL_IDS:
            self.position_read.addParam(dxl_id)
            self.velocity_read.addParam(dxl_id)
            self.current_read.addParam(dxl_id)

        # Create ROS publishers and subscribers
        self.motor_status_publisher = self.create_publisher(Float32MultiArray, 'dynamixel_status', sensor_qos)
        self.current_publisher = self.create_publisher(Float32MultiArray, 'dynamixel_current', sensor_qos)

        # Subscribe to `goal_positions` and `/robot/motor_commands`
        self.goal_position_subscriber = self.create_subscription(
            Float32MultiArray,
            'goal_positions',
            self.goal_position_callback,
            10
        )
        self.motor_commands_subscriber = self.create_subscription(
            Float32MultiArray,
            '/robot/motor_commands',
            self.motor_commands_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)

    def enable_torque(self):
        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, X_Series["ADDR_TORQUE_ENABLE"], X_Series["TORQUE_ENABLE"]
            )
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                self.get_logger().info(f"Motor {dxl_id} torque enabled successfully")
            else:
                self.get_logger().error(f"Failed to enable torque for Motor {dxl_id}, error: {dxl_comm_result}")

    def set_motor_gains(self, kp=640, kd=0):
        kp = int(kp)
        kd = int(kd)

        for dxl_id in self.DXL_IDS:
            operating_mode, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
                self.port_handler, dxl_id, 11
            )

            if operating_mode not in [3, 4]:
                self.get_logger().error(f"Motor {dxl_id} is not in Position Control Mode! Current mode: {operating_mode}")
                continue

            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                self.port_handler, dxl_id, 84, kp
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                self.get_logger().error(f"Failed to set Kp for Motor {dxl_id}, error: {dxl_comm_result}")

            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                self.port_handler, dxl_id, 80, kd
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                self.get_logger().error(f"Failed to set Kd for Motor {dxl_id}, error: {dxl_comm_result}")

        self.get_logger().info(f"Set Kp={kp}, Kd={kd} for all motors.")

    def move_to_zero_position(self):
        for dxl_id in self.DXL_IDS:
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(2048)),
                DXL_HIBYTE(DXL_LOWORD(2048)),
                DXL_LOBYTE(DXL_HIWORD(2048)),
                DXL_HIBYTE(DXL_HIWORD(2048))
            ]
            self.goal_position_write.addParam(dxl_id, param_goal_position)

        self.goal_position_write.txPacket()
        self.goal_position_write.clearParam()
        self.get_logger().info("Motors moved to zero position, waiting 0.1s...")

        time.sleep(1)
        self.ready_to_move = True
        self.get_logger().info("Ready to move to target positions.")

    def goal_position_callback(self, msg):
        if not self.motor_commands_available:
            if len(msg.data) == len(self.DXL_IDS):
                self.goal_positions = [int(pos) for pos in msg.data]

    def motor_commands_callback(self, msg):
        if len(msg.data) == 12:
            self.motor_commands_available = True
            self.target_positions = [int(pos) for pos in msg.data]
            self.get_logger().info(f"Received 12 motor commands: {msg.data}")
        elif len(msg.data) == 8:
            self.motor_commands_available = True
            for i, motor_index in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
                self.target_positions[motor_index] = int(msg.data[i])
            for motor_index in [8, 9, 10, 11]:
                self.target_positions[motor_index] = 2048
            self.get_logger().info(f"Received 8 motor commands, padding [8, 9, 10, 11] with 2048: {msg.data}")
        else:
            self.get_logger().warn(f"Invalid motor commands length: {len(msg.data)}, expected 8 or 12")

    def convert_to_signed(self, value, bit_length=32):
        if value >= (1 << (bit_length - 1)):
            value -= (1 << bit_length)
        return value

    def control_loop(self):
        # Use the appropriate positions based on whether motor_commands is available
        raw_positions = self.target_positions if self.motor_commands_available else self.goal_positions
        motor_mapping = {
            0: raw_positions[0],  1: raw_positions[1],  2: raw_positions[2],  3: raw_positions[3],
            4: raw_positions[4],  5: raw_positions[5],  6: raw_positions[6],  7: raw_positions[7],
            8: raw_positions[8],  9: raw_positions[9],  10: raw_positions[10], 11: raw_positions[11]
        }

        # Send goal positions
        for dxl_id in self.DXL_IDS:
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(motor_mapping[dxl_id])),
                DXL_HIBYTE(DXL_LOWORD(motor_mapping[dxl_id])),
                DXL_LOBYTE(DXL_HIWORD(motor_mapping[dxl_id])),
                DXL_HIBYTE(DXL_HIWORD(motor_mapping[dxl_id]))
            ]
            self.goal_position_write.addParam(dxl_id, param_goal_position)

        self.goal_position_write.txPacket()
        self.goal_position_write.clearParam()

        # Read position, velocity and current
        self.position_read.txRxPacket()
        self.velocity_read.txRxPacket()
        self.current_read.txRxPacket()

        # Publish motor status (position and velocity)
        status_msg = Float32MultiArray()
        motor_data = []
        for dxl_id in self.DXL_IDS:
            pos = self.position_read.getData(dxl_id, X_Series["ADDR_PRESENT_POSITION"], 4)
            vel = self.velocity_read.getData(dxl_id, X_Series["ADDR_PRESENT_VELOCITY"], 4)
            vel = self.convert_to_signed(vel, 32)
            motor_data.extend([float(dxl_id), float(pos), float(vel)])
        status_msg.data = motor_data
        self.motor_status_publisher.publish(status_msg)

        # Publish current data
        current_msg = Float32MultiArray()
        current_data = []
        for dxl_id in self.DXL_IDS:
            curr = self.current_read.getData(dxl_id, X_Series["ADDR_PRESENT_CURRENT"], 2)
            curr = self.convert_to_signed(curr, 16)
            current_data.extend([float(dxl_id), float(curr)])
        current_msg.data = current_data
        self.current_publisher.publish(current_msg)

def main(): 
    rclpy.init()
    node = DynamixelControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()