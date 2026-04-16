import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import sys
import select
import termios
import tty

def apply_deadzone(value, threshold=0.1):
    """Applies a deadzone to filter out small noise values."""
    return value if abs(value) > threshold else 0.0

def clamp(value, min_value, max_value):
    """Clamps a value within the specified range."""
    return max(min_value, min(value, max_value))

class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')
        self.publisher_ = self.create_publisher(Twist, '/motor_commands', 10)
        self.mode_publisher = self.create_publisher(String, '/gait_mode', 10)
        self.gait_delay_publisher = self.create_publisher(Float32, '/gait_delay', 10)
        self.gait_turn_publisher = self.create_publisher(String, '/gait_turn', 10)

        self.A_x = 0.0
        self.A_y = 0.0
        self.ang_vel = 0.0
        self.height = 0.2
        self.speed_increment = 0.25
        self.angular_increment = 0.02
        self.height_increment = 0.01

        self.gait_delay = 0.1
        self.delay_increment = 0.002
        self.min_delay = 0.001
        self.max_delay = 0.1

        self.current_mode = 'RL'
        self.turn_mode = 'normal'  # Track turn mode for 'w' logic

        self.prev_A_x = self.A_x
        self.prev_A_y = self.A_y
        self.prev_ang_vel = self.ang_vel
        self.prev_height = self.height
        self.prev_gait_delay = self.gait_delay

        self.get_logger().info("Keyboard control node started.")
        self.get_logger().info("W/S: A_x or gait speed (faster/slower) or switch back from turn, A/D: A_y or turn modes in GAIT, ←/→: ang_vel, ↑/↓: height, F: GAIT (drag_square), R: RL, Q: Quit")

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.timer = self.create_timer(0.05, self.control_loop)

    def get_key(self):
        key = None
        if select.select([sys.stdin], [], [], 0.05)[0]:
            key = sys.stdin.read(1)
            if key == '\x1b':
                key += sys.stdin.read(2)
        return key

    def control_loop(self):
        key = self.get_key()
        updated = False

        if key == 'w':
            if self.current_mode == 'RL':
                self.A_x = clamp(self.A_x + self.speed_increment, -1.0, 1.0)
            else:  # GAIT
                if self.turn_mode != 'normal':
                    self.turn_mode = 'normal'
                    turn_msg = String()
                    turn_msg.data = 'normal'
                    self.gait_turn_publisher.publish(turn_msg)
                    self.get_logger().info("Switched back to normal GAIT mode.")
                else:
                    self.gait_delay = clamp(self.gait_delay - self.delay_increment, self.min_delay, self.max_delay)
                    delay_msg = Float32()
                    delay_msg.data = self.gait_delay
                    self.gait_delay_publisher.publish(delay_msg)
                    self.get_logger().info(f"Gait delay decreased to {self.gait_delay:.4f}s (effective frequency: {1/self.gait_delay:.2f} Hz)")
            updated = True
        elif key == 's':
            if self.current_mode == 'RL':
                self.A_x = clamp(self.A_x - self.speed_increment, -1.0, 1.0)
            else:  # GAIT
                self.gait_delay = clamp(self.gait_delay + self.delay_increment, self.min_delay, self.max_delay)
                delay_msg = Float32()
                delay_msg.data = self.gait_delay
                self.gait_delay_publisher.publish(delay_msg)
                self.get_logger().info(f"Gait delay increased to {self.gait_delay:.4f}s (effective frequency: {1/self.gait_delay:.2f} Hz)")
            updated = True
        elif key == 'a':
            if self.current_mode == 'RL':
                self.A_y = clamp(self.A_y + self.speed_increment, -1.0, 1.0)
            else:  # GAIT
                self.turn_mode = 'left'
                turn_msg = String()
                turn_msg.data = 'left'
                self.gait_turn_publisher.publish(turn_msg)
                self.get_logger().info("Switched to left turn GAIT mode (right front limb active).")
            updated = True
        elif key == 'd':
            if self.current_mode == 'RL':
                self.A_y = clamp(self.A_y - self.speed_increment, -1.0, 1.0)
            else:  # GAIT
                self.turn_mode = 'right'
                turn_msg = String()
                turn_msg.data = 'right'
                self.gait_turn_publisher.publish(turn_msg)
                self.get_logger().info("Switched to right turn GAIT mode (left front limb active).")
            updated = True
        elif key == '\x1b[D':  # Left arrow
            self.ang_vel = clamp(self.ang_vel + self.angular_increment, -1.0, 1.0)
            updated = True
        elif key == '\x1b[C':  # Right arrow
            self.ang_vel = clamp(self.ang_vel - self.angular_increment, -1.0, 1.0)
            updated = True
        elif key == '\x1b[A':  # Up arrow
            self.height = clamp(self.height + self.height_increment, 0.1, 0.28)
            updated = True
        elif key == '\x1b[B':  # Down arrow
            self.height = clamp(self.height - self.height_increment, 0.1, 0.28)
            updated = True
        elif key == 'x':
            self.A_x = 0.0
            self.A_y = 0.0
            self.ang_vel = 0.0
            updated = True
            self.get_logger().info("All values reset to zero.")
        elif key == 'f':
            self.current_mode = 'GAIT'
            mode_msg = String()
            mode_msg.data = 'GAIT'
            self.mode_publisher.publish(mode_msg)
            self.gait_delay = 0.0081  # Reset to default
            delay_msg = Float32()
            delay_msg.data = self.gait_delay
            self.gait_delay_publisher.publish(delay_msg)
            self.turn_mode = 'normal'
            turn_msg = String()
            turn_msg.data = 'normal'
            self.gait_turn_publisher.publish(turn_msg)
            self.get_logger().info("Switched to GAIT (drag_square) with default delay 0.0081s.")
        elif key == 'r':
            self.current_mode = 'RL'
            mode_msg = String()
            mode_msg.data = 'RL'
            self.mode_publisher.publish(mode_msg)
            self.get_logger().info("Switched to RL.")
        elif key == 'q':
            self.get_logger().info("Exiting keyboard control node.")
            rclpy.shutdown()

        twist_msg = Twist()
        twist_msg.linear.x = apply_deadzone(self.A_x)
        twist_msg.linear.y = apply_deadzone(self.A_y)
        twist_msg.linear.z = apply_deadzone(self.ang_vel)
        twist_msg.angular.z = apply_deadzone(self.height)
        self.publisher_.publish(twist_msg)

        if updated:
            self.get_logger().info(f"A_x: {self.A_x:.2f}, A_y: {self.A_y:.2f}, ang_vel: {self.ang_vel:.2f}, height: {self.height:.2f}, gait_delay: {self.gait_delay:.4f}s, turn_mode: {self.turn_mode}")

        self.prev_A_x = self.A_x
        self.prev_A_y = self.A_y
        self.prev_ang_vel = self.ang_vel
        self.prev_height = self.height
        self.prev_gait_delay = self.gait_delay

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()