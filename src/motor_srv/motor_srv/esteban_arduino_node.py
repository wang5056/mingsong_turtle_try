import rclpy
from rclpy.node import Node
import serial
import threading
from std_msgs.msg import Float32, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class ArduinoSerialNode(Node):
    def __init__(self):
        super().__init__('esteban_arduino_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Serial parameters
        self.serial_port = '/dev/ttyUSB1'  # Adjust based on your system (e.g., '/dev/ttyUSB0' or 'COM3' on Windows)
        self.baud_rate = 115200

        # Publishers for all sensor data
        self.mv_pub = self.create_publisher(Float32, 'arduino/mv', sensor_qos)
        self.mw_pub = self.create_publisher(Float32, 'arduino/mw', sensor_qos)
        self.latitude_pub = self.create_publisher(Float32, 'arduino/latitude', sensor_qos)
        self.longitude_pub = self.create_publisher(Float32, 'arduino/longitude', sensor_qos)
        self.altitude_pub = self.create_publisher(Float32, 'arduino/altitude', sensor_qos)
        self.height_pub = self.create_publisher(Float32, 'arduino/height', sensor_qos)
        self.p_pos_psi_pub = self.create_publisher(Float32, 'arduino/p_pos_psi', sensor_qos)
        self.p_neg_psi_pub = self.create_publisher(Float32, 'arduino/p_neg_psi', sensor_qos)

        # Subscriber for sending commands to Arduino
        self.command_sub = self.create_subscription(
            String, 'arduino/command', self.command_callback, sensor_qos
        )

        # Connect to serial port
        try:
            self.serial_conn = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            self.get_logger().info(f"Connected to {self.serial_port} at {self.baud_rate} baud.")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            return

        # Start serial reading thread
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def read_serial(self):
        """ Read data from Arduino serial and publish to ROS2 topics """
        while rclpy.ok():
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                self.get_logger().info(f"Raw data from Arduino: {line}")
                # Check for initialization or status messages
                if any(x in line for x in ["Starting", "GNSS OK", "VL53L4CD OK", "Setup complete", 
                                          "Inflating", "Deflating", "Jamming", "Unjamming", 
                                          "Floating", "Sinking", "stopped", "Running", 
                                          "LED ON", "LED OFF", "VL53 init attempt", 
                                          "VL53L4CD not detected", "INA260 found", 
                                          "INA260 not detected", "GPS connected", 
                                          "GPS not detected"]):
                    self.get_logger().info(f"Arduino status: {line}")
                    continue

                # Parse sensor data
                if line.startswith("mV:"):
                    # Split into sections
                    sections = line.split(", ")
                    if len(sections) != 8:
                        self.get_logger().warn(f"Invalid data format: {line}")
                        continue

                    # Parse each section
                    try:
                        # mV
                        mv_str = sections[0].split(": ")[1]
                        mv = float(mv_str) if mv_str != 'ovf' else float('nan')  # Handle 'ovf' as NaN

                        # mW
                        mw_str = sections[1].split(": ")[1]
                        mw = float(mw_str) if mw_str != 'ovf' else float('nan')  # Handle 'ovf' as NaN

                        # Lat
                        lat_str = sections[2].split(": ")[1]
                        latitude = float(lat_str)

                        # Lon
                        lon_str = sections[3].split(": ")[1]
                        longitude = float(lon_str)

                        # Alt (strip 'm')
                        alt_str = sections[4].split(": ")[1].rstrip('m')
                        altitude = float(alt_str)

                        # Height (strip 'mm' and convert to m)
                        height_str = sections[5].split(": ")[1].rstrip('mm')
                        height_mm = float(height_str)
                        height = height_mm / 1000.0  # Convert mm to m

                        # P_pos_psi (strip 'psi')
                        p_pos_psi_str = sections[6].split(": ")[1].rstrip('psi')
                        p_pos_psi = float(p_pos_psi_str)

                        # P_neg_psi (strip 'psi')
                        p_neg_psi_str = sections[7].split(": ")[1].rstrip('psi')
                        p_neg_psi = float(p_neg_psi_str)

                        # Publish
                        self.mv_pub.publish(Float32(data=mv))
                        self.mw_pub.publish(Float32(data=mw))
                        self.latitude_pub.publish(Float32(data=latitude))
                        self.longitude_pub.publish(Float32(data=longitude))
                        self.altitude_pub.publish(Float32(data=altitude))
                        self.height_pub.publish(Float32(data=height))
                        self.p_pos_psi_pub.publish(Float32(data=p_pos_psi))
                        self.p_neg_psi_pub.publish(Float32(data=p_neg_psi))

                        self.get_logger().info(
                            f"Published: mV={mv}, mW={mw}, "
                            f"Lat={latitude}, Lon={longitude}, Alt={altitude} m, Height={height} m, "
                            f"P_pos_psi={p_pos_psi} psi, P_neg_psi={p_neg_psi} psi"
                        )
                    except (ValueError, IndexError) as e:
                        self.get_logger().warn(f"Failed to parse data: {line}, Error: {e}")
            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}")
                break
            except UnicodeDecodeError as e:
                self.get_logger().warn(f"Failed to decode serial data: {e}")
                continue

    def command_callback(self, msg):
        """ Listen to ROS2 topic and send commands to Arduino """
        command = msg.data.strip()
        
        # Split the command into parts (e.g., 'u,6' -> ['u', '6'])
        parts = command.split(',')
        
        # Check for valid commands
        if len(parts) == 1 and parts[0] == 'b':  # 'b' command has no number
            self.serial_conn.write((command + "\n").encode('utf-8'))
            self.get_logger().info(f"Sent command to Arduino: {command}")
        elif len(parts) == 2 and parts[0] in ['i', 'd', 'j', 'u', 'f', 's']:  # Commands with number
            try:
                # Ensure the second part is a valid number
                number = int(parts[1])  # Convert to integer to validate
                self.serial_conn.write((command + "\n").encode('utf-8'))
                self.get_logger().info(f"Sent command to Arduino: {command}")
            except ValueError:
                self.get_logger().warn(f"Invalid number in command: {command}")
        else:
            self.get_logger().warn(f"Invalid command received: {command}")

    def destroy_node(self):
        """ Clean up resources when shutting down """
        super().destroy_node()
        if hasattr(self, 'serial_conn') and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info("Serial connection closed.")

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Arduino Serial Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()