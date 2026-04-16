# LUART

ROS 2 workspace for the LUART turtle robot. Runs inside the NVIDIA Isaac ROS dev container on a Jetson-class host and provides motor control, RL policy execution, Arduino sensor I/O, and COT (cost of transport) computation.

---

## 1. Prerequisites (host machine)

- Jetson (or x86 host) with NVIDIA driver + Docker + NVIDIA Container Toolkit installed. Follow [dusty-nv/jetson-containers setup](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) (docker group, swap size).
- Workspace cloned as `${ISAAC_ROS_WS}` (typically `~/Downloads/isaac_ros-dev`). This repo lives at `${ISAAC_ROS_WS}/src/mingsong_turtle_try`.
- `isaac_ros_common` cloned under `${ISAAC_ROS_WS}/src/isaac_ros_common` — provides `scripts/run_dev.sh` used to enter the container.
- Hardware connected over USB: Dynamixel U2D2, Arduino, RealSense D435i.

Generate the NVIDIA CDI config once per host (needed for GPU passthrough):
```bash
sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml
```

Grant access to serial devices (redo after every reboot / replug):
```bash
sudo usermod -a -G dialout $USER
ls -l /dev/ttyUSB* /dev/ttyACM*
sudo chmod 777 /dev/ttyUSB0   # Dynamixel / Arduino
sudo chmod 777 /dev/ttyUSB1
sudo chmod 777 /dev/ttyACM0
```

---

## 2. Enter the Docker container

From the workspace root (`${ISAAC_ROS_WS}`):

```bash
./src/mingsong_turtle_try/run_isaac_ros.sh
```

This is just a wrapper for:
```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/run_dev.sh ${ISAAC_ROS_WS}
```

Subsequent terminals attach to the same running container:
```bash
sudo docker exec -it <container_name> bash
```
(`docker ps` to find the name, typically `isaac_ros_dev-*`).

Inside the container the workspace is mounted at `/workspaces/isaac_ros-dev`.

---

## 3. Build the workspace (inside the container)

First-time build, or after adding a new Python entry point:

```bash
./src/mingsong_turtle_try/build_isaac_ros.sh
```

which runs:
```bash
cd /workspaces/isaac_ros-dev && \
  colcon build --symlink-install && \
  source install/setup.bash
```

If a build fails due to stale `--symlink-install` state, clear and rebuild:
```bash
rm -rf build/* install/* log/*
colcon build --symlink-install
source install/setup.bash
```

Every new terminal must source the overlay:
```bash
source /workspaces/isaac_ros-dev/install/setup.bash
```

### Adding a new node
1. Drop the Python file into `src/mingsong_turtle_try/src/motor_srv/motor_srv/`.
2. Add an entry point in `src/mingsong_turtle_try/src/motor_srv/setup.py` under `console_scripts`.
3. Rebuild with `build_isaac_ros.sh`.

---

## 4. Running the robot — standard startup sequence

Each step runs in its own terminal attached to the container. Source `install/setup.bash` in every terminal before running nodes.

All operational nodes live in the package `motor_srv` (source: `src/mingsong_turtle_try/src/motor_srv/motor_srv/`).

### Step 0 — Reset the robot to its home pose
Zeros out the Dynamixels and brings the joints to a known starting configuration.
```bash
ros2 run motor_srv sync_read_write
```

### Step 1 — Start the IMU (RealSense)
Must be launched first, before any node that consumes IMU data (RL policy, COT, etc.).
```bash
ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true
```

### Step 2 — Arduino sensor bridge (energy, GPS, pressure)
Reads battery current/voltage, GPS, and pressure from the Arduino and publishes them as ROS topics.
```bash
ros2 run motor_srv esteban_arduino_node
```

### Step 3 — COT (Cost of Transport) computation
Subscribes to motor status + power topics and computes the COT while the robot moves.
```bash
ros2 run motor_srv COT_jue
```

### Step 4 — Run the robot (RL policy)
Loads the trained policy and drives the motors. Requires the IMU (Step 1) to already be publishing.
```bash
ros2 run motor_srv RL_implementation
```

### Manual control (alternative to the RL policy)
Drive the robot from the keyboard instead of using the trained policy:
```bash
ros2 run motor_srv keyboard_control_node
```

---

## 5. Quick reference — node → purpose

| Command | File | Purpose |
|---|---|---|
| `ros2 run motor_srv sync_read_write` | `sync_read_write.py` | Reset robot to home pose |
| `ros2 run motor_srv keyboard_control_node` | `keyboard_control_node.py` | Manual keyboard teleop |
| `ros2 run motor_srv esteban_arduino_node` | `esteban_arduino_node.py` | Arduino: energy / GPS / pressure |
| `ros2 run motor_srv COT_jue` | `COT_jue.py` | Cost-of-transport calculation |
| `ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true` | — | RealSense IMU (gyro + accel) |
| `ros2 run motor_srv RL_implementation` | `RL_implementation.py` | Run trained RL policy on hardware |

---

## 6. Recording data
```bash
ros2 bag record -O /rosbag/test.bag /power /visual_slam/tracking/vo_pose
```
Convert to CSV after: see `rosbag2csv.py` at the workspace root.

---

## 7. Troubleshooting

- **No `/dev/ttyUSB*` visible** — check USB cable; `lsusb` should list the U2D2 / Arduino. Re-apply `chmod 777`.
- **RealSense not detected** — install the udev rules once: copy `99-realsense-libusb.rules` to `/etc/udev/rules.d/`, then `sudo udevadm control --reload-rules && sudo udevadm trigger`.
- **Build fails with "symlink already exists"** — clear `build/`, `install/`, `log/` and rebuild.
- **Node not found by `ros2 run`** — you added a file but didn't register it in `setup.py`, or forgot to rebuild. Rerun `build_isaac_ros.sh`.
