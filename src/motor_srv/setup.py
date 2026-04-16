from setuptools import setup
import os
from glob import glob

package_name = 'motor_srv'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),  # Package resource index
        ('share/' + package_name, ['package.xml']),  # Package.xml file
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),  # Install launch files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),  # Install config files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_service = motor_srv.motor_srv_func:main',
            'motor_client = motor_srv.motor_cli_func:main',
            'joystick = motor_srv.motor_cli_func_joy:main',
            'robot_status_sub = motor_srv.robot_status_sub:main',
            'arduino_node = motor_srv.arduino_node:main',
            'mock_joystick =motor_srv.mock_joystick:main',
            'instant_cot =motor_srv.robot_status_sub_instant_COT:main',
            'sync_read_write = motor_srv.sync_read_write:main',
            'goal_publisher= motor_srv.goal_publisher:main',
            'imu_processor= motor_srv.imu_processor:main',
            'imu_processor_kalman= motor_srv.imu_processor_kalman:main',
            'imu_rgbd_fused= motor_srv.imu_rgbd_fused:main',
            'imu_rgbd_publisher = motor_srv.imu_rgbd_publisher:main',
            'keyboard_control_node = motor_srv.keyboard_control_node:main',
            'feedback_RL = motor_srv.feedback_RL:main',
            'RL_implementation = motor_srv.RL_implementation:main',
            "sinusoidal_goal_publisher = motor_srv.sinusoidal_goal_publisher:main",
            'RL_new = motor_srv.RL_new:main',
            'jue_arduino_node = motor_srv.jue_arduino_node:main',
            'robot_motion_tracker = motor_srv.robot_motion_tracker:main',
            'COT_jue = motor_srv.COT_jue:main',
            'esteban_arduino_node = motor_srv.esteban_arduino_node:main',
            'RL_pre_gait = motor_srv.RL_pre_gait:main'
        ],
    },
)
