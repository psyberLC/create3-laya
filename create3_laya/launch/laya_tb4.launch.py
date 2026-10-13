"""TurtleBot 4 (Create 3 + OAK-D camera + RPLIDAR) in Gazebo, with SLAM, Nav2 and the Laya node.

No named places here: tasks use what the robot can see ("go to where the pallets are"),
odometry ("go 3 meters forward"), and what it has seen before ("go back to the pallets").

  ros2 launch create3_laya laya_tb4.launch.py                 # depot world
  ros2 launch create3_laya laya_tb4.launch.py world:=warehouse
  ros2 launch create3_laya laya_tb4.launch.py gui:=false      # Gazebo server only (faster)

Start Laya (scripts/serve_laya.ps1) and the vision server (vision/serve_vision.py) first.
"""

import os
import re
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = [
    DeclareLaunchArgument('world', default_value='depot',
                          description='turtlebot4_gz_bringup world: depot, warehouse, maze'),
    DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite']),
    DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                          description='Gazebo window; false runs the server alone'),
    DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
    DeclareLaunchArgument('driver', default_value='code', choices=['code', 'laya']),
    DeclareLaunchArgument('laya_url', default_value='http://127.0.0.1:8000'),
    DeclareLaunchArgument('vision_url', default_value='http://127.0.0.1:8001'),
    DeclareLaunchArgument('task_model', default_value='typed-decisions'),
    DeclareLaunchArgument('driving_model', default_value='create3-driving'),
] + [DeclareLaunchArgument(axis, default_value='0.0', description=f'Spawn {axis}')
     for axis in ('x', 'y', 'yaw')]


IMU_SYSTEM = '<plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>'


def _world_file(world):
    """turtlebot4_gz_bringup's world plus Gazebo's IMU system, as a copy in /tmp.

    create3_sim's IMU plugin exists only for Gazebo Classic, so under Harmonic the robot's
    gyro stays silent unless the world loads the IMU system; turns then rely on wheel
    odometry, which came out 84 deg for 90 in the depot. The world keeps its name, so every
    topic stays the same."""
    source = os.path.join(get_package_share_directory('turtlebot4_gz_bringup'), 'worlds',
                          f'{world}.sdf')
    with open(source, encoding='utf-8') as f:
        sdf = f.read()
    if 'gz-sim-imu-system' not in sdf:
        sdf = re.sub(r'(<world\b[^>]*>)', r'\1\n    ' + IMU_SYSTEM, sdf, count=1)
    path = os.path.join(tempfile.gettempdir(), 'create3_laya', f'{world}.sdf')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sdf)
    return path


def _gazebo(context):
    """What turtlebot4_gz_bringup's sim.launch.py does, on our copy of the world, with the
    IMU bridged to /imu; gui:=false runs the server alone."""
    tb4_bringup = get_package_share_directory('turtlebot4_gz_bringup')
    world = LaunchConfiguration('world').perform(context)
    shares = [get_package_share_directory(p) for p in (
        'turtlebot4_description', 'irobot_create_description', 'irobot_create_gz_bringup')]
    resources = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', ':'.join([
        os.path.join(tb4_bringup, 'worlds'), os.path.join(shares[2], 'worlds'),
        str(Path(shares[0]).parent.resolve()), str(Path(shares[1]).parent.resolve())]))
    gui_plugins = SetEnvironmentVariable('GZ_GUI_PLUGIN_PATH', ':'.join(
        os.path.join(get_package_share_directory(p), 'lib')
        for p in ('turtlebot4_gz_gui_plugins', 'irobot_create_gz_plugins')))
    if LaunchConfiguration('gui').perform(context) == 'true':
        window = ' --gui-config ' + os.path.join(
            tb4_bringup, 'gui', LaunchConfiguration('model').perform(context), 'gui.config')
    else:
        window = ' -s'
    server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments=[('gz_args', _world_file(world) + ' -r -v 2' + window)])
    # Sim time at 100 Hz rather than every 3 ms physics step (see clock_relay.py).
    clock = Node(package='create3_laya', executable='clock_relay', output='screen',
                 parameters=[{'rate': 100.0}])
    imu = f'/world/{world}/model/turtlebot4/link/imu_link/sensor/imu/imu'
    imu_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='imu_bridge',
                      output='screen', arguments=[f'{imu}@sensor_msgs/msg/Imu[gz.msgs.IMU'],
                      remappings=[(imu, 'imu')], parameters=[{'use_sim_time': True}])
    return [resources, gui_plugins, server, clock, imu_bridge]


def generate_launch_description():
    tb4_bringup = get_package_share_directory('turtlebot4_gz_bringup')
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [tb4_bringup, 'launch', 'turtlebot4_spawn.launch.py'])),
        launch_arguments=[('model', LaunchConfiguration('model')),
                          ('rviz', LaunchConfiguration('rviz')),
                          ('x', LaunchConfiguration('x')), ('y', LaunchConfiguration('y')),
                          ('yaw', LaunchConfiguration('yaw')),
                          # turtlebot4_spawn starts SLAM and Nav2 for these
                          ('slam', 'true'), ('nav2', 'true')])

    laya_node = Node(
        package='create3_laya',
        executable='laya_create3',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'driver': LaunchConfiguration('driver'),
            'laya_url': LaunchConfiguration('laya_url'),
            'task_model': LaunchConfiguration('task_model'),
            'driving_model': LaunchConfiguration('driving_model'),
            'places_file': 'none',
            'world_sdf': 'none',
            'dock_in_map': False,
            'nav_backend': 'nav2',
            # Goals in SLAM's map, which (unlike odom) agrees with the gyro-aided pose.
            'nav_frame': 'map',
            'camera': True,
            # No IR sensors in this model (see the end of this file): read them off the lidar.
            'ir_from_lidar': True,
            'vision_url': LaunchConfiguration('vision_url'),
            'spawn_x': ParameterValue(LaunchConfiguration('x'), value_type=float),
            'spawn_y': ParameterValue(LaunchConfiguration('y'), value_type=float),
            'spawn_yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float),
        }])

    # The Create 3's IR and cliff sensors are GPU lidars that Gazebo renders while the
    # simulation waits; this robot navigates by its own lidar, so leave them out of the model
    # (scripts/wsl/patch_ray_sensors.sh makes the rates switchable).
    no_ray_sensors = [SetEnvironmentVariable('CREATE3_IR_RATE', '0'),
                      SetEnvironmentVariable('CREATE3_CLIFF_RATE', '0')]
    return LaunchDescription(ARGUMENTS + no_ray_sensors + [OpaqueFunction(function=_gazebo),
                                                           robot, laya_node])
