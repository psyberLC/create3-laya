"""create3_sim in the maze world plus the Laya task node.

  ros2 launch create3_laya laya_create3.launch.py
  ros2 launch create3_laya laya_create3.launch.py driver:=laya laya_url:=http://172.20.0.1:8000

Start the Laya server first (scripts/serve_laya.sh, or serve_laya.ps1 on the Windows host).
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGUMENTS = [
    DeclareLaunchArgument('world', default_value='maze',
                          description='create3_sim world; routes need one with local geometry'),
    DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
    DeclareLaunchArgument('use_rviz', default_value='true', choices=['true', 'false']),
    DeclareLaunchArgument('driver', default_value='code', choices=['code', 'laya'],
                          description='Who picks motions along the route'),
    DeclareLaunchArgument('laya_url', default_value='http://127.0.0.1:8000',
                          description='Local Laya server'),
    DeclareLaunchArgument('task_model', default_value='typed-decisions'),
    DeclareLaunchArgument('driving_model', default_value='create3-driving'),
    DeclareLaunchArgument('pose_source', default_value='odom', choices=['odom', 'ground_truth'],
                          description='Wheel odometry (drifts, like the real robot) or the '
                                      "simulator's ground-truth pose"),
    DeclareLaunchArgument('places_file', default_value='',
                          description='Named places JSON (default: the maze catalog)'),
] + [DeclareLaunchArgument(axis, default_value='0.0', description=f'Spawn {axis}')
     for axis in ('x', 'y', 'yaw')]


def generate_launch_description():
    gz_bringup = get_package_share_directory('irobot_create_gz_bringup')
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [gz_bringup, 'launch', 'create3_gz.launch.py'])),
        launch_arguments=[(name, LaunchConfiguration(name)) for name in
                          ('world', 'namespace', 'use_rviz', 'x', 'y', 'yaw')])

    laya_node = Node(
        package='create3_laya',
        executable='laya_create3',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'driver': LaunchConfiguration('driver'),
            'laya_url': LaunchConfiguration('laya_url'),
            'task_model': LaunchConfiguration('task_model'),
            'driving_model': LaunchConfiguration('driving_model'),
            'places_file': LaunchConfiguration('places_file'),
            'pose_source': LaunchConfiguration('pose_source'),
            'world_sdf': PathJoinSubstitution(
                [gz_bringup, 'worlds', [LaunchConfiguration('world'), '.sdf']]),
            'spawn_x': ParameterValue(LaunchConfiguration('x'), value_type=float),
            'spawn_y': ParameterValue(LaunchConfiguration('y'), value_type=float),
            'spawn_yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float),
        }])

    # The gyro: create3_sim's IMU plugin is Gazebo Classic only, so under Harmonic the setup
    # script adds Gazebo's IMU system to the maze and this bridges the sensor to <ns>/imu.
    imu = PathJoinSubstitution(['/world', LaunchConfiguration('world'),
                                'model/create3/link/imu_link/sensor/imu/imu'])
    imu_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='imu_bridge',
                      namespace=LaunchConfiguration('namespace'), output='screen',
                      arguments=[[imu, '@sensor_msgs/msg/Imu[gz.msgs.IMU']],
                      remappings=[(imu, 'imu')], parameters=[{'use_sim_time': True}])

    # Cliff sensors are GPU lidars Gazebo renders while the simulation waits, and the maze has
    # no drops; the IR sensors stay, the code driver steers by them. Sim time comes from
    # clock_relay at 100 Hz instead of create3_sim's bridge at every 3 ms physics step
    # (scripts/wsl/patch_create3_sim.sh adds both switches).
    speedups = [SetEnvironmentVariable('CREATE3_CLIFF_RATE', '0'),
                SetEnvironmentVariable('CREATE3_CLOCK_BRIDGE', '0')]
    clock = Node(package='create3_laya', executable='clock_relay', output='screen',
                 parameters=[{'rate': 100.0}])
    return LaunchDescription(ARGUMENTS + speedups + [simulator, clock, imu_bridge, laya_node])
