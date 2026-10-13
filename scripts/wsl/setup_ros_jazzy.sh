#!/usr/bin/env bash
# One-time (idempotent) setup inside WSL2 Ubuntu 24.04: ROS 2 Jazzy, Gazebo Harmonic, and a
# colcon workspace with create3_sim + irobot_create_msgs + this repo's create3_laya package.
#
# From Windows:
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/c/<path to>/create3-laya/scripts/wsl/setup_ros_jazzy.sh"
#
# Creates a passwordless sudo user (default "ros") inside the VM so Gazebo's GUI runs as a
# normal user, and makes it the distro's default user.
set -euo pipefail

USER_NAME="${CREATE3_USER:-ros}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="/home/${USER_NAME}/create3_ws"
export DEBIAN_FRONTEND=noninteractive

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: wsl -d Ubuntu-24.04 -u root -- bash $0" >&2
  exit 1
fi

echo "== user ${USER_NAME}"
if ! id "${USER_NAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo,video,render "${USER_NAME}" 2>/dev/null \
    || useradd -m -s /bin/bash -G sudo "${USER_NAME}"
fi
echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${USER_NAME}"
chmod 440 "/etc/sudoers.d/90-${USER_NAME}"
if ! grep -q "^default=${USER_NAME}" /etc/wsl.conf 2>/dev/null; then
  printf '\n[user]\ndefault=%s\n' "${USER_NAME}" >> /etc/wsl.conf
fi

echo "== apt: ROS 2 Jazzy + Gazebo Harmonic"
apt-get update -q
apt-get install -y -q locales curl software-properties-common git
locale-gen en_US.UTF-8 >/dev/null
add-apt-repository -y universe >/dev/null
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  version="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  curl -fsSL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${codename}_all.deb"
  dpkg -i /tmp/ros2-apt-source.deb
fi
apt-get update -q
apt-get install -y -q ros-jazzy-desktop ros-jazzy-ros-gz ros-dev-tools ros-jazzy-rmw-cyclonedds-cpp
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || rosdep init
if [ -n "${WITH_TB4}" ]; then
  # TurtleBot 4 = Create 3 + OAK-D camera + RPLIDAR; adds SLAM and Nav2 for the vision tasks.
  echo "== apt: TurtleBot 4 simulator + navigation"
  apt-get install -y -q ros-jazzy-turtlebot4-simulator ros-jazzy-turtlebot4-navigation \
    ros-jazzy-turtlebot4-description ros-jazzy-nav2-msgs
fi

echo "== workspace ${WS}"
as_user() { sudo -u "${USER_NAME}" -H bash -c "$1"; }
as_user "mkdir -p '${WS}/src'"
as_user "cd '${WS}/src' && { [ -d create3_sim ] || git clone -q -b jazzy https://github.com/iRobotEducation/create3_sim.git; }"
as_user "cd '${WS}/src' && { [ -d irobot_create_msgs ] || git clone -q -b jazzy https://github.com/iRobotEducation/irobot_create_msgs.git; }"
as_user "ln -sfn '${REPO_DIR}/create3_laya' '${WS}/src/create3_laya'"
# create3_sim (jazzy) bug: maze.sdf names its world 'default', but the ROS-Gazebo bridges build
# sensor topics from the launch argument ('/world/maze/...'), so IR, bumper and cliff data
# never reach ROS in the maze. Rename the world to match.
as_user "sed -i \"s/<world name='default'>/<world name='maze'>/\" \
  '${WS}/src/create3_sim/irobot_create_gz/irobot_create_gz_bringup/worlds/maze.sdf'"
# create3_sim's IMU plugin is Gazebo Classic only: under Harmonic the robot's gyro publishes
# nothing unless the world loads Gazebo's IMU system (laya_create3.launch.py bridges it to
# /imu). Without it, turns rely on wheel odometry, which reads ~7% more turning than happens.
maze_world="${WS}/src/create3_sim/irobot_create_gz/irobot_create_gz_bringup/worlds/maze.sdf"
grep -q 'gz-sim-imu-system' "${maze_world}" || sudo -u "${USER_NAME}" sed -i \
  "s#<world name='maze'>#&\n        <plugin filename='gz-sim-imu-system' name='gz::sim::systems::Imu'/>#" \
  "${maze_world}"
# Physics at 333 Hz, as the TurtleBot 4 worlds run the same Create 3 base: the maze's 1 kHz
# made physics its main cost (0.58x real time in WSL; 0.82x at 333 Hz, shadows irrelevant).
sudo -u "${USER_NAME}" sed -i -e 's#<max_step_size>0.001</max_step_size>#<max_step_size>0.003</max_step_size>#' \
  -e 's#<real_time_update_rate>1000</real_time_update_rate>#<real_time_update_rate>333</real_time_update_rate>#' \
  "${maze_world}"

# Simulated sensor rates. Gazebo renders each of the Create 3's 7 IR and 4 cliff sensors as a
# GPU lidar at 62 Hz, the TurtleBot 4's RPLIDAR at 62 Hz and its camera at 30 Hz, and the
# simulation waits for each render; under WSLg every render is a GPU round trip, which held
# the TurtleBot 4 depot sim at 0.03-0.06x real time. 20 Hz IR/cliff (the Laya safety loop runs
# at 20 Hz), 10 Hz lidar (a real RPLIDAR A1 spins at 5.5-10 Hz) and a 10 Hz camera brought it
# to ~0.75x; leaving out the sensors a launch doesn't use (the TurtleBot 4 navigates by its
# lidar; the maze has no drops) brought the depot to the real-time cap.
set_rate() {  # file sed-expression expected-text
  sudo -u "${USER_NAME}" sed -i "$2" "$1"
  grep -qF "$3" "$1" || echo "WARNING: sensor rate patch did not apply to $1" >&2
}
sudo -u "${USER_NAME}" bash "${REPO_DIR}/scripts/wsl/patch_create3_sim.sh" "${WS}/src/create3_sim"
if [ -n "${WITH_TB4}" ]; then
  # turtlebot4_description from source, overlaying the apt package, with the lidar and camera
  # rates lowered; only that package is built from this repo.
  as_user "cd '${WS}/src' && { [ -d turtlebot4 ] || git clone -q -b jazzy https://github.com/turtlebot/turtlebot4.git; }"
  for p in turtlebot4_msgs turtlebot4_navigation turtlebot4_node; do
    as_user "[ ! -d '${WS}/src/turtlebot4/${p}' ] || touch '${WS}/src/turtlebot4/${p}/COLCON_IGNORE'"
  done
  tb4_sensors="${WS}/src/turtlebot4/turtlebot4_description/urdf/sensors"
  set_rate "${tb4_sensors}/rplidar.urdf.xacro" 's/update_rate="62\.0"/update_rate="10.0"/' \
    'update_rate="10.0"'
  set_rate "${tb4_sensors}/oakd.urdf.xacro" 's#<update_rate>30</update_rate>#<update_rate>10</update_rate>#' \
    '<update_rate>10</update_rate>'
fi

as_user "rosdep update -q"
as_user "source /opt/ros/jazzy/setup.bash && cd '${WS}' && rosdep install --from-paths src -yi --rosdistro jazzy -q"

echo "== build"
as_user "source /opt/ros/jazzy/setup.bash && cd '${WS}' && colcon build --symlink-install --event-handlers console_cohesion- 2>&1 | tail -n 25"

# ros_env.sh: ROS + workspace + CycloneDDS on loopback (Fast DDS discovery fails under WSL's
# mirrored networking).
line="source '${REPO_DIR}/scripts/wsl/ros_env.sh'"
as_user "grep -qxF \"${line}\" ~/.bashrc || echo \"${line}\" >> ~/.bashrc"
echo "== done: open the distro (wsl -d Ubuntu-24.04) and run scripts/wsl/gazebo_test.sh"
