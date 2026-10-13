#!/usr/bin/env bash
# usage: patch_create3_sim.sh <create3_sim source dir>
# Speed patches for create3_sim under Gazebo Harmonic in WSL. Idempotent: run again after
# pulling create3_sim, then rebuild (setup_ros_jazzy.sh does both).
#
# 1. IR and cliff sensors switchable per launch. Each is a GPU lidar, and the simulation stops
#    stepping while the sensors due at that moment render; under WSLg each render pass costs
#    tens of ms. 11 of them at 20 Hz kept the depot at 0.7x real time with stalls, although
#    the TurtleBot 4 navigates by its own lidar. Rates come from CREATE3_IR_RATE and
#    CREATE3_CLIFF_RATE (default 20 Hz); 0 leaves the sensor out of the model.
# 2. joint_state_broadcaster at 62 Hz instead of every physics step (333 Hz): wheel joint
#    states only feed TF and two helper nodes, and robot_state_publisher turned each one into
#    a /tf message for every listener.
# 3. The /clock bridge can be left out (CREATE3_CLOCK_BRIDGE=0) for create3_laya's
#    clock_relay, which passes sim time on at 100 Hz rather than every physics step.
set -e
src="${1:?usage: patch_create3_sim.sh <create3_sim source dir>}"
urdf="${src}/irobot_create_common/irobot_create_description/urdf"
control="${src}/irobot_create_common/irobot_create_control/config/control.yaml"
sim_launch="${src}/irobot_create_gz/irobot_create_gz_bringup/launch/sim.launch.py"
patch() {  # file sed-expression expected-text
  grep -qF "$3" "$1" || sed -i "$2" "$1"
  grep -qF "$3" "$1" || { echo "WARNING: create3_sim patch did not apply to $1" >&2; return 1; }
}
patch "${urdf}/sensors/ir_intensity.urdf.xacro" \
  's/update_rate:=\(62\|20\)\.0/update_rate:=$(optenv CREATE3_IR_RATE 20.0)/' \
  'update_rate:=$(optenv CREATE3_IR_RATE 20.0)'
patch "${urdf}/sensors/cliff_sensor.urdf.xacro" \
  's/\(name="update_rate" *value=\)"\(62\|20\)"/\1"$(optenv CREATE3_CLIFF_RATE 20)"/' \
  'value="$(optenv CREATE3_CLIFF_RATE 20)"'
# The ray_sensor macro's Gazebo Harmonic branch: no sensor at all for a rate of 0 (lidar,
# dock and docking-IR sensors pass their own non-zero rates and are unaffected).
patch "${urdf}/common_properties.urdf.xacro" \
  "s/value=\"\${gazebo == 'ignition'}\">/value=\"\${gazebo == 'ignition' and float(update_rate) > 0}\">/" \
  "gazebo == 'ignition' and float(update_rate) > 0"
patch "${sim_launch}" \
  "s/^    ld.add_action(clock_bridge)$/    if os.environ.get('CREATE3_CLOCK_BRIDGE', '1') != '0':  # create3-laya: clock_relay\n        ld.add_action(clock_bridge)/" \
  "os.environ.get('CREATE3_CLOCK_BRIDGE', '1') != '0'"
if ! grep -q 'create3-laya: joint states' "${control}"; then
  printf '\n  joint_state_broadcaster:\n    ros__parameters:\n      update_rate: 62  # create3-laya: joint states at the odometry rate\n' \
    >> "${control}"
fi
echo "create3_sim patched in ${src}"
