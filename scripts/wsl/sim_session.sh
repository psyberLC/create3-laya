#!/usr/bin/env bash
# One simulator session for the Windows launcher (launcher/app.py): start Gazebo with the robot
# and the Laya node, unpause, wait until the robot can drive, print "@@ready", then send each
# stdin line as a task and print every task_status message as "@@<json>". End of input (the
# launcher closing the pipe) or "quit" stops the simulation.
#
#   scripts/wsl/sim_session.sh create3            # Create 3 in create3_sim's maze
#   GUI=false scripts/wsl/sim_session.sh tb4      # TurtleBot 4 in the depot, no Gazebo window
#   scripts/wsl/sim_session.sh speed 2            # run the running simulation at 2x real time
#   scripts/wsl/sim_session.sh stop               # stop a session something left running
#
# Environment: DRIVER (code|laya), LAYA_URL, VISION_URL, POSE_SOURCE (create3), WORLD and GUI
# (tb4). The Laya server (and, for tb4, the vision server) must already be running.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/ros_env.sh"
source "${HERE}/launch_ctl.sh"

LAYA_URL="${LAYA_URL:-http://127.0.0.1:8000}"
VISION_URL="${VISION_URL:-http://127.0.0.1:8001}"
DRIVER="${DRIVER:-code}"

fail() {  # message: shown in the launcher's conversation, then exit
  python3 -c 'import json, sys; print("@@" + json.dumps({"event": "error", "detail": sys.argv[1]}))' "$1"
  echo "$1" >&2
  exit 1
}

# Only one simulation fits the GPU (and Gazebo's ports): clear one a crashed caller left.
if [ "${1:-}" = create3 ] || [ "${1:-}" = tb4 ]; then
  stop_all_launches >/dev/null 2>&1 || true
  # Gazebo's window needs WSLg. It starts with WSL, and if it crashes (it can, on a GPU
  # reset) Gazebo dies with "no Qt platform plugin" until WSL restarts.
  if [ "${1}" = create3 ] || [ "${GUI:-true}" != false ]; then
    for _ in $(seq 1 20); do [ -S /tmp/.X11-unix/X0 ] && break; sleep 1; done
    [ -S /tmp/.X11-unix/X0 ] || fail "WSL's display (WSLg) is not running, so the Gazebo window \
can't open. In Windows, run 'wsl --shutdown', then start again (or untick 'Show the Gazebo \
window' for the TurtleBot 4)."
  fi
fi

case "${1:-}" in
  create3)
    LOG=/tmp/create3_laya_launch.log
    echo "starting create3_sim (maze) with the Laya node; log: ${LOG}"
    start_launch "${LOG}" create3_laya laya_create3.launch.py use_rviz:=false \
      driver:="${DRIVER}" pose_source:="${POSE_SOURCE:-ground_truth}" laya_url:="${LAYA_URL}"
    ready_args=(--timeout 240)
    ;;
  tb4)
    LOG=/tmp/create3_laya_tb4.log
    echo "starting the TurtleBot 4 (${WORLD:-depot}) with SLAM, Nav2 and the Laya node; log: ${LOG}"
    start_launch "${LOG}" create3_laya laya_tb4.launch.py world:="${WORLD:-depot}" \
      driver:="${DRIVER}" gui:="${GUI:-true}" laya_url:="${LAYA_URL}" vision_url:="${VISION_URL}"
    ready_args=(--timeout 300 --nav2 --min-ir 0)   # no IR: see laya_tb4.launch.py
    ;;
  speed)
    # Gazebo runs as fast as it can up to this multiple of real time: in WSL about 2x with
    # the window, 3.5x without (TurtleBot 4 depot).
    factor="${2:?usage: sim_session.sh speed <factor>}"
    world="$(gz service -l | grep -m1 -oE '^/world/[^/]+/set_physics$' | sed 's#/set_physics$##')"
    [ -n "${world}" ] || { echo "no simulation running" >&2; exit 1; }
    # The physics step stays as the world set it (3 ms in both worlds here).
    step="$(timeout 5 gz topic -e -t "${world}/stats" -n 1 2>/dev/null |
      awk '/step_size/ {s = 1} s && /nsec/ {print $2 / 1e9; exit}')"
    gz service -s "${world}/set_physics" --reqtype gz.msgs.Physics --reptype gz.msgs.Boolean \
      --timeout 5000 --req "max_step_size: ${step:-0.003}, real_time_factor: ${factor}" |
      grep -q 'data: true' || { echo "Gazebo did not accept the speed" >&2; exit 1; }
    echo "simulation speed set to ${factor}x real time (${world#/world/})"
    exit 0
    ;;
  stop)
    stop_all_launches
    echo "no simulation running"
    exit 0
    ;;
  *)
    sed -n '2,13p' "$0" >&2
    exit 2
    ;;
esac
trap 'echo "stopping the simulation"; stop_launch' EXIT
trap 'exit 130' INT TERM HUP

pkill -f _ros2_daemon 2>/dev/null || true   # the ROS CLI daemon hangs under mirrored networking
unpause_world 180
echo "waiting for the robot to be ready"
python3 "${HERE}/wait_ready.py" "${ready_args[@]}" || { echo "not ready; see ${LOG}" >&2; exit 1; }
echo "@@ready"
ros2 run create3_laya laya_task --console
