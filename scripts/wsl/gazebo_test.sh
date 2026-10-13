#!/usr/bin/env bash
# Direct test in the native Create 3 simulator (Gazebo Harmonic, create3_sim's maze world):
# start the sim and the Laya task node, unpause, then send plain-English tasks one by one.
#
#   scripts/wsl/gazebo_test.sh
#   scripts/wsl/gazebo_test.sh "go to the east hall" "turn around" "return to the dock"
#   DRIVER=laya scripts/wsl/gazebo_test.sh "go to the south room"
#   KEEP_OPEN=1 scripts/wsl/gazebo_test.sh       # leave Gazebo running afterwards
#
# The Laya server must be reachable at $LAYA_URL (default http://127.0.0.1:8000; run
# scripts\serve_laya.ps1 on Windows, which WSL reaches through mirrored networking).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/ros_env.sh"     # ROS + workspace + CycloneDDS on loopback (see ros_env.sh)
source "${HERE}/launch_ctl.sh"  # start_launch / stop_launch

LAYA_URL="${LAYA_URL:-http://127.0.0.1:8000}"
DRIVER="${DRIVER:-code}"
# Wheel odometry drifts ~0.3 m over a 20 m mission in Gazebo (slip), enough to steer a
# map-planned route into a wall corner. The test uses the simulator's ground-truth pose by
# default; POSE_SOURCE=odom shows what the real, lidar-less Create 3 would do.
POSE_SOURCE="${POSE_SOURCE:-ground_truth}"
LOG="${LOG:-/tmp/create3_laya_launch.log}"
TASKS=()
for t in "$@"; do if [ -n "${t// }" ]; then TASKS+=("$t"); fi; done   # drop empty args
[ ${#TASKS[@]} -gt 0 ] || TASKS=("go to the east hall, then come back to the dock")

probe='{"state": "ping", "questions": {"p": {"type": "choice", "instructions": "Is this a test?",
        "criteria": {"yes": "yes", "no": "no"}}}}'
if ! curl -fsS -m 5 -X POST "${LAYA_URL}/v1/systemone" -H 'Content-Type: application/json' \
     -d "${probe}" >/dev/null; then
  echo "Laya is not reachable at ${LAYA_URL}. Start it on Windows (scripts\\serve_laya.ps1)" >&2
  echo "and make sure %UserProfile%\\.wslconfig has networkingMode=mirrored (then wsl --shutdown)." >&2
  exit 1
fi
echo "Laya reachable at ${LAYA_URL}"

echo "starting create3_sim (maze) + laya_create3; log: ${LOG}"
echo "pose source: ${POSE_SOURCE}"
start_launch "${LOG}" create3_laya laya_create3.launch.py use_rviz:=false driver:="${DRIVER}" \
  pose_source:="${POSE_SOURCE}" laya_url:="${LAYA_URL}"
cleanup() { if [ -z "${KEEP_OPEN}" ]; then echo "stopping the simulation"; stop_launch; fi; }
trap cleanup EXIT

pkill -f _ros2_daemon 2>/dev/null || true   # the ROS CLI daemon hangs under mirrored networking

unpause_world 120 || { echo "see ${LOG}" >&2; exit 1; }
echo "waiting for the robot and the Laya node"
python3 "${HERE}/wait_ready.py" --timeout 240 || { echo "not ready; see ${LOG}" >&2; exit 1; }

for task in "${TASKS[@]}"; do
  echo
  echo ">>> ${task}"
  ros2 run create3_laya laya_task "${task}"
done

echo
echo "final odometry:"
timeout 10 ros2 topic echo --no-daemon --once --field pose.pose.position /odom || true
if [ -n "${KEEP_OPEN}" ]; then
  echo "Gazebo left running (launch pid ${launch_pid})."
fi
