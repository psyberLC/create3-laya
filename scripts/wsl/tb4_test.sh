#!/usr/bin/env bash
# Vision + odometry test on the TurtleBot 4 in Gazebo (depot world, no named places):
# start the sim with SLAM + Nav2 + the Laya node, unpause, then send plain-English tasks.
#
#   scripts/wsl/tb4_test.sh
#   scripts/wsl/tb4_test.sh "go to where the pallets are stored" "come back to where you started"
#   WORLD=warehouse KEEP_OPEN=1 scripts/wsl/tb4_test.sh
#
# Needs the Laya server (scripts\serve_laya.ps1) and the vision server
# (scripts\serve_vision.ps1) running on Windows.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/ros_env.sh"
source "${HERE}/launch_ctl.sh"

LAYA_URL="${LAYA_URL:-http://127.0.0.1:8000}"
VISION_URL="${VISION_URL:-http://127.0.0.1:8001}"
WORLD="${WORLD:-depot}"
DRIVER="${DRIVER:-code}"
GUI="${GUI:-true}"
LOG="${LOG:-/tmp/create3_laya_tb4.log}"
TASKS=()
for t in "$@"; do if [ -n "${t// }" ]; then TASKS+=("$t"); fi; done   # drop empty args
[ ${#TASKS[@]} -gt 0 ] || TASKS=("go 2 meters forward" "go to where the pallets are stored"
                                  "come back to where you started")

probe='{"state": "ping", "questions": {"p": {"type": "choice", "instructions": "Is this a test?",
        "criteria": {"yes": "yes", "no": "no"}}}}'
curl -fsS -m 5 -X POST "${LAYA_URL}/v1/systemone" -H 'Content-Type: application/json' \
  -d "${probe}" >/dev/null || { echo "Laya is not reachable at ${LAYA_URL}" >&2; exit 1; }
curl -fsS -m 5 "${VISION_URL}/v1/health" >/dev/null \
  || { echo "vision server is not reachable at ${VISION_URL} (scripts\\serve_vision.ps1)" >&2; exit 1; }
echo "Laya and vision servers reachable"

pkill -f _ros2_daemon 2>/dev/null || true
echo "starting TurtleBot 4 (${WORLD}) + SLAM + Nav2 + laya_create3; log: ${LOG}"
start_launch "${LOG}" create3_laya laya_tb4.launch.py world:="${WORLD}" driver:="${DRIVER}" \
  gui:="${GUI}" laya_url:="${LAYA_URL}" vision_url:="${VISION_URL}"
cleanup() { if [ -z "${KEEP_OPEN}" ]; then echo "stopping the simulation"; stop_launch; fi; }
trap cleanup EXIT

unpause_world 180 || { echo "see ${LOG}" >&2; exit 1; }

echo "waiting for the robot, the Laya node and Nav2"
python3 "${HERE}/wait_ready.py" --timeout 300 --nav2 --min-ir 0 || { echo "not ready; see ${LOG}" >&2; exit 1; }

for task in "${TASKS[@]}"; do
  echo
  echo ">>> ${task}"
  ros2 run create3_laya laya_task "${task}"
done

echo
echo "what the camera saw:"
grep -a "saw " "${LOG}" | sed 's/.*\]: //' | tail -n 10 || true
if [ -n "${KEEP_OPEN}" ]; then
  echo "Gazebo left running (launch pid ${launch_pid})."
fi
