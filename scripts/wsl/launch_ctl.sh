# Start and stop a `ros2 launch` from a test script (source this file).
#
# A non-interactive shell starts background jobs with SIGINT ignored, so `kill -INT` never
# reached the launch and the test scripts hung on exit. Instead, run the launch as its own
# session and stop that whole session (every node and Gazebo) with SIGTERM, then SIGKILL.

start_launch() {  # start_launch <log> <ros2 launch args...>; sets launch_pid
  local log="$1"; shift
  setsid ros2 launch "$@" > "${log}" 2>&1 &
  launch_pid=$!
}

launch_running() {  # any live (non-zombie) process left in the launch's session?
  ps -o stat= -s "${launch_pid}" 2>/dev/null | grep -qv '^Z'
}

stop_launch() {  # stop_launch [grace seconds, default 20]
  [ -n "${launch_pid:-}" ] || return 0
  kill -TERM -- "-${launch_pid}" 2>/dev/null || true
  local t
  for ((t = 0; t < ${1:-20}; t++)); do launch_running || break; sleep 1; done
  kill -KILL -- "-${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true   # in case Gazebo left the session
}

stop_all_launches() {  # stop every create3_laya launch session, e.g. one a crashed caller left
  local pid
  for pid in $(pgrep -f "ros2 launch create3_laya" || true); do
    launch_pid="$(ps -o sid= -p "${pid}" | tr -d ' ')"
    [ -n "${launch_pid}" ] && stop_launch
  done
  pkill -f "gz sim" 2>/dev/null || true
}

unpause_world() {  # unpause_world [timeout s, default 120]
  # create3_sim starts Gazebo paused, and ros2_control can only activate the robot's
  # controllers while the simulation runs, so unpause as soon as the world is up.
  local i control=''
  for ((i = 0; i < ${1:-120}; i++)); do
    control="$(gz service -l 2>/dev/null | grep -m1 -E '^/world/[^/]+/control$' || true)"
    [ -n "${control}" ] && break
    launch_running || { echo "the launch exited before Gazebo came up" >&2; return 1; }
    sleep 1
  done
  [ -n "${control}" ] || { echo "timed out waiting for the Gazebo world" >&2; return 1; }
  echo "unpausing the simulation (${control})"
  gz service -s "${control}" --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'pause: false' >/dev/null
}
