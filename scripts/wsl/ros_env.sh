# Source this in WSL before using ROS for create3-laya:  source scripts/wsl/ros_env.sh
# CycloneDDS on loopback: Fast DDS discovery does not work under WSL's mirrored networking,
# which is what lets WSL reach the Laya server on Windows at 127.0.0.1.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
[ -f "${HOME}/create3_ws/install/local_setup.bash" ] && source "${HOME}/create3_ws/install/local_setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${_here}/cyclonedds_localhost.xml"
# A loopback address that mirrored mode leaves on Linux (127.0.0.1 detours through Windows;
# see the XML): add 127.0.0.2 after a WSL restart, else take any other address on lo.
if ! ip -4 -o addr show dev lo | grep -q ' 127\.0\.0\.2/'; then
  sudo -n ip addr add 127.0.0.2/32 dev lo 2>/dev/null || true
fi
CREATE3_DDS_ADDRESS="$(ip -4 -o addr show dev lo | awk '{sub(/\/.*/, "", $4); print $4}' |
  grep -v '^127\.0\.0\.1$' | sort -r | head -n 1)"
if [ -n "${CREATE3_DDS_ADDRESS}" ]; then
  export CREATE3_DDS_ADDRESS CREATE3_DDS_MAX_MESSAGE=65000B CREATE3_DDS_FRAGMENT=60000B
else
  unset CREATE3_DDS_ADDRESS
  echo "ros_env: no second loopback address; ROS traffic takes the slow path via Windows" >&2
fi
# WSLg's Mesa falls back to llvmpipe (CPU) unless told to use D3D12, and then picks the first
# adapter (an iGPU on hybrid laptops). Gazebo renders every IR/lidar/camera sensor, so this
# decides whether the simulation runs near real time or at a few percent of it.
if [ -e /dev/dxg ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
fi
unset _here
