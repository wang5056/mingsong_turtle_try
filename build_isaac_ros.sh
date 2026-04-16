# build the workspace from the top
cd /workspaces/isaac_ros-dev && \
  colcon build --symlink-install && \
  source install/setup.bash