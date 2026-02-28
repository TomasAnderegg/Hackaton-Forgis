#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /zivid_ws/install/setup.bash

echo "Starting Zivid ROS 2 driver..."
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"

ros2 launch zivid_camera zivid_camera.launch.py
