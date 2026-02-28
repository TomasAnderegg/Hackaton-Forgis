FROM ros:humble

# Install Zivid SDK from official apt repository
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://downloads.zivid.com/sdk/releases/2.13.1+18e79e79-1/u2204/amd64/ZividSDK-ubuntu2204-amd64-2.13.1+18e79e79-1.deb \
    -o /tmp/ZividSDK.deb && \
    apt-get install -y /tmp/ZividSDK.deb && \
    rm /tmp/ZividSDK.deb && \
    rm -rf /var/lib/apt/lists/*

# Install ROS 2 build tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Build zivid_ros from source (official Zivid ROS 2 driver)
RUN mkdir -p /zivid_ws/src && \
    cd /zivid_ws/src && \
    git clone --depth 1 --branch humble https://github.com/zivid/zivid-ros.git && \
    cd /zivid_ws && \
    . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install

COPY scripts/ ./scripts/
RUN sed -i 's/\r$//' scripts/*.sh && chmod +x scripts/*.sh

CMD ["bash", "scripts/start_driver.sh"]
