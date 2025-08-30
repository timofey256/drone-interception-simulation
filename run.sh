#!/bin/sh

xhost +local:root
docker run --rm -it --net=host -e DISPLAY -e XDG_RUNTIME_DIR   -v /tmp/.X11-unix:/tmp/.X11-unix   -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR   -v ./plugins:/root/gazebo-plugins   -v ./models:/root/.gazebo/models   -v ./worlds:/root/worlds   -e GAZEBO_PLUGIN_PATH=/root/gazebo-plugins/build   --device /dev/dri/card1   --device /dev/dri/renderD128   gazebo-dev   gazebo /root/worlds/shahed_test.world --verbose
