#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Add the build folder so Gazebo can find your .so libraries
export GAZEBO_PLUGIN_PATH="$(pwd)/build:${GAZEBO_PLUGIN_PATH}"
export GAZEBO_MODEL_PATH="$(pwd)/models:${GAZEBO_MODEL_PATH}"

gazebo worlds/defense.world
