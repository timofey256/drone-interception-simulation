#!/usr/bin/env bash
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Choose X11 (default) or Wayland. Your current xhost +local: suggests X11.
# X11 path (works on Wayland via XWayland too):
DISPLAY_FLAGS="-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"

# Wayland alternative (uncomment if you go native Wayland):
# DISPLAY_FLAGS="-e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR"

# Adjust your DRI devices to match your system (cardN, renderDXXX)
DRI_FLAGS="--device /dev/dri/card1 --device /dev/dri/renderD128"

docker run --rm -it --net=host \
  ${DISPLAY_FLAGS} \
  -v "${REPO_DIR}/models:/root/.gazebo/models" \
  -v "${REPO_DIR}/worlds:/root/worlds" \
  -v "${REPO_DIR}:/root/workspace" \
  ${DRI_FLAGS} \
  gazebo:latest /bin/bash
