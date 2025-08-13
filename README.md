## Setup

Prereq: You need Linux on Wayland (X11 works yoo but tweak the image start).

1. If you are on NixOS, modify your config:
```
hardware.graphics.enable = true;
virtualisation.docker.enable = true;
```

2. Install all the packages listed in `shell.nix` or, if you are on NixOS, just run `nix-shell`.

3. Make sure you are in a docker group by running `groups` in your terminal. If you are not there, add yourself and reboot (NixOS rebuild won't help btw, don't make my mistakes!).

4. Pull the PX4 submodule. It contains the required worlds and models. 
```
git submodule update --init --recursive
```

5. Autorize your DRM device. This command should print `non-network local connections being added to access control list`.
```
$ xhost +local:
```

6. Pull the docker image. 
```
docker pull gazebo:latest
```

7. Run docker image. Please read the command and modify it slightly to fit your DRI (card index and render number) and paths.
```
$ docker run --rm -it  --net=host \
    -e DISPLAY -e XDG_RUNTIME_DIR \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
    -v /path/to/this/repo/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:/root/.gazebo/models  -v /path/to/this/repo/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds:/root/worlds  \
    --device /dev/dri/card1   --device /dev/dri/renderD128 \
    gazebo:latest gazebo /root/worlds/iris.world
```

This should start your Gazebo-Classic and you can add models there.
