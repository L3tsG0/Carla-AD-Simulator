#!/usr/bin/env bash

PORT="${1:-2000}"

docker run \
  --gpus all \
  --net=host \
  -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name carla-server \
  carlasim/carla:0.9.15 \
  ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port="$PORT" -quality-level=Low -opengl
