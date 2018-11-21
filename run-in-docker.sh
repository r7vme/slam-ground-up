#!/bin/bash

docker run --net host \
    --rm \
    -ti \
    -e DISPLAY \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $HOME/sim:/sim \
    -v $(pwd):/code \
    -e QT_X11_NO_MITSHM=1 \
    -w /code \
    --device=/dev/dri:/dev/dri \
        slam-ground-up bash
