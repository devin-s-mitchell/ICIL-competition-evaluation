#!/bin/bash
set -e
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
exec "$@"
