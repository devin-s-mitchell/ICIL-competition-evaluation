#!/bin/bash
set -e
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec "$@"
