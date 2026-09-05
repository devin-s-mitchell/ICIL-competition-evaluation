#!/bin/bash
# Build the model-side image. Usage: docker/build.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-icilval/model:dev}
rm -rf dist && python -m pip wheel --no-deps -w dist . >/dev/null
docker build -f docker/Dockerfile -t "$TAG" .
echo "built $TAG"
