#!/bin/bash
CONTAINER_NAME=math-user:local
docker run --rm --name math-user-temp -v $(pwd):/tmp --workdir /tmp ${CONTAINER_NAME} make && make clean
#docker run --rm --name math-user-temp -v $(pwd):/tmp --workdir /tmp $(docker images --format "table {{.Repository}}:{{.Tag}}" | fzf) make && make clean
