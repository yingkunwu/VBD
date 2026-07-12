WORKSPACE_DIR="."
IMAGE_NAME="vbd"
IMAGE_VERSION="latest"

container_name="vbd"
index=0

# Find next available index
while [ $(docker ps -a --filter name=^/${container_name}_${index}$ -q) ]; do
	((index++))
done

docker run -it --rm \
	--name ${container_name}_${index} \
	-v $WORKSPACE_DIR:/workspace \
	-v /mnt:/mnt \
	--gpus all \
	--pid host \
	--ipc=host \
	--net=host \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $HOME/.Xauthority:/root/.Xauthority:ro \
	$IMAGE_NAME:$IMAGE_VERSION /bin/bash
