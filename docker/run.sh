#!/usr/bin/env bash
#
# Start an instance of the humble-base docker container.
# See below or run this script with -h or --help to see usage options.
#
# This script should be run from the root dir of the project.
#
#     $ cd /path/to/your/turtle_ros
#     $ docker/run.sh
#

# /proc or /sys files aren't mountable into docker

# sudo docker run -it -v /home/turtle_ros:/home/turtle_ros --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name turtle --network host

# check for display

sudo xhost +si:localuser:root
DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix"

# check for V4L2 devices
V4L2_DEVICES=""
for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# check for usb ports ()
USB_DEVICES=""
for i in {0..9}
do
	if [ -a "/dev/ttyUSB$i" ]; then
		USB_DEVICES="$USB_DEVICES --device /dev/ttyUSB$i "
	fi
done
# use Erick's image
CONTAINER_IMAGE="f00cc102bf69" 
NAME="--name=turtle"
DEV_VOLUME="-v /home/$USER/turtle_ros:/turtle_ros" # Corrected path

sudo docker run --runtime nvidia -it --name turtle_noport \
    --network host \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /etc/nv_tegra_release:/etc/nv_tegra_release \
    $DISPLAY_DEVICE $V4L2_DEVICES $USB_DEVICES\
    $DEV_VOLUME \
    $CONTAINER_IMAGE
