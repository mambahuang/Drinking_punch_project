# Drinking_punch_project

## Devices needed
* Jetson Nano 4GB
* Jetson audio card
* USB轉音訊模組免驅音效卡(適用於Jetson nano)
* 2.4G Wifi 發射器
* Logitech C270 Webcam
* 128GB Micro SD card

## Demo Video on Youtube
Full project:
https://youtu.be/njIzLgYLK_Y?si=EH6p-7wpnfle2mgD \
SVM model Demo only:
https://youtu.be/9p9VhpmyNac?si=4MXHjh-fC6NMxKo_

## Prerequisite Software for Writing image on Jetson Nano
BalenaEtcher
https://etcher.balena.io/
SD card formatter
https://www.sdcard.org/downloads/formatter/
Download image
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write

## Docker Environment
Ubuntu 18.04 based on Python 3.6.9

Build Docker Environment:
1. `sudo docker pull jim0406/nano_course:latest `
2. `sudo docker build -t jim0406/nano_course:latest .`
3. `xhost +local:root`
4. `sudo chmod 777 /dev/video0`
5. > sudo docker run -it \\
    --net=host \\
    --runtime nvidia \\
    --group-add audio \\
    -e DISPLAY=$DISPLAY \\
    --env="XAUTHORITY=/root/.Xauthority" \\
    --env="PULSE_SERVER=unix:/run/user/\$(id -u)/pulse/native" \\
    -v /tmp/.X11-unix/:/tmp/.X11-unix \\
    -v $XAUTHORITY:/root/.Xauthority \\
    --device=/dev/video0 \\
    --device /dev/snd \\
    -v /Your project location:/location/in/container \\
    jim0406/nano_course
6. `cd location/in/container`
7. `apt-get install python3-sklearn pygame`
8. `python3 main.py`
9. `exit`


## How to re-enter Docker Environment after exit
`xhost +`
`xhost +SI:localusr:root`
`sudo docker restart <container_ID or name>`
`sudo docker exec -it <container_ID or name>`

## 【Error】Encounter "cannot allocate memory in static TLS block"
Run this command below:(replace the path below to your error path)
`export LD_PRELOAD=<parent path to python3.6>/python3.6/site-packages/torch/lib/libgomp-d22c30c5.so.1`

## Project Detail
* **UI interface**
    * PyQt5
    * Qt Designer
* **Data Collection**
    * Run `./SVM_test/Data_Collect_V3.py`
* **Training Collected data**
    * Method_1: Run `./SVM_test/Train_Model_V3.py` for SVM method
    * Method_2: Run `./SVM_test/trainNN.py` for simple 2 layers neural network and save as pth file (including loss and accuracy images)
* **Testing Model Accuracy**
    * Method_1: Run `./SVM_test/Demo_V3_Multi.py`
    * Method_2: Test using `./Jetson_NANO_group10_no/main.py`
* **'Audio' or 'No Audio'**
    * Run `./Jetson_NANO_group10_no` with gesture detection only
    * Run `./Jetson_NANO_group10_audio` or `./Jetson_NANO_group10_audioWithThread` with sound (`./Jetson_NANO_group10_audioWithThread` version let the game (sound and camera) run more smoothly)

## Drinking punch Game Rules
### Button function
1. Press `Start` button to **start** the game
2. Press `Pause` button to **pause** the game, and Press `Resume` button to **resume** the game
3. Press `Restart` button to **restart** game (You can restart the game at any time when playing)
### Gesture you can use in the game
1. `Zero` --> Face camera and pose two fists
2. `Five` --> There're two ways: Left hand is fist and right hand is paper. Left hand is paper, right hand is fist.
4. `Ten` --> Face camera and pose two paper.
### Scoring rule
1. When the game start, you will see a **picture** and **Question**. The way to get the point is to let the gesture on picture **add** the gesture you pose not equal to the **Question**.
2. You need to use two hands to play the game, or no point you will get while playing game.


## Different models and methods I try before this final version
Tool: Google Colab Pro
* **Resnet18** and** Resnet50** with custom dataset
* Custom **CNN** and custom dataset
* **VGG19** with custom dataset
* **Yolo v3** (darknet53.conv.74), **Yolo v4** (yolov4.conv.137) with labelImg(customize dataset and label it manually)

Reference:
* Training Yolo using darknet \
https://github.com/AlexeyAB/darknet \
https://pjreddie.com/darknet/yolo/ \

* Labeling Tool \
https://github.com/HumanSignal/labelImg

