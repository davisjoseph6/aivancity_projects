# Pulls an ubuntu image (with NVIDIA Libs if necessary)
FROM ubuntu
WORKDIR /usr/src/app

# Installs Python, tensorflow and any other necessary library
RUN sudo apt-get update && sudo apt install python3 && pip install tensorflow2

# Downloads a Yolo detector (choose as you wish)
RUN git clone https://github.com/ultralytics/yolov5 && cd yolov5 && pip install -r requirements.txt  

# Runs the Yolo detector on images available in a repertory in your computer
RUN python3 detect.py

# Stores back the detection images in your computer
# ENTRYPOINT 
