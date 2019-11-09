# Setup Instruction

## Hardware Requirement
    >   NVIDIA Jetson Nano 
    >   SD Card [Current setup tested on 32gb, recommended: 64-128GB]
    >   DC 5V/4A
    >   Geekworm NVIDIA Jetson Nano WiFi Adapter Dual Band Wireless USB 3.0 Adapter 5GHz
    >   NVIDIA Jetson Nano Metal Case/Enclosure with Fan Cooling, Power & Reset Control Switch for NVIDIA Jetson Nano
    >   LI-IMX219-MIPI-FF-NANO-H145 [Camera for video streaming]
    
## Software Requirement
    >   Boot the NVIDIA Jetson Nano as per the setup instruction on official web page
    >   Install greengrass if needed
    >   Python3 [will had the requirements.txt soon]
    >   Install opencv with gstreamer -ON
    >   DLR i.e. runtime for ML models compiled by AWS SageMaker Neo
        https://github.com/neo-ai/neo-ai-dlr

## Important commands
    - Ubuntu latest dependencies
    sudo apt-get upgrade
    - Check video feed
    gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e
    - CV2 build information
    print (cv2.getBuildInformation())
    - gstreamer ON
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python2=OFF \
    -D CMAKE_INSTALL_PREFIX=$(python3 -c “import sys; print(sys.prefix)”) \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c “from distutils.sysconfig import get_python_inc; print(get_python_inc())”) \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c “from distutils.sysconfig import get_python_lib; print(get_python_lib())”) \
    -D WITH_GSTREAMER=ON \
    -D BUILD_EXAMPLES=ON ..
    
Once all the requirements and setup completes, store the model and local.py in the home dir /home/<user>
Run the file as python3 local.py 

// For now, we are taking top 1, but I am editing the code to get topK [K based on confidence threshold]    
Output ->  {"Object" : "Person", "confidence": "0.92" }


    
    
    