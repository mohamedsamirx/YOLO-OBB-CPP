# YOLO V8, V11 Oriented Bounding Boxes (OBB) Format ONNX CPP

![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.19.2-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)
![CMake](https://img.shields.io/badge/CMake-3.22.1-blue.svg)

## Overview  
This project is a single C++ header high-performance application designed for real-time object detection using the YOLOv8 and YOLOv11 models in Oriented Bounding Box (OBB) format. Leveraging the power of [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [OpenCV](https://opencv.org/), it provides seamless integration with YOLOv8 and YOLOv11 implementations for image, video, and live camera inference. Whether you're developing for research, production, or hobbyist projects, this application offers flexibility and efficiency while supporting accurate oriented object detection.




## Output

<div align="center">
  <img src="data/OBB_test_1_output.jpg" alt="Image Output" width="500">
  
  <img src="data/OBB_test_2_output.jpg" alt="Image Output" width="500">
</div>




### Integration in your c++ projects

## Detection Example

```cpp

// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "YOLO11-OBB.hpp" 

int main(){

    // Configuration parameters
    const std::string labelsPath = "../models/Dota.names";       // Path to class labels
    const std::string modelPath  = "../models/yolo11n-obb.onnx";     // Path to YOLO model
    const std::string imagePath  = "../data/OBB_test_1_output.jpg";           // Path to input image
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the detector
    YOLO11OBBDetector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object detection to get bboxs
    std::vector<Detection> detections = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBoxMask(image, detections);

    // Display the annotated image
    cv::imshow("OBB Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}


```

> **Note:** For more usage, check the source files: [camera_inference.cpp](src/camera_inference.cpp), [image_inference.cpp](src/image_inference.cpp), [video_inference.cpp](src/video_inference.cpp).

## Features

- **ONNX Runtime Integration**: Leverages ONNX Runtime for optimized inference on both CPU and GPU, ensuring high performance.
  - **Dynamic Shapes Handling**: Adapts automatically to varying input sizes for improved versatility.
  - **Graph Optimization**: Enhances performance using model optimization with `ORT_ENABLE_ALL`.
  - **Execution Providers**: Configures sessions for CPU or GPU (e.g., `CUDAExecutionProvider` for GPU support).
  - **Input/Output Shape Management**: Manages dynamic input tensor shapes per model specifications.
  - **Optimized Memory Allocation**: Utilizes `Ort::MemoryInfo` for efficient memory management during tensor creation.
  - **Batch Processing**: Supports processing multiple images, currently focused on single-image input.
  - **Output Tensor Extraction**: Extracts output tensors dynamically for flexible result handling.

- **OpenCV Integration**: Uses OpenCV for image processing and rendering bounding boxes and labels.

- **Real-Time Inference**: Capable of processing images, videos, and live camera feeds instantly.

- **Easy-to-Use Scripts**: Includes shell scripts for straightforward building and running of different inference modes.



## Requirements

Before building the project, ensure that the following dependencies are installed on your system:

- **C++ Compiler**: Compatible with C++14 standard (e.g., `g++`, `clang++`, or MSVC).
- **CMake**: Version 3.0.0 or higher.
- **OpenCV**: Version 4.5.5 or higher.
- **ONNX Runtime**: Tested with version 1.16.3 and 1.19.2, backward compatibility [Installed and linked automatically during the build].

## Installation

### Clone Repository

First, clone the repository to your local machine:

```bash 
git clone https://github.com/mohamedsamirx/YOLO-OBB-CPP.git
cd YOLO-OBB-CPP
```


### Configure

1. make sure you have opencv c++ installed
2. set the ONNX Runtime version you need e.g. ONNXRUNTIME_VERSION="1.16.3" in [build.sh](build.sh) to download ONNX Runtime headers also set GPU.

4. Optional: control the debugging and timing using [Config.hpp](tools/Config.hpp)



### Build the Project

Execute the build script to compile the project using CMake:

```bash
./build.sh
```

This script will download onnxruntime headers, create a build directory, configure the project, and compile the source code. Upon successful completion, the executable files (camera_inference, image_inference, video_inference) will be available in the build directory.

### Usage

After building the project, you can perform object detection on images, videos, or live camera feeds using the provided shell scripts.

#### Run Image Inference

To perform object detection on a single image:

```bash
./run_image.sh 
```

This command will process The image and display the output image with detected bounding boxes and labels.

#### Run Video Inference

To perform object detection on a video file:

```bash
./run_video.sh 
```

#### Run Camera Inference

To perform real-time object detection using a usb cam:

```bash
./run_camera.sh 
```

This command will activate your usb and display the video feed with real-time object detection.


**Class Names:**
- Dota.names: Contains the list of class labels used by the models.

### License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

### Acknowledgment

- [https://github.com/Geekgineer/YOLOs-CPP/tree/main](https://github.com/Geekgineer/YOLOs-CPP/tree/main)
