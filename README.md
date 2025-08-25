# 🖥️ Jetson-Yolo-Tracking-TensorRT - Real-Time Object Tracking Made Easy

## 📦 Download Now
[![Download](https://img.shields.io/badge/Download-Latest%20Release-brightgreen)](https://github.com/spc-web-dev/Jetson-Yolo-Tracking-TensorRT/releases)

## 🚀 Getting Started
Welcome to Jetson-Yolo-Tracking-TensorRT! This application allows you to perform real-time YOLO detection and track multiple objects effectively. It works seamlessly with NVIDIA Jetson devices, enhancing your projects in computer vision, robotics, and surveillance.

## 📋 Features
- **Real-Time Detection**: Quickly identify objects in your environment.
- **Multi-Object Tracking**: Track various objects simultaneously.
- **Interactive Object Locking**: Focus on important objects with ease.
- **Optimized Performance**: Leverage NVIDIA's TensorRT for superior speed.

## 🛠️ System Requirements
- **Operating System**: Ubuntu 18.04 or later
- **NVIDIA Jetson Devices**: Compatible with Jetson Orin NX
- **CUDA Toolkit**: Version 10.0 or later
- **GStreamer**: For input handling through Argus camera
- **Python**: Version 3.6 or later
- **OpenCV**: For image processing capabilities

## 📥 Download & Install
To get started with Jetson-Yolo-Tracking-TensorRT, visit the Releases page to download the latest version:

[Download the release here](https://github.com/spc-web-dev/Jetson-Yolo-Tracking-TensorRT/releases)

1. Click on the above link.
2. Select the latest version from the list.
3. Download the package suitable for your system.
4. Follow the installation instructions included in the package.

## 🔧 Installation Instructions
1. **Extract Files**: After downloading, locate the downloaded file and extract it.
2. **Open Terminal**: Open a terminal window on your Jetson device.
3. **Navigate to Directory**: Use the `cd` command to navigate to the extracted directory.
4. **Install Dependencies**: Run the following command to install required packages:
   ```bash
   sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
   ```
5. **Run the Application**: To start the application, execute:
   ```bash
   ./run_yolo_tracking
   ```
   Make sure the Argus camera is properly set up and connected.

## 📷 Using the Application
Once the application is running:
- The camera feed will appear on the screen.
- You will see detected objects highlighted in real time.
- Use keyboard shortcuts to lock onto objects or to toggle settings.

## ⚙️ Configuration Options
You can customize the detection settings through a configuration file. Adjust parameters like:
- **Confidence Threshold**: Set the minimum confidence level for detection.
- **Object Classes**: Select specific object categories for tracking.

## 🛡️ Troubleshooting
If you encounter issues:
- **Check Dependencies**: Ensure that all necessary packages are installed.
- **Camera Connection**: Verify that the Argus camera is operating correctly.
- **Logs**: Check the terminal output for error messages to guide your troubleshooting.

## 🌐 Community Support
Join our community for support and insights:
- **GitHub Discussions**: Ask questions and share your experiences.
- **User Documentation**: Find additional resources and tutorials.

## 📞 Contact
For further assistance, contact us through the GitHub issues page on the repository.

## 📚 Additional Resources
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [YOLO Model Information](https://pjreddie.com/darknet/yolo/)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)

Thank you for choosing Jetson-Yolo-Tracking-TensorRT! Enjoy your object tracking experience.

### Reminder:
Don't forget to visit the Releases page again for future updates!

[Download the release here](https://github.com/spc-web-dev/Jetson-Yolo-Tracking-TensorRT/releases)