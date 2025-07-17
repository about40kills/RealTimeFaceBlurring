# RealTimeFaceBlurring

This project provides a real-time face blurring (and pixelation) tool using your webcam. It uses OpenCV’s deep learning face detector and a tracker for smooth, fast face anonymization.

## Features

- **Real-time webcam feed**
- **Face detection** using OpenCV DNN (SSD Caffe model)
- **Face tracking** for performance
- **Two anonymization modes:**  
  - Blur (default)  
  - Pixelate (press `p` to switch)
- **Switch modes** on the fly (`b` for blur, `p` for pixelate)
- **Mirror image** display (like a selfie camera)

## Requirements

- Python 3.x
- OpenCV (with contrib modules)
- Webcam

## Setup

1. **Install dependencies:**
    ```bash
    pip install opencv-contrib-python
    ```

2. **Download the face detection model files:**
    - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
    - [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
    - Place both files in the same directory as `face_blur.py`.

3. **Run the script:**
    ```bash
    python face_blur.py
    ```

## Usage

- The webcam window will open and display a mirrored video feed.
- Detected faces will be blurred (default) or pixelated.
- Press `b` to switch to blur mode.
- Press `p` to switch to pixelate mode.
- Press `q` to quit.

---

## How it works

- Uses a DNN model to detect faces in the webcam feed.
- Initializes a tracker on the detected face for smooth, fast updates.
- Applies either a blur or pixelation effect to the face region.
- Overlays the processed face back onto the video frame.
- Displays the result in real time.

---

## Notes

- The script uses OpenCV’s legacy tracker API. If you get errors, ensure you have `opencv-contrib-python` installed.
- For best results, use in a well-lit environment.

---

## License

MIT License