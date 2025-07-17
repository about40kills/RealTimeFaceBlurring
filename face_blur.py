import cv2 as cv

# Load DNN face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize webcam
capture = cv.VideoCapture(0)
tracker = None
track_initialized = False
mode = 'blur'

def pixelate(face_roi, pixel_size=10):
    h, w = face_roi.shape[:2]
    temp = cv.resize(face_roi, (w // pixel_size, h // pixel_size), interpolation=cv.INTER_NEAREST)
    return cv.resize(temp, (w, h), interpolation=cv.INTER_NEAREST)

while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    if not ret:
        break

    h, w = frame.shape[:2]

    # Face detection if no active tracker
    if not track_initialized:
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Replace this line with your OpenCV version's tracker
                tracker = cv.legacy.TrackerKCF_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                track_initialized = True
                break

    else:
        success, box = tracker.update(frame)
        if success:
            (x1, y1, w_box, h_box) = [int(v) for v in box]
            x2, y2 = x1 + w_box, y1 + h_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            if mode == 'pixelate':
                face_roi = pixelate(face_roi)
            elif mode == 'blur':
                ksize = (151, 151)
                face_roi = cv.GaussianBlur(face_roi, ksize, 0)

            # Apply the processed face back to frame
            frame[y1:y2, x1:x2] = face_roi
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            track_initialized = False  # Re-trigger DNN

    # Display mode
    cv.putText(frame, f"Mode: {mode.upper()} | [B]lur | [P]ixelate | [Q]uit",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.imshow('Video', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        mode = 'blur'
    elif key == ord('p'):
        mode = 'pixelate'

capture.release()
cv.destroyAllWindows()
