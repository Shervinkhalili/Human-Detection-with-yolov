import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

video = cv2.VideoCapture("Gucci.mp4")
skip_factor = 5  # Number of frames to skip

frame_counter = 0  # Counter for frames processed

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    frame_counter += 1

    # Skip frames based on skip factor
    if frame_counter % skip_factor != 0:
        continue

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run forward pass and get the output
    output_layers = net.forward(net.getUnconnectedOutLayersNames())

    # Lists to store the detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process the output
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections for humans
            if class_id == 0 axnd confidence > 0.5:
                # Get the bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Store the bounding box, confidence, and class ID
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

    # Check if any detections are present
    if len(indices) > 0:
        # Draw the final bounding boxes
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Human Detection", frame)

    # Decrease the delay for faster playback
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()
