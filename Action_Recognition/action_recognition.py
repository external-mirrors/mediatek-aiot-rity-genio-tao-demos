import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque
import argparse

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNELS = 3
FRAME_STACK = 32
NUM_CLASSES = 5
CLASS_LABELS = ["push", "fall", "walk", "run", "ride_bike"]

def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return normalized

def apply_ema_smoothing(predictions, alpha=0.2):
    smoothed = np.zeros_like(predictions)
    smoothed[0] = predictions[0]
    for i in range(1, len(predictions)):
        smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def main(model_path, video_path):

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_buffer = deque(maxlen=FRAME_STACK)
    prediction_buffer = deque(maxlen=10)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        frame_buffer.append(processed_frame)
        frame_count += 1

        if frame_count < FRAME_STACK:
            # Skip the initial frames until we have enough frames in the buffer
            continue

        if len(frame_buffer) == FRAME_STACK:
            input_data = np.concatenate(list(frame_buffer), axis=2)
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = output_data.reshape(-1)

            # Apply softmax to the output data
            probabilities = softmax(output_data)

            prediction_buffer.append(probabilities)
            smoothed_predictions = apply_ema_smoothing(np.array(prediction_buffer))

            if len(smoothed_predictions) > 0:
                latest_prediction = smoothed_predictions[-1]
                predicted_class = np.argmax(latest_prediction)
                confidence = latest_prediction[predicted_class]
                action = CLASS_LABELS[predicted_class]

                cv2.putText(frame, f"Action: {action} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                print(f"Frame {frame_count}: Action: {action}, confidence: {confidence:.2f}")
                print(f"Raw probabilities: {output_data}")
                print(f"Smoothed probabilities: {latest_prediction}")

        cv2.imshow("Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Recognition using TFLite model")
    parser.add_argument("--model", type=str, required=True, help="Path to the TFLite model file")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    main(args.model, args.video)
