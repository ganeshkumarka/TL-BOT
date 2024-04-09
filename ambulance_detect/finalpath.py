import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="t9MiLnODPWvJmI3X6QOx"
)

def detect_ambulance(image):
    result = CLIENT.infer(image, model_id="traffic_management-1hjbr/2")
    predictions = result['predictions']
    for prediction in predictions:
        class_label = prediction['class']
        if class_label.lower() == 'ambulance':  # Convert to lowercase for case-insensitive comparison
            return True
    return False

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file

    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:
            break  # Check if the video has ended

        # Detect ambulance in the frame
        is_ambulance = detect_ambulance(frame)

        # Display the frame and ambulance detection result
        cv2.putText(frame, "Ambulance Detected" if is_ambulance else "No Ambulance", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow( frame)  # Show the frame

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = input("Enter the path to the video file: ")
    process_video(video_path)
