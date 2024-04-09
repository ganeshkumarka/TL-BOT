# import cv2
# from inference_sdk import InferenceHTTPClient
# import winsound
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="t9MiLnODPWvJmI3X6QOx"
# )

# def detect_ambulance(image):
#     result = CLIENT.infer(image, model_id="traffic_management-1hjbr/2")
#     predictions = result['predictions']
#     for prediction in predictions:
#         class_label = prediction['class']
#         if class_label.lower() == 'ambulance':  # Convert to lowercase for case-insensitive comparison
#             return True
#     return False

# def process_video(video_path=None):
#     if video_path is None:
#         cap = cv2.VideoCapture(0)  # Open the webcam
#     else:
#         cap = cv2.VideoCapture(video_path)  # Open the video file

#     while True:
#         ret, frame = cap.read()  # Read a frame from the video

#         if not ret:
#             break  # Check if the video has ended

#         # Detect ambulance in the frame
#         is_ambulance = detect_ambulance(frame)

#         # Display the frame and ambulance detection result
#         cv2.putText(frame, "Ambulance Detected" if is_ambulance else "No Ambulance", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#         cv2.imshow("Video", frame)  # Show the frame
#         if is_ambulance:
#             winsound.Beep(1000, 500) 
#             cv2.waitKey(0)  # Wait indefinitely if an ambulance is detected
            

#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("1. Use Webcam")
#     print("2. Use Video File")
#     choice = input("Enter your choice (1 or 2): ")

#     if choice == '1':
#         process_video()
#     elif choice == '2':
#         video_path = input("Enter the path to the video file: ")
#         process_video(video_path)
#     else:
#         print("Invalid choice!")



# import cv2
# from inference_sdk import InferenceHTTPClient
# import winsound

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="t9MiLnODPWvJmI3X6QOx"
# )

# def detect_ambulance(image):
#     result = CLIENT.infer(image, model_id="traffic_management-1hjbr/2")
#     predictions = result['predictions']
#     for prediction in predictions:
#         class_label = prediction['class']
#         if class_label.lower() == 'ambulance':  # Convert to lowercase for case-insensitive comparison
#             return True
#     return False

# def process_video(video_path=None):
#     if video_path is None:
#         cap = cv2.VideoCapture(0)  # Open the webcam
#     else:
#         cap = cv2.VideoCapture(video_path)  # Open the video file

#     while True:
#         ret, frame = cap.read()  # Read a frame from the video

#         if not ret:
#             break  # Check if the video has ended

#         # Detect ambulance in the frame
#         is_ambulance = detect_ambulance(frame)

#         # Display the frame and ambulance detection result
#         cv2.putText(frame, "Ambulance Detected" if is_ambulance else "No Ambulance", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#         cv2.imshow("Video", frame)  # Show the frame
#         if is_ambulance:
#             winsound.Beep(1000, 500)  # Make sound when ambulance is detected

#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and close all windows
#     # cap.release()
#     # cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("1. Use Webcam")
#     print("2. Use Video File")
#     choice = input("Enter your choice (1 or 2): ")

#     if choice == '1':
#         process_video()
#     elif choice == '2':
#         video_path = input("Enter the path to the video file: ")
#         process_video(video_path)
#     else:
#         print("Invalid choice!")

#claude

# import cv2
# import time
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="t9MiLnODPWvJmI3X6QOx"
# )

# def detect_ambulance(image):
#     result = CLIENT.infer(image, model_id="traffic_management-1hjbr/2")
#     predictions = result['predictions']
#     for prediction in predictions:
#         class_label = prediction['class']
#         if class_label.lower() == 'ambulance':
#             return True
#     return False

# def create_traffic_light(window_name):
#     cv2.namedWindow(window_name)
#     cv2.resizeWindow(window_name, 100, 300)
#     red_light = cv2.resize(cv2.imread('dataset/red.png', cv2.IMREAD_UNCHANGED), (100, 100))
#     yellow_light = cv2.resize(cv2.imread('dataset/yellow.png', cv2.IMREAD_UNCHANGED), (100, 100))
#     green_light = cv2.resize(cv2.imread('dataset/green.png', cv2.IMREAD_UNCHANGED), (100, 100))
#     return red_light, yellow_light, green_light

# def update_traffic_lights(lights, state):
#     red_light, yellow_light, green_light = lights
#     if state == 'red':
#         cv2.imshow('Traffic Light 1', red_light)
#         cv2.imshow('Traffic Light 2', red_light)
#         cv2.imshow('Traffic Light 3', red_light)
#         cv2.imshow('Traffic Light 4', red_light)
#     elif state == 'green':
#         cv2.imshow('Traffic Light 1', green_light)
#         cv2.imshow('Traffic Light 2', red_light)
#         cv2.imshow('Traffic Light 3', red_light)
#         cv2.imshow('Traffic Light 4', red_light)
#     elif state == 'yellow':
#         cv2.imshow('Traffic Light 1', yellow_light)
#         cv2.imshow('Traffic Light 2', red_light)
#         cv2.imshow('Traffic Light 3', red_light)
#         cv2.imshow('Traffic Light 4', red_light)

# def process_video(video_path=None):
#     cap = cv2.VideoCapture(0) if video_path is None else cv2.VideoCapture(video_path)
#     traffic_light_states = ['red', 'green', 'yellow']
#     state_index = 0
#     traffic_lights = [create_traffic_light(f'Traffic Light {i+1}') for i in range(4)]
#     update_traffic_lights(traffic_lights[0], traffic_light_states[state_index])
#     ambulance_detected = False
#     ambulance_detection_time = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         is_ambulance = detect_ambulance(frame)
#         cv2.putText(frame, "Ambulance Detected" if is_ambulance else "No Ambulance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Video", frame)

#         if is_ambulance and not ambulance_detected:
#             ambulance_detected = True
#             ambulance_detection_time = time.time()
#             update_traffic_lights(traffic_lights[0], 'green')
#             for light in traffic_lights[1:]:
#                 update_traffic_lights(light, 'red')

#         if ambulance_detected:
#             elapsed_time = time.time() - ambulance_detection_time
#             if elapsed_time >= 10:
#                 ambulance_detected = False
#                 state_index = (state_index + 1) % len(traffic_light_states)
#                 for light in traffic_lights:
#                     update_traffic_lights(light, traffic_light_states[state_index])

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         if not ambulance_detected:
#             time.sleep(30)
#             state_index = (state_index + 1) % len(traffic_light_states)
#             for light in traffic_lights:
#                 update_traffic_lights(light, traffic_light_states[state_index])

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("1. Use Webcam")
#     print("2. Use Video File")
#     choice = input("Enter your choice (1 or 2): ")
#     if choice == '1':
#         process_video()
#     elif choice == '2':
#         video_path = input("Enter the path to the video file: ")
#         process_video(video_path)
#     else:
#         print("Invalid choice!")


#with black

import numpy as np
import cv2
import time
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
        if class_label.lower() == 'ambulance':
            return True
    return False

# def create_traffic_light():
#     red_light = cv2.resize(cv2.imread('dataset/red.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
#     yellow_light = cv2.resize(cv2.imread('dataset/yellow.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
#     green_light = cv2.resize(cv2.imread('dataset/green.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
#     black_image = cv2.resize(cv2.imread('dataset/black.png', cv2.IMREAD_GRAYSCALE), (50, 50), interpolation=cv2.INTER_AREA)
#     return red_light, yellow_light, green_light, black_image
def create_traffic_light():
    red_light = cv2.resize(cv2.imread('dataset/red.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
    yellow_light = cv2.resize(cv2.imread('dataset/yellow.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
    green_light = cv2.resize(cv2.imread('dataset/green.png', cv2.IMREAD_GRAYSCALE), (50, 150), interpolation=cv2.INTER_AREA)
    black_image = cv2.resize(cv2.imread('dataset/black.png', cv2.IMREAD_GRAYSCALE), (50, 50), interpolation=cv2.INTER_AREA)
    
    # Check for None values
    if red_light is None or yellow_light is None or green_light is None or black_image is None:
        raise ValueError("One or more images could not be loaded.")

    return red_light, yellow_light, green_light, black_image


# def update_traffic_lights(frame, lights, state):
#     red_light, yellow_light, green_light, black_image = lights
#     # Ensure all images are grayscale and have the same dimensions
#     if red_light.ndim == 3: # If it's a color image, convert to grayscale
#         red_light = cv2.cvtColor(red_light, cv2.COLOR_BGR2GRAY)
#     if yellow_light.ndim == 3:
#         yellow_light = cv2.cvtColor(yellow_light, cv2.COLOR_BGR2GRAY)
#     if green_light.ndim == 3:
#         green_light = cv2.cvtColor(green_light, cv2.COLOR_BGR2GRAY)
#     if black_image.ndim == 3:
#         black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    
#     # Ensure all images have the same dimensions
#     if red_light.shape[0] != black_image.shape[0]:
#         black_image = cv2.resize(black_image, (black_image.shape[1], red_light.shape[0]))
    
#     if state == 'red':
#         traffic_light = cv2.hconcat([red_light, cv2.vconcat([black_image, black_image])])
#     elif state == 'green':
#         traffic_light = cv2.hconcat([cv2.vconcat([black_image, green_light]), black_image])
#     elif state == 'yellow':
#         traffic_light = cv2.hconcat([black_image, cv2.vconcat([black_image, yellow_light])])

#     roi = frame[20:170, 400:500]
#     roi[:] = cv2.resize(traffic_light, (100, 150), interpolation=cv2.INTER_AREA)
def update_traffic_lights(frame, lights, state):
    red_light, yellow_light, green_light, _ = lights  # We don't need the black_image here anymore

    roi = frame[20:170, 400:450]
    if state == 'red':
        red_light_color = cv2.cvtColor(red_light, cv2.COLOR_GRAY2BGR)
        roi[:, :50] = red_light_color
        black_image = np.zeros_like(red_light, dtype=np.uint8)  # Create a black image with the same dimensions as red_light
        roi[:, 50:] = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)  # Convert black_image to color (grayscale)
    elif state == 'green':
        green_light_color = cv2.cvtColor(green_light, cv2.COLOR_GRAY2BGR)
        black_image = np.zeros_like(green_light, dtype=np.uint8)  # Create a black image with the same dimensions as green_light
        roi[:, :50] = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
        roi[:, 50:] = green_light_color
    elif state == 'yellow':
        yellow_light_color = cv2.cvtColor(yellow_light, cv2.COLOR_GRAY2BGR)
        black_image = np.zeros_like(yellow_light, dtype=np.uint8)  # Create a black image with the same dimensions as yellow_light
        roi[:, :50] = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
        roi[:, 50:] = yellow_light_color

    roi_bottom = frame[170:220, 400:450]
    roi_bottom[:, :] = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)  # Convert black_image to color (grayscale)

    roi = frame[20:220, 450:500]
    roi[:, :] = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)  # Convert black_image to color (grayscale)

def process_video(video_path=None):
    cap = cv2.VideoCapture(0) if video_path is None else cv2.VideoCapture(video_path)
    traffic_light_states = ['red', 'green', 'yellow']
    state_index = 0
    traffic_lights = create_traffic_light()
    ambulance_detected = False
    ambulance_detection_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_ambulance = detect_ambulance(frame)
        cv2.putText(frame, "Ambulance Detected" if is_ambulance else "No Ambulance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_ambulance and not ambulance_detected:
            ambulance_detected = True
            ambulance_detection_time = time.time()
            update_traffic_lights(frame, traffic_lights, 'green')
        else:
            update_traffic_lights(frame, traffic_lights, traffic_light_states[state_index])

        if ambulance_detected:
            elapsed_time = time.time() - ambulance_detection_time
            if elapsed_time >= 10:
                ambulance_detected = False
                state_index = (state_index + 1) % len(traffic_light_states)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ambulance_detected:
            time.sleep(30)
            state_index = (state_index + 1) % len(traffic_light_states)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("1. Use Webcam")
    print("2. Use Video File")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        process_video()
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        process_video(video_path)
    else:
        print("Invalid choice!")