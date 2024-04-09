from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="t9MiLnODPWvJmI3X6QOx"
)

result = CLIENT.infer("car-race-438467_640.jpg", model_id="traffic_management-1hjbr/2")
predictions = result['predictions']

# Iterate over predictions and print the class for each one
for prediction in predictions:
    class_label = prediction['class']
    print("Detected class:", class_label)