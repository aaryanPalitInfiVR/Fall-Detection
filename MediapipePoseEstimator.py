import cv2
import mediapipe as mp
import requests
from datetime import datetime
import time
from MediapipeUtilities import MediapipeUtilities

class MediapipePoseEstimator:
    def __init__(self, sensor_id):
        """
        Initialize the Pose Estimator with dynamic sensor ID
        :param sensor_id: ID of the sensor provided by the user
        """
        self.pretrainedModel = None
        self.videoCapture = None
        self.videoWidth = None
        self.videoHeight = None
        self.__mpPose = None
        self.__mpDrawing = mp.solutions.drawing_utils
        self.__outputWriter = None
        self.fallCount = 0  # Initialize Fall Count
        self.previousPoseState = "none"  # Track previous pose state
        self.fallenStartTime = None  # Timer for prolonged "fallen" state
        self.fallenThreshold = 2  # Duration (in seconds) to consider "fallen" as prolonged
        self.notFallenThreshold = 1  # Threshold (in seconds) to wait before resetting the timer when pose transitions
        self.stateStartTime = None  # To track the last state and the time spent in it
        self.sensorId = sensor_id  # Dynamic sensor ID
        self.logFilePath = "fallEventLogs.txt"  # Local log file path

    def init_video(self, file_path: str):
        """
        Initialize video for estimation

        :param file_path: path to video which is to be pose-estimated
        """
        self.videoCapture = cv2.VideoCapture(file_path)
        self.videoWidth = self.videoCapture.get(3)
        self.videoHeight = self.videoCapture.get(4)

    def init_mediapipe(self, model):
        """
        Initialize model from sklearn for pose-estimation

        :param model: example sklearn.neural_network._multilayer_perceptron.MLPClassifier(required_model_config).fit(required_train_data)
        """
        self.__mpPose = mp.solutions.pose
        self.pretrainedModel = model

    def __init_output_video_writer(self, output_save_path: str):
        """
        Initializes video writer

        :param output_save_path: path to generated mp4 file
        :return: cv2.VideoWriter
        """
        if not output_save_path.lower().endswith(".mp4"):
            output_save_path += ".mp4"  # Add extension only if it's not present

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outputWriter = cv2.VideoWriter(
            output_save_path, fourcc, 30.0,
            (int(self.videoCapture.get(3)), int(self.videoCapture.get(4)))
        )
        return outputWriter

    def logFallEvent(self):
        """
        Logs fall event to an API and maintains a local log file.
        """
        url = "http://elderly-healthcare.infivr.com/api/events/create"
        headers = {"Content-Type": "application/json"}

        eventData = {
            "eventName": "Fall detected",
            "userName": "Aaryan Palit",  # Customize this
            "applicationName": "Fall Detection App",
            "eventTime": datetime.now().isoformat(),
            "severity": "high",
            "sensorId": self.sensorId,  # Dynamic sensor ID
            "fallCount": self.fallCount  # Add fall count to the API payload
        }

        # Send the post request to log the fall event
        response = requests.post(url, json=eventData, headers=headers)
        if response.status_code == 201:
            print(f"Fall event logged successfully. Fall Count: {self.fallCount}")
        else:
            print(f"Failed to log fall event: {response.text}")

        # Maintain a local log file
        with open(self.logFilePath, "a") as log_file:
            log_file.write(f"{datetime.now().isoformat()} - Fall Count: {self.fallCount}\n")

    def estimate_pose(self, output_save_path=None):
        """
        Shows video with real-time pose estimation using mediapipe holistic and sklearn classification.
        """
        fpsTime = 0
        frameN = 0
        poseLabel = "none"

        if output_save_path:
            self.__outputWriter = self.__init_output_video_writer(output_save_path)

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as holistic:
            while self.videoCapture.isOpened():
                success, image = self.videoCapture.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                # Prepare the frame
                frameN += 1
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmarks
                try:
                    self.DrawingCustomLandmarks(image, results)
                except Exception as e:
                    print(f"Error in drawing custom landmarks: {e}")

                # Pose classification logic
                try:
                    keyPoints = MediapipeUtilities().get_keypoints(results.pose_landmarks, self.videoWidth, self.videoHeight)
                    coordinatesLine = MediapipeUtilities().get_coords_line(keyPoints)
                    if 67 >= len(coordinatesLine) >= 1:
                        poseCode = self.pretrainedModel.predict([coordinatesLine])
                        poseLabel = MediapipeUtilities().get_pose_from_num(poseCode)

                        # If the pose is "fallen", start the timer
                        if poseLabel == "fallen":
                            if self.fallenStartTime is None:  # Start timer when first detected
                                self.fallenStartTime = time.time()
                        else:
                            # Check if the pose has transitioned to something else for more than the threshold before resetting timer
                            if self.fallenStartTime is not None:
                                elapsed_time = time.time() - self.fallenStartTime
                                if elapsed_time >= self.notFallenThreshold:
                                    self.fallenStartTime = None  # Reset the timer if pose transitioned for enough time

                        # If "fallen" state lasts longer than the threshold, increment the fall count
                        if self.fallenStartTime is not None:
                            elapsed_time = time.time() - self.fallenStartTime
                            if elapsed_time >= self.fallenThreshold:
                                # Increment fall count only once per "fall"
                                if self.previousPoseState != "fallen":
                                    self.fallCount += 1
                                    self.previousPoseState = "fallen"
                                    self.fallenStartTime = None  # Reset after counting

                                    # Log the fall event
                                    self.logFallEvent()

                        # If the pose is no longer "fallen", check if it was brief or prolonged
                        if poseLabel != "fallen" and self.previousPoseState == "fallen":
                            self.previousPoseState = "none"
                            self.fallenStartTime = None

                        # Display pose information
                        cv2.putText(
                            image,
                            "pose: %s" % poseLabel,
                            (int(keyPoints[0][0]), int(keyPoints[0][1]) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                except Exception as e:
                    print(f"Error in pose classification: {e}")

                # Render FPS, Pose label, and Fall count
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    image,
                    "FPS: %f" % (1.0 / (time.time() - fpsTime)),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 224, 0),
                    2
                )
                cv2.putText(
                    image,
                    "Detected Pose: %s" % poseLabel,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 224, 0),
                    2
                )
                cv2.putText(
                    image,
                    "Fall Count: %d" % self.fallCount,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 224, 0),
                    2
                )

                # Save and show the video
                if output_save_path:
                    self.__outputWriter.write(image)

                cv2.imshow('Fall Detection System', image)
                fpsTime = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.videoCapture.release()
        cv2.destroyAllWindows()
        if output_save_path:
            self.__outputWriter.release()

    def DrawingCustomLandmarks(self, image, results):
        """
        Draws custom landmarks for pose, hands, and other features detected.
        """
        # Draw pose connections
        if results.pose_landmarks:
            self.__mpDrawing.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                self.__mpDrawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                self.__mpDrawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
