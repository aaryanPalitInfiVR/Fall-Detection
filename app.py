import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from MediapipePoseEstimator import MediapipePoseEstimator
import pickle
import os

pathToPretrainedModel = "pretrained_models/pm_37vtrain_mp.pkl"
defaultSavePath = os.path.join(os.getcwd(), "results.mp4")  # Default path (current working directory)

def runDetection(videoInput, sensor_id=None):
    """Run the detection using the MediapipePoseEstimator."""
    recordedVideoSavePath = "results.mp4"  # Temporary save path
    with open(pathToPretrainedModel, 'rb') as file:
        pretrainedModel = pickle.load(file)

    video_pose_classifier = MediapipePoseEstimator(sensor_id=sensor_id)
    video_pose_classifier.init_video(videoInput)
    video_pose_classifier.init_mediapipe(pretrainedModel)
    video_pose_classifier.estimate_pose(recordedVideoSavePath)

    messagebox.showinfo("Success", f"Detection complete. Video saved as '{recordedVideoSavePath}'.")

def selectVideo():
    """Select a video file and run detection."""
    videoPath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if videoPath:
        runDetection(videoPath)

def startDefaultFeed():
    """Start detection using the default webcam feed."""
    sensor_id = sensorEntry.get()  # Get the sensor ID from the entry field
    runDetection(0, sensor_id)

def checkExternalSensor():
    """Handle the external sensor feed input dynamically by detecting connected cameras."""
    def detectExternalCamera():
        for port in range(1, 10):  # Scan ports 1-9 for external cameras
            cap = cv2.VideoCapture(port)
            if cap.isOpened():
                cap.release()
                return port
        return None

    def onConfirm():
        externalPort = detectExternalCamera()
        sensor_id = sensorEntry.get()  # Get the sensor ID from the entry field
        if externalPort is not None:
            runDetection(externalPort, sensor_id)  # Pass the sensor_id to the runDetection function
            sensorWindow.destroy()
        else:
            messagebox.showwarning("No External Sensor", "No external sensor found. Please connect a USB camera.")

    sensorWindow = tk.Toplevel()
    sensorWindow.title("External Sensor Feed")
    sensorWindow.geometry("300x200")

    tk.Label(sensorWindow, text="Detecting external sensor....").pack(pady=10)

    confirmButton = tk.Button(sensorWindow, text="Check for External Sensor", command=onConfirm)
    confirmButton.pack(pady=20)

def createApp():
    """Create the main Tkinter application window."""
    root = tk.Tk()
    root.title("Fall Detection System")
    root.geometry("400x500")

    tk.Label(root, text="Select an Option:", font=("Arial", 14)).pack(pady=20)

    tk.Button(root, text="Default Feed", command=startDefaultFeed, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Upload a Video", command=selectVideo, width=20, height=2).pack(pady=10)
    tk.Button(root, text="External Sensor Feed", command=checkExternalSensor, width=20, height=2).pack(pady=10)

    # Add entry widget below the buttons to input sensor ID
    tk.Label(root, text="Enter Sensor ID (default is 3):").pack(pady=5)
    global sensorEntry
    sensorEntry = tk.Entry(root)
    sensorEntry.insert(0, "3")  # Default value is 3
    sensorEntry.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    createApp()
