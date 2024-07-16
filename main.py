import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """Calculate the angle between three points.
    
    Args:
        a, b, c: Three points (hip, knee, ankle).
        
    Returns:
        Angle in degrees between the three points.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    # Calculate the angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure angle is within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Load image
image_path = '/home/sam/Codes/Rapid/cycle_pose/image.png'
image = cv2.imread(image_path)

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Process image to detect pose
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# Extract landmarks
landmarks = results.pose_landmarks.landmark

# Define points (assuming left leg for this example)
rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
# Calculate angle
rangle = calculate_angle(rhip, rknee, rankle)
langle = calculate_angle(lhip, lknee, lankle)

print(f'The angle between the left hip, knee, and ankle is: {180 - langle} degrees')
print(f'The angle between the right hip, knee, and ankle is: {180 - rangle} degrees')
