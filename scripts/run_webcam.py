import cv2
from ultralytics import YOLO

# 1. Initialize the YOLO Pose model and Webcam
model = YOLO("models/yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# The 17 COCO keypoints tracked by YOLO
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Define which keypoint pairs should connect to draw the skeleton bones
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face lines
    (5, 6),                              # Shoulder to shoulder
    (5, 7), (7, 9),                      # Left arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                     # Right arm (Shoulder -> Elbow -> Wrist)
    (5, 11), (6, 12), (11, 12),          # Torso / Hips
    (11, 13), (13, 15),                  # Left leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16)                   # Right leg (Hip -> Knee -> Ankle)
]

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 2. Extract Raw Inference Data Only
    results = model(frame, conf=0.55, classes=0, verbose=False)
    
    # Create a clean working copy of the raw video frame
    display_frame = frame.copy()

    frame_result = results[0]

    # 3. Extract and Process Raw Coordinates
    if frame_result.keypoints is not None and len(frame_result.keypoints.xy) > 0:
        all_people_kpts = frame_result.keypoints.xy

        # Loop through each person detected in the frame
        for person_idx, person_kpts in enumerate(all_people_kpts):
            
            # --- DRAW SKELETON LINES ---
            for p1_idx, p2_idx in SKELETON_CONNECTIONS:
                # Get the coordinates for both joints in the connection pair
                pt1 = person_kpts[p1_idx]
                pt2 = person_kpts[p2_idx]
                
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                # Only draw the connection line if both joints are actively detected
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue bones

            # --- DRAW JOINTS AND LABELS ---
            for joint_idx, coordinate in enumerate(person_kpts):
                x = float(coordinate[0])
                y = float(coordinate[1])
                
                if x > 0 and y > 0:
                    text = f"{KEYPOINT_NAMES[joint_idx]}: ({int(x)}, {int(y)})"
                    
                    # Draw a tiny dot on the joint position
                    cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1) # Green dots
                    
                    # Overlay the coordinate text string right next to the dot
                    cv2.putText(display_frame, text, (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # 4. Render Output Window
    cv2.imshow("Daycare Child Protection System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
