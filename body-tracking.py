import cv2
import numpy as np
import mediapipe as mp
import json

class FaceAndBodyTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize to store user interactions
        self.user_data = {
            'user_id': 1,
            'tried_on': [],   # Stores items tried on
            'liked_items': [] # Stores items liked
        }
        
        self.current_item = None  # Track currently displayed item

    def find_face_and_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_detection.process(img_rgb)
        self.pose_results = self.pose.process(img_rgb)
        
        # Face detection
        if self.face_results.detections:
            for detection in self.face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
        
        # Pose tracking
        if self.pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return img

    def track_user_interaction(self, item_id):
        # Simulate trying on a new item
        self.current_item = item_id
        if self.current_item not in self.user_data['tried_on']:
            self.user_data['tried_on'].append(self.current_item)
        print(f"User is trying on item {item_id}")

    def like_item(self):
        # User likes the current item
        if self.current_item and self.current_item not in self.user_data['liked_items']:
            self.user_data['liked_items'].append(self.current_item)
        print(f"User liked item {self.current_item}")

    def save_user_data(self):
        # Save user interaction data to a JSON file
        with open('user_data.json', 'w') as f:
            json.dump(self.user_data, f)
        print("User data saved!")


def recommend_similar_items(liked_items, df):
    # Simple recommendation based on similar style or color
    recommendations = df[df['item_id'].isin(liked_items)]['style'].unique()
    recommended_items = df[df['style'].isin(recommendations)]
    return recommended_items['item_id'].tolist()


def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceAndBodyTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.find_face_and_pose(frame)
        
        # Simulating trying on item 1
        tracker.track_user_interaction(item_id=1)
        
        # Simulate liking the item when 'l' is pressed
        if cv2.waitKey(1) & 0xFF == ord('l'):
            tracker.like_item()

        # Display the frame
        cv2.imshow("Face and Body Tracking", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save user data when the session ends
    tracker.save_user_data()

    # Example recommendation logic using saved liked items
    products = {
        'item_id': [1, 2, 3, 4],
        'color': ['red', 'blue', 'green', 'red'],
        'style': ['casual', 'formal', 'sports', 'casual'],
    }

    import pandas as pd
    df = pd.DataFrame(products)
    
    # Get recommendations based on liked items
    liked_items = tracker.user_data['liked_items']  # Correctly accessing liked_items now
    recommended_items = recommend_similar_items(liked_items, df)
    print(f"Recommended items based on your likes: {recommended_items}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
