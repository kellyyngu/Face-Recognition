import face_recognition
import cv2
import os # for file path operations
import glob # for pattern-based file searching
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print(f"{len(images_path)} encoding images found.")
        
        encodings_by_name = {}

        # store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}. Skipping.")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # face recognition expects RGB

            # breaks down the full file path
            basename = os.path.basename(img_path) # eg "images/kelly.jpg" → "kelly.jpg"
            filename, ext = os.path.splitext(basename) # eg "kelly.jpg" → ("kelly", ".jpg")
            name = filename.split('_')[0]  # "Kelly_1" -> "Kelly"

            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) == 0:
                print(f"Warning: No face found in {img_path}. Skipping.")
                continue
            img_encoding = encodings[0]
            
            if name not in encodings_by_name:
                encodings_by_name[name] = []
            encodings_by_name[name].append(img_encoding)


        # Average encodings for each person
        self.known_face_encodings = []
        self.known_face_names = []
        for name, enc_list in encodings_by_name.items():
            avg_encoding = np.mean(enc_list, axis=0)
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(name)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        # shrinks the frame size to make face detection faster
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # detects all faces in the current frame and returns their positions
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        
        # for every face detected, process it one by one
        for face_encoding in face_encodings:
            # compares the current face to all the stored ones (self.known_face_encodings)
            # matches is a list of True/False values showing whether it matched each known person.
            # If no match is found, keep the default name as "Unknown"
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.35)
            name = "Unknown"

            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            #  returns the index of the smallest distance (the closest match)
            best_match_index = np.argmin(face_distances)
            
            # if the closest match is a match, use that name
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # convert face locations back to original size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        
        # return final results
        return face_locations.astype(int), face_names
