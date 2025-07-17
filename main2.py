import cv2
from simple_facerec import SimpleFacerec


class FaceRecognition:
    def __init__(
        self, image_folder="images/", width=400, height=400, process_every_n_frames=4
    ):
        # encode faces from a folder
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images(image_folder)

        # load camera
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame_count = 0
        self.process_every_n_frames = process_every_n_frames
        self.face_locations = []  # coordinates of detected faces
        self.face_names = []

    def process_frame(self, frame):
        self.face_locations, self.face_names = self.sfr.detect_known_faces(frame)

    def run(self):
        while True:
            ret, frame = self.cam.read()  # read a frame from the camera
            frame = cv2.flip(frame, 1)  # flip horizontally to un-mirror the image
            self.frame_count = self.frame_count + 1

            # check if the current frame should be processed
            if self.frame_count % self.process_every_n_frames == 0:
                self.process_frame(frame)

            for face_loc, name in zip(self.face_locations, self.face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(
                    frame,
                    name,
                    (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 0),
                    2,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # display resulting frame
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:  # 27 is the ASCII code for the Esc key
                break

        self.cam.release()  # release the camera so other applications can use it
        cv2.destroyAllWindows()  # closes all OpenCV windows


if __name__ == "__main__":
    app = FaceRecognition()
    app.run()
