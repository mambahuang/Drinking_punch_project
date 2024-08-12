import sys
import cv2
import mediapipe as mp
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import numpy as np
import threading

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the QLabel to display the video feed
        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Initialize OpenCV capture
        self.capture = cv2.VideoCapture(0)  # 0 is typically the default camera

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        # Timer to refresh the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        camera_thread = threading.Thread(target=self.update_frame)
        camera_thread.start()
        self.timer.start(50)  # Update every 50 ms

    def update_frame(self):
        self.all_skeleton = []
        ret, frame = self.capture.read()
        if ret:
            # Process the frame with MediaPipe Hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Draw the hand landmarks on the frame
            if results.multi_hand_landmarks:
                for index,hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.all_skeleton.append([index,hand_landmarks])
            # self.skeleton_with_background = white_bg
            # Convert the frame to QImage for display
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.mirrored_image = q_image.mirrored(True, False)
            # Update the QLabel with the new QImage
            self.image_label.setPixmap(QPixmap.fromImage(self.mirrored_image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def set_Size(self, width, height):
        self.image_label.setGeometry(0, 0, width, height)
        self.width = width
        self.height = height

    def _skeleton_numpy(self):
        return self.all_skeleton

    def qimage_to_numpy(self,qimage):
        '''将 QImage 转换为 NumPy 数组'''
        # 转换 QImage 到 QImage.Format.Format_RGB888 格式
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)

        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        # 将字节数据转换为 NumPy 数组
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))

        # OpenCV 使用 BGR 格式，可能需要颜色通道转换
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return arr
    def only_used_if_run_this_code(self):
        self.setGeometry(100, 100, 800, 600)
        self.image_label.setGeometry(0, 0, 800, 600)

    def compute_distances(self,landmarks):
        # Define pairs for distance calculation
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
                 (0, 5), (0, 6), (0, 7), (0, 8),
                 (0, 9), (0, 10), (0, 11), (0, 12),
                 (0, 13), (0, 14), (0, 15), (0, 16),
                 (0, 17), (0, 18), (0, 19), (0, 20),
                 (4, 8), (8, 12), (12, 16), (16, 20)]

        distances = []
        reference_distance = np.linalg.norm(
            np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]) -
            np.array([landmarks.landmark[9].x, landmarks.landmark[9].y])
        )

        for pair in pairs:
            p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
            p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
            distance = np.linalg.norm(p1 - p2)
            distances.append(distance / reference_distance)  # Normalize the distance using the reference distance

        return distances

if __name__ == '__main__':

    app = QApplication(sys.argv)
    camera_app = CameraApp()
    camera_app.show()
    camera_app.only_used_if_run_this_code()
    sys.exit(app.exec_())
