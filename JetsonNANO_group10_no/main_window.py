from datetime import datetime
import time
import os
import random
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, pyqtSlot, QTimer, QPropertyAnimation
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QVBoxLayout, QFrame
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QIcon
from Camera import CameraApp
import numpy as np
import torch.nn as nn
import torch



CONFIDENCE_THRESHOLD = 0.7


class MainWindow(QMainWindow):
    save_response_time_emitter = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._window = None
        self.questions = ["0", "5", "10", "15", "20"]
        self.ui_setup()
        self.setup_event_connection()

    @property
    def window(self):
        """MainWindow widget."""
        return self._window

    def ui_setup(self):
        loader = loadUi('./user_interface/form/main_window.ui')
        self._window = loader
        self._window.installEventFilter(self)

        self._window.setWindowIcon(QIcon('./user_interface/media/icon.png'))
        self._window.setWindowTitle("Jetson Nano Group 10")

        self.camera_display()
        self.init_gesture_display()
        self.init_other_display()
        self.set_display()
        self.load_images()
        self.game_timer = QTimer(self)
        self.game_timer.setInterval(3000)
        self.game_timer.timeout.connect(self.show_random_image)
        self.predict_timer = QTimer(self)
        self.predict_timer.setInterval(500)
        self.predict_timer.timeout.connect(self.decision)
        self.model_init()
        self.init_test_txt_btn()

    def model_init(self):
        self.model = SimpleNN(24, 2)
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()

        self.labels = ['five', 'zero']
        self.right_label = None
        self.left_label = None

    def setup_event_connection(self):
        self.btn_start.clicked.connect(self.btn_click_start)
        self.btn_pause.clicked.connect(self.btn_click_pause)
        self.btn_restart.clicked.connect(self.btn_click_restart)

    def camera_display(self):
        self.camera_app = CameraApp()
        self.camera_app.setParent(self._window.cameraFrame)
        self.camera_app.setGeometry(0, 0, self._window.cameraFrame.width(), self._window.cameraFrame.height())
        self.camera_app.set_Size(self._window.cameraFrame.width(), self._window.cameraFrame.height())
        self.camera_app.show()

    def init_gesture_display(self):
        pixmap = QPixmap("./user_interface/media/beer_gameStart.jpg")
        frame_size = self._window.gestureFrame.size()
        pixmap = pixmap.scaled(frame_size)

        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setParent(self._window.gestureFrame)

    def init_other_display(self):
        self.score = 0
        self.score_label = QLabel(f"Score: {self.score}\t  ")
        self.score_label.setStyleSheet(
            "QLabel {"
            "   font-size: 20px;"  # 字體大小
            "   color: Red;"  # 字體顏色
            "   border:2px solid black;"  # 邊框
            "   padding: 8px;"  # 內框距
            "   border-radius: 5px;"  # 邊框圓角
            "   background-color: white;"  # 背景顏色
            "}"
        )
        self.score_label.setParent(self._window.scoresheetFrame)
        self._window.scoresheetFrame.raise_()

        self.predict_label_L = QLabel("Predict left hand:\t\t\t\t\t")
        self.predict_label_L.setStyleSheet("font-size: 16px;")
        self.predict_label_L.setParent(self._window.predictFrame_L)

        self.predict_label_R = QLabel("Predict right hand:\t\t\t\t\t")
        self.predict_label_R.setStyleSheet("font-size: 16px;")
        self.predict_label_R.setParent(self._window.predictFrame_R)

        self._window.predictFrame.raise_()

        self.question_label = QLabel("Question: None")
        self.question_label.setStyleSheet(""
                                          "font-size: 20px;"
                                          "font-weight: bold;"
                                          "border: 2px solid black;"
                                          "padding: 5px;"
                                          "")
        self.question_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.question_label, alignment=Qt.AlignCenter)
        self._window.questionFrame.setLayout(layout)

        self.question_label.setParent(self._window.questionFrame)

        self._window.questionFrame.raise_()

        #     Remove border of frame
        self._window.cameraFrame.setStyleSheet("border: none;")
        self._window.gestureFrame.setStyleSheet("border: none;")
        self._window.questionFrame.setStyleSheet("border: none;")
        self._window.scoresheetFrame.setStyleSheet("border: none;")
        self._window.predictFrame.setStyleSheet("border: none;")
        self._window.predictFrame_L.setStyleSheet("border: none;")
        self._window.predictFrame_R.setStyleSheet("border: none;")
        self._window.setFrame.setStyleSheet("border: none;")

    def set_display(self):
        self.btn_start = QPushButton("Start", self)
        self.btn_pause = QPushButton("Pause", self)
        self.btn_pause.hide()
        self.btn_restart = QPushButton("Restart", self)
        self.btn_restart.hide()

        # size of button
        button_width = 300
        button_height = 60
        self.btn_start.setFixedSize(button_width, button_height)
        self.btn_pause.setFixedSize(button_width, button_height)
        self.btn_restart.setFixedSize(button_width, button_height)

        # style of button
        button_style = (
            "QPushButton {"
            "   border-radius: 10px;"  # 圓角
            "   background-color: #4CAF50;"  # 背景顏色
            "   color: white;"  # 文字顏色
            "   font-size: 20px;"  # 文字大小
            "}"
            "QPushButton:hover {"
            "   background-color: #45a049;"  # 滑鼠懸停時的背景顏色
            "}"
        )
        self.btn_start.setStyleSheet(button_style)
        self.btn_pause.setStyleSheet(button_style)
        self.btn_restart.setStyleSheet(button_style)

        v_layout = QVBoxLayout(self._window.setFrame)
        v_layout.addWidget(self.btn_start)
        v_layout.addWidget(self.btn_pause)
        v_layout.addWidget(self.btn_restart)

    # 代修正
    @pyqtSlot()
    def btn_click_start(self):
        self.game_timer.start()
        print("Button - Start")
        self.btn_start.hide()
        self.btn_pause.show()
        self.btn_restart.show()
        pixmap = QPixmap("./user_interface/media/game_start.png")
        self.image_label.setPixmap(pixmap)

    @pyqtSlot()
    def btn_click_pause(self):
        if (self.btn_pause.text() == "Pause"):
            print("Button - Pause")
            self.game_timer.stop()
            self.game_timer.setInterval(300)
            self.predict_timer.stop()
            self.btn_pause.setText("Resume")
        else:
            print("Button - Resume")
            self.game_timer.start()
            self.game_timer.setInterval(3000)
            self.predict_timer.start()
            self.btn_pause.setText("Pause")

    @pyqtSlot()
    def btn_click_restart(self):
        print("Button - Restart")
        self.restart()

    def restart(self):
        self.btn_start.show()
        self.btn_pause.hide()
        self.btn_restart.hide()
        self.score = 0
        self.score_label.setText(f"Score: {self.score}")
        self.game_timer.stop()
        self.predict_timer.stop()
        self.game_timer.setInterval(3000)

        pixmap = QPixmap("./user_interface/media/beer_gameStart.jpg")
        frame_size = self._window.gestureFrame.size()
        pixmap = pixmap.scaled(frame_size)

        self.predict_label_L.setText("Predict right hand:")
        self.predict_label_R.setText("Predict left hand:")

        self.question_label.setText("Question: None")

        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setParent(self._window.gestureFrame)
        self._window.gestureFrame.raise_()
        self.image_label.show()

    def update_score_label(self):
        self.score_label.setText(f"Score: {self.score}")

    # def load_audio(self):
    #     pygame.mixer.init()
    #     pygame.mixer.music.load('./user_interface/media/audio/ahh.wav')
    #     pygame.mixer.music.load('./user_interface/media/audio/yeah.wav')

    def load_images(self):
        self.gesture_images = []
        base_path = './user_interface/media/gestures'
        categories = ['0', '5', '10']

        for category in categories:
            path = os.path.join(base_path, category)
            for img in os.listdir(path):
                self.gesture_images.append((os.path.join(path, img), category))

    def show_random_image(self):

        self.start_time = time.time()
        # self.game_timer.setInterval(3000)
        img_path, category = random.choice(self.gesture_images)
        pixmap = QPixmap(img_path)
        self.image_label.setPixmap(pixmap)
        # print(f"Figure: {img_path}, Dir: {category}")
        self.current_category = category

        if category == "0":
            random_number = random.randint(0, 2)
        elif category == "5":
            random_number = random.randint(1, 3)
        else:
            random_number = random.randint(2, 4)

        q_text = f"Question: {self.questions[random_number]}"
        self.question_label.setText(q_text)
        self.randQ = self.questions[random_number]

        self.predict_timer.start()

    # 從Camera抓圖進來做判斷
    def capture_and_predict(self):
        hands_label = ["None", "None"]
        try:
            skeleton_data = self.camera_app._skeleton_numpy()  # landmarks has already been converted into numpy
            for hand_label, landmarks in skeleton_data:
                hand_label = "Left" if hand_label else "Right"

                distances = self.compute_distances(landmarks)
                distances_tensor = torch.tensor([distances], dtype=torch.float32)

                # prediction = self.clf.predict(distances)
                # confidence = np.max(self.clf.predict_proba(distances))
                with torch.no_grad():
                    outputs = self.model(distances_tensor)
                    _, prediction = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, dim=1).max().item()

                label = self.labels[prediction[0]]
                if hand_label == "Right":
                    if confidence >= CONFIDENCE_THRESHOLD:
                        self.right_label = f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)"
                        print(f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)")
                    else:
                        self.right_label = f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)"
                    hands_label[1] = label
                elif hand_label == "Left":
                    if confidence >= CONFIDENCE_THRESHOLD:
                        self.left_label = f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)"
                        print(f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)")
                    else:
                        self.left_label = f"{hand_label} Hand: {label} ({confidence * 100:.2f}%)"
                    hands_label[0] = label
            return hands_label

            # # 獲取預測類別
            # prediction = np.argmax(output.cpu().numpy(), axis=1)
            #
            # user_move_name = self.categories[prediction[0]]
            # return user_move_name
            #
        except Exception as e:
            print(e)

    @pyqtSlot()
    def decision(self):
        predicted_hands_label = self.capture_and_predict()
        self.predict_label_L.setText(self.left_label)
        self.predict_label_R.setText(self.right_label)
        img_category = self.current_category
        question = self.randQ
        # print(f"Left predict: {predicted_hands_label[0]}, Rifht predict: {predicted_hands_label[1]}")

        if predicted_hands_label[0] == "None" and predicted_hands_label[1] == "None":
            self.predict_label_L.setText("No hand detected")
            self.predict_label_R.setText("")
        elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
            self.predict_label_L.setText("Only one hand detected")
            self.predict_label_R.setText("")

        if img_category == "0" and question == "0":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "zero":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "0" and question == "5":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "zero":
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

            elif predicted_hands_label[0] == "five" and predicted_hands_label[1] == "five":
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                # play ahh (lose) sound
                pass

        elif img_category == "0" and question == "10":
            if predicted_hands_label[0] == "five" and predicted_hands_label[1] == "five":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "5" and question == "5":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "zero":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "5" and question == "10":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "five":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "five" and predicted_hands_label[1] == "zero":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()

                self.end_time = time.time()
                self.save_response_time_emitter.emit()
        elif img_category == "5" and question == "15":
            if predicted_hands_label[0] == "five" and predicted_hands_label[1] == "five":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "10" and question == "10":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "zero":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "10" and question == "15":
            if predicted_hands_label[0] == "zero" and predicted_hands_label[1] == "five":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "five" and predicted_hands_label[1] == "zero":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

        elif img_category == "10" and question == "20":
            if predicted_hands_label[0] == "five" and predicted_hands_label[1] == "five":
                # play ahh (lose) sound
                pass
            elif predicted_hands_label[0] == "None" or predicted_hands_label[1] == "None":
                # play ahh (lose) sound
                pass
            else:
                self.score += 1
                self.update_score_label()
                # play yeah (win) sound

                self.end_time = time.time()
                self.save_response_time_emitter.emit()

    def compute_distances(self, landmarks):
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

    def init_test_txt_btn(self):
        self.save_response_time_emitter.connect(self.write_txt)
        self.click_count = 0
        self.create_txt()

    def create_txt(self):
        if not os.path.exists('results'):
            os.makedirs('results')
        date = datetime.now().strftime('%Y-%m-%d')
        file_path = f'results/{date}.txt'
        counter = 1
        while os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_{counter}{ext}"
            counter += 1
        self.file_path = file_path

    @pyqtSlot()
    def write_txt(self):
        self.click_count += 1
        with open(self.file_path, 'a') as f:
            if self.click_count == 1:
                f.write(f"Recorded Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            # 改成time segment
            f.write(f"{self.click_count}\t{self.end_time - self.start_time}\n")


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
