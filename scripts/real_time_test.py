import datetime
import os
import cv2
import keras

from models import MODELS_PATH
from utils.utils import process_video

inception_model_path = os.path.join(MODELS_PATH, "inception3.h5")
model = keras.models.load_model(inception_model_path)

vcap = cv2.VideoCapture(0)

video_writer = cv2.VideoWriter(f"output{datetime.datetime.now().strftime('%H:%mm')}.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"), 8, (640, 480))

process_video(model, vcap, video_writer)
