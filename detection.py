# !pip -q install ultralytics
# !pip -q install ffmpeg-python

import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import ffmpeg
import datetime

from PIL import Image
from ultralytics import YOLO
from datetime import timedelta
from collections import Counter

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')
# новые названия классов
mn = {0: 'Каска',
 1: 'Перчатки',
 2: 'Обувь',
 3: 'Одежда',
 4: 'Курение',
 5: 'Каска. Нарушение',
 6: 'Перчатка. Нарушение',
 7: 'Обувь. Нарушение',
 8: 'Одежда. Нарушение',
 9: 'Рабочий'}

# время кадра
def frame_time(milliseconds):
  seconds = milliseconds / 1000
  minutes, seconds = divmod(seconds, 60)
  hours, minutes = divmod(minutes, 60)
  return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# Переводим секунды в кадр
def time_to_seconds(time_str, fps=20):
  # Разделяем строку на части: часы, минуты и секунды
  h, m, s = time_str.split(':')
  # Преобразуем каждую часть в целое число
  h, m, s = int(h), int(m), int(s)
  # Переводим часы, минуты и секунды в секунды
  total_seconds = h  *  3600 + m  *  60 + s
  # Переводим секунды в кадр
  return int(total_seconds*fps)

# прогноз по видео
def predict(video, step=None, start=None, finish=None):
  conf = {}
  conf['n_frame'] = list() # порядковый номер кадра в цикле с пропусками
  n_frame = 0 # номер первого кадра
  conf['CAP_PROP_POS_FRAMES'] = list() # номер кадра из видео
  conf['boxes_cls'] = list()  # классы в кадре
  conf['batch'] = list() # значение > 0, когда в кадре нарушение
  conf['frame_time'] = list() # время кадра

  # Open the video file
  cap = cv2.VideoCapture(video)

  # Get the total number of frames in the video
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Set the start frame
  if start == None:
    start = 0
  else: 
    start = time_to_seconds(start, fps=cap.get(cv2.CAP_PROP_FPS))
  cap.set(cv2.CAP_PROP_POS_FRAMES, start)

  # Set the last frame
  if finish == None:
    finish = total_frames
  else: 
    finish = time_to_seconds(finish, fps=cap.get(cv2.CAP_PROP_FPS))

  # Step
  if step == None:
    step = 600

  # Loop through the video frames
  while cap.isOpened():
      # Read a frame from the video
      success, frame = cap.read()
      if success:
          if cap.get(cv2.CAP_PROP_POS_FRAMES) in range(start, finish, step):
            # Получить время текущего кадра
            conf['frame_time'].append(frame_time(cap.get(cv2.CAP_PROP_POS_MSEC)) )
            # номер кадра из видео
            conf['CAP_PROP_POS_FRAMES'].append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # порядковый номер кадра в цикле с пропусками
            conf['n_frame'].append(n_frame)
            n_frame += 1
            # Run YOLO inference on the frame
            results = model(frame)
            # классы на кадр
            rbc = results[0].boxes.cls
            conf['boxes_cls'].append(dict(Counter([mn[int(i)] for i in sorted(rbc)])))
            # серия кадров с классами нарушенений
            if len([x for x in rbc if x in [4, 5, 6, 7, 8]]) > 0:
              if len(conf['batch']) == 0: conf['batch'].append(1)
              else: conf['batch'].append(conf['batch'][-1]+1)
            else: conf['batch'].append(0)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
      else:
          # Break the loop if the end of the video is reached
          break
  # Release the video capture object and close the display window
  cap.release()
  cv2.destroyAllWindows()
  return conf

# функция возвращает итоговый результат
def fr(conf):
  frame_range = dict()
  frame_range['start'] = list()
  frame_range['final'] = list()
  frame_range['alert'] = list()

  if conf['batch'][0] == 1:
    frame_range['start'].append(conf['frame_time'][0])

  for x in range(1,len(conf['batch'])):
    if conf['batch'][x] == 1:
      frame_range['start'].append(conf['frame_time'][x])
    if conf['batch'][x] < conf['batch'][x-1]:
      frame_range['final'].append(conf['frame_time'][x-1])

  if len(frame_range['start']) > len(frame_range['final']):
    frame_range['final'].append(conf['frame_time'][-1])

  def alerts(i):
    start = frame_range['start'][i]
    finish = frame_range['final'][i]
    lst = []
    if start != finish:
      for x in conf['boxes_cls'][conf['frame_time'].index(start) : conf['frame_time'].index(finish)]:
        lst += x.keys()
    if start == finish:
      for x in conf['boxes_cls'][conf['frame_time'].index(finish)]:
        lst += [x]

    lst = np.unique([x for x in lst if x.endswith('Нарушение') or x=="Курение"]).tolist()
    return ', '.join(lst).replace(".", "").replace(" Нарушение", "").lower()

  k = 0 
  for i in range(len(frame_range['start'])):
    start_finish = [frame_range[x][i] for x in ['start', 'final']]
    frame_range['alert'].append(f'Время {start_finish[0]} - {start_finish[1]}, нарушения: {alerts(i)}.')
    k += 1

  return frame_range['alert']

def video_detection(video, frame_step=None, time_start=None, time_finish=None):
  return fr(predict(video, frame_step, time_start, time_finish))  