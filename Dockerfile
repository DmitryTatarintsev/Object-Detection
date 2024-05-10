
FROM python:3.10.12

# Установка конкретных версий библиотек
RUN pip install pip==23.1.2

RUN apt-get update && apt-get install -y build-essential cmake pkg-config libjpeg-dev libwebp-dev libpng-dev libtiff-dev libopenjp2-7-dev libglib2.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
RUN unzip opencv.zip
RUN mv opencv-4.8.0 opencv

RUN pip install ultralytics==8.2.2 Pillow==9.4.0
RUN pip install ffmpeg-python==0.2.0 datetime==5.5
RUN pip install fastapi==0.110.2 uvicorn==0.29.0
RUN pip install python-multipart==0.0.9 ngrok==1.2.0 
RUN pip install numpy==1.25.2 matplotlib==3.7.1
RUN pip install pyngrok==7.1.6 protobuf==3.20.3
RUN pip install vidgear==0.3.2

# Создание opencv директории
WORKDIR /opencv
RUN mkdir build
WORKDIR /opencv/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF ..
RUN make -j$(nproc)
RUN make install

# Создание рабочей директории
WORKDIR /app
# Копирование содержимого каталога в директорию приложения
COPY detection.py /app/
COPY best.pt /app/
COPY main.py /app/

EXPOSE 8000

# Определение команды для запуска приложения внутри контейнера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Флаг --host 0.0.0.0 отвечает за то, чтобы наше веб-приложение работало на всех сетевых интерфейсах контейнера на порту 80.

# Дополнительные команды (оставьте только те, которые нужны)
RUN apt-get update && apt-get install -y mesa-common-dev
RUN apt-get update && apt-get upgrade -y
