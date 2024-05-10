@echo off
title Installing python libraries and create start.bat

:: Клонируем репозиторий
cd %~dp0
git clone https://github.com/DmitryTatarintsev/internship.git

:: Установка всех необходимых пакеток последней версии
pip install pip
pip install streamlit
pip install opencv-contrib-python
pip install opencv-python
pip install opencv-python-headless
pip install ultralytics
pip install Pillow
pip install ffmpeg-python
pip install datetime
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install ngrok
pip install numpy
pip install matplotlib
pip install pyngrok 
pip install protobuf
pip install vidgear

:: Создаем файл запуска приложения
if not exist "Start.bat" (
    echo @echo off >> Start.bat
    echo title Run Uvicorn >> Start.bat
    echo ^:: Переходим в папку internship/siz_detection >> Start.bat
    echo cd internship/siz_detection >> Start.bat
    echo ^:: Запускаем приложение uvicorn >> Start.bat
    echo python -m uvicorn main:app >> Start.bat
)

@ECHO OFF
DEL %0
