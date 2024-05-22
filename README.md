<div>
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/wbkms.jpg" alt="альтернативный текст" title="заголовок изображения" width="100%" style="float: left; margin-right: 10px">
</div>

# Детекция средств индивидуальной защиты и нарушений

**Веб-сервис (Демо версия)**: [Детекция средств индивидуальной защиты и нарушений по снимкам. РАБОЧАЯ ССЫЛКА](https://huggingface.co/spaces/dmitry212010/siz-detection-demo-streamlit)

Описание: веб-приложение принимает на вход снимок и возвращает его с детектированными объектами. Все детектированные объекты выделяются в прямоугольником (баундинг боксом), но только нарушения имеют дополнителую подпись над боксом.

**FAST API Детекция по видео (Рабочая версия)**: 

Описание: веб-приложение принимает на вход видео и возвращает текстовое сообщение о нарушении сиз и курении с указанием времени.

**Статус проекта: ЗАВЕРШЕНО.**

**Стэк**: *cv2, PIL, numpy, matplotlib, os, moviepy, vidgear, zipfile, sys, google, ultralytics. (описание ниже)*

**Цель:** написать модель нейронной сети по детекции средств индивидуальной защиты и нарушений.

**План:** *анализ стека разработки **-->** парсинг датасета **-->** создание демонстрационной версии*

**Идея:** одна модель YOLOv9-C, без каскада, без черных прямоугольников на человека, пытается в одиночку искать наличие и нарушение правильного ношения сизов. 

Для этого созданы две большие группы классов: "нарушение" и "правильное ношение" условно. 

Так же будет создан отдельный скрипт по верх обученной модели, который делает невидимым потенциально неправильно определенные баундиг боксы, за переделами бокса "Рабочие".

**Классы:**

- **nc**: 10

- **names:** ['Каска', 'Перчатка', 'Обувь', 'Одежда', 'Курение', 'Каска. Нарушение', 'Перчатка. Нарушение', 'Обувью Нарушение', 'Одежда. Нарушение', 'Рабочие']

**Файлы:**

- *Парсинг.*

- - [`create_frames.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/create_frames.ipynb): генерация изображений размером 1280x1280 на основе идентификаторов аннотаций и кадров размеченного видео. Разбивается на выборки. Запускается индивидуально для каждого видео. Затем папки train-val-test вручную переносятся в общий каталог.
- - [`data.yaml`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/data.yaml): файл конфигурации обучения Yolo 9. Содержит информацию по классам и массиву данных в целом.
- - [`sample_size_reduction.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/sample_size_reduction.ipynb): алгоритм сжатия массива данных по числу и размеру снимков. Запускается после `create_frames.ipynb` и объединения.
- - [`Augmentations.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Augmentations.ipynb): алгоритм аугментации тренировочных данных. Запускается после `sample_size_reduction.ipynb`.
- - [`requirements.txt`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/requirements.txt): актуальные версии библиотек.

- *Обучение.*

- - [`main_Yolo9_SIZ_.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/main_Yolo9_SIZ_.ipynb): обучени YOLOv9-C, без каскада, без черных прямоугольников на человека на наличие нарушений и правильного ношения сизов.

- *Веб-сервис.*

- - [`Web_service.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Web_service.ipynb): алгоритм демо версии веб-сервиса.
- - [`app.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/app.py): код модели под Streamlit app для детекции снимков.
- - [`Dockerfile_streamlit.txt`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Dockerfile_streamlit.txt): блокнот-Dockerfile для создания образа на основе файлов этого репозитория.

- - [`Demoversion.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Demoversion.ipynb): приложение fast api.
- - [`Dockerfile`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Dockerfile): Dockerfile для создания образа на основе файлов этого репозитория.
- - [`main.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/main.py): код Fast API.
- - [`detection.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/detection.py): код модели под Fast API для детекции видео.

 
### Streamlit Детекция по снимкам (Демо версия). Веб-сервис.  Инструкция:
В этом каталоге лежит dockerfile для активации образа приложения. Но что бы не создавать путаницы с основной рабочей версией fatsapi приложения. Этот образ просто записан в блокнот [`Dockerfile_streamlit.txt`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Dockerfile_streamlit.txt). Для создания этого образа нужно удалить другой Dockerfile этого каталога, а Dockerfile_streamlit.txt переименовать в Dockerfile.
Так же, после скачивания директори вы можете без docker активировать приложение командой в cmd Prompt:
```cmd Prompt
C:\Users\...>git clone https://github.com/NeuronsUII/Zavod_Ural_n.git
C:\Users\...>cd "C:\Users\...\Zavod_Ural_n\dtatarintsev"
C:\Users\...\Zavod_Ural_n\dtatarintsev>python -m pip install streamlit==1.33.0
C:\Users\...\Zavod_Ural_n\dtatarintsev>python -m streamlit run app.py
Ваш основной браузер автоматически запустит http://localhost с приложением.
Для отключения приложения достаточно выйти из браузера или в консоле нажав на кнопки клавиатуры "Cntrl" + "C"
```

**Средняя скорость работы модели без визулализации на снимок:**
```
0: 960x1280 1 Каска, 2 Перчаткаs, 2 Обувьs, 3 Одеждаs, 4 Каска. Нарушениеs, 1 Обувью Нарушение, 2 Одежда. Нарушениеs, 5 Рабочиеs, 86.8ms
Speed: 30.1ms preprocess, 86.8ms inference, 2.3ms postprocess per image at shape (1, 3, 960, 1280)
```
• 0: Индекс изображения

• 960x1280: Размеры изображения

• 1 Каска, 2 Перчаткаs, 2 Обувьs, 3 Одеждаs, 4 Каска. Нарушениеs, 1 Обувью Нарушение, 2 Одежда. Нарушениеs: Обнаруженные объекты и их количество

• 86.8ms: Время вывода на одном изображении

**Общее время на один снимок:**

• 30.1 мс + 86.8 мс + 2.3 мс = 119.2 мс

**Итоговая скорость на один снимок:**

1000 мс / 119.2 мс = 8.38 кадра в секунду

Эти показатели означают, что модель YOLO обнаружила в данном изображении в общей сложности 15 объектов (1 каску, 2 перчатки, 2 пары обуви, 3 предмета одежды, 4 нарушения касок, 1 нарушение обуви и 2 нарушения одежды). Для обработки этого изображения и вывода результатов потребовалось 119,2 мс, что соответствует скорости 8,38 кадров в секунду.

**Разбивка времени:**

• 30,1 мс: Предварительная обработка (изменение размера и нормализация изображения)

• 86,8 мс: Вывод (выполнение модели и обнаружение объектов)

• 2,3 мс: Постобработка (фильтрация и нерабочие операции)

Время вывода по-прежнему является наиболее важным показателем, и в данном случае оно составляет 86,8 мс, что также довольно быстро.


### FAST API Детекция по видео (Рабочая версия). Инструкция:
В каталоге лежит докер файл со всеми необходимыми зависимостями для создания докер образа на основе файлов этой директории Zavod_Ural_n/dtatarintsev. Так же, после скачивания директори вы можете без docker активировать приложение командой в cmd Prompt:
```cmd Prompt
C:\Users\...>git clone https://github.com/NeuronsUII/Zavod_Ural_n.git
C:\Users\...>cd "C:\Users\...\Zavod_Ural_n\dtatarintsev"
C:\Users\...\Zavod_Ural_n\dtatarintsev>python -m pip install uvicorn==0.29.0
C:\Users\...\Zavod_Ural_n\dtatarintsev>python -m uvicorn main:app
Далее запустить в любом браузере http://localhost:8000/docs
После, вы можете отключить приложение в консоле нажав на кнопки клавиатуры "Cntrl" + "C"
```

<div>
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/fastapi.png" alt="альтернативный текст" title="заголовок изображения" width="650" style="float: left; margin-right: 10px">
</div>

**Пример запроса**
```
import requests

url = 'http://localhost:8000/predict'
files = {'file': open('test.mp4', 'rb')}
params = {'frame_step': 150, 'start_time': '00:00:10', 'stop_time': None}
response = requests.post(url, files=files, params=params)
print(response)
print(response.json())
```
```
<Response [200]>
{'test.mp4, от 00:00:10 до конца c шагом кадра 150.': ['Время 00:00:17 - 00:00:24, нарушения: каска, обувь, одежда.']}
```

**files** - здесь указывается точный адрес до видео файла. Например, `video.mp4`.

**url** - локальный адрес для запроса приложения. Менять не желательно.

***params***

**frame_step:**
При `None` будет определено значение 600.
Определяет шаг между кадрами для детекции. Влияет на скорость и качество. Если frame_step = 1, то будет детекрирован каждый кадр.

- каждый 50 кадр видео: для серъезных вычислительных мощностей.
- каждый 200 кадр видео: занимает много времени, сильная детекция. Желательно GPU.
- каждый 600 кадр видео: оптимально. Баланс между скоростью и качеством для любых вычислительных мощностей. Детекция каждые 30 секунд видео.
- каждый 1200 кадр видео: очень быстро, слабая детекция, для слабых вычислительных мощностей. Например, CPU. Детекция 1 раз в минуту.

Значение frame_step не должно превышать, число кадров во всем видео. Однако, получить такую ошибку будет не просто. 1200 кадров при fps 20 - это 1 минута видео.

**start_time**
При `None` будет определено значение первого кадра, '00:00:00'.
Задает время откуда начнется детекция. Например, значение start_time='00:30:00' начнет детекцию видео на тридцатой минуте видео.

**stop_time**
При `None` будет определено значение последнего кадра.
Задает время где закончится детекция. Например, значение stop_time='02:01:30' остановит детекцию видео на втором часу первой минуты и тридцати секунд видео.

**Внимание! Параметры не влияют на скорость загрузки видео в приложение.**

**requests.post** - алгоритм запроса к приложению.

**print(response)**

<Response 200> - запрос на URL /predict обрабатывается успешно и возвращает код 200, что указывает на то, что функция /predict работает корректно.

<Response 400> - запросы на URL / и /favicon.ico возвращают код 404, что означает, что эти страницы не найдены. Это может быть связано с тем, что вы не настроили обработку этих URL в вашем приложении.

Ошибка <Response 500> указывает на внутреннюю ошибку сервера. Возможно, сервер, к которому вы обращаетесь, временно недоступен или возникла ошибка при обработке вашего запроса.
Ошибка JSONDecodeError говорит о том, что сервер вернул ответ, который невозможно интерпретировать как корректный JSON. Это может произойти, если сервер вернул неправильный формат данных или ответ вообще не содержит данных.

**print(response.json())** - результат работы модели. Возвращает словарь с указанными параметрами и названием видео в ключе и списком всех детектированных нарушений.


### Качество работы модели на тестовых данных:

<div style="display: flex;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/confusion_matrix.png" alt="Image 1" style="width: 500px; height: auto;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/confusion_matrix_normalized.png" alt="Image 2" style="width: 500px; height: auto;">
</div>

<div style="display: flex;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/F1_curve.png" alt="Image 1" style="width: 500px; height: auto;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/PR_curve.png" alt="Image 2" style="width: 500px; height: auto;">
</div>

<div style="display: flex;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/P_curve.png" alt="Image 1" style="width: 500px; height: auto;">
    <img src="https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/R_curve.png" alt="Image 2" style="width: 500px; height: auto;">
</div>


*Описание стэка:*

- `cv2`: *библиотека OpenCV для компьютерного зрения. Используется для обработки изображений и видео, включая чтение, запись, изменение размера, фильтрацию и многое другое.*
- `PIL`: *библиотека Python Imaging Library. Используется для работы с изображениями, включая открытие, изменение размера, обрезку, наложение фильтров и другие операции.*
- `numpy`: *библиотека для работы с многомерными массивами и математическими операциями над ними. Часто используется для обработки изображений и видео.*
- `matplotlib`: *библиотека для создания графиков и визуализации данных. Используется для отображения изображений, графиков и диаграмм.*
- `os`: *модуль Python для работы с операционной системой. Используется для выполнения операций с файлами и директориями, таких как чтение, запись, перемещение и удаление.*
- `moviepy`: *библиотека для обработки видео. Используется для создания, редактирования и конвертации видеофайлов.*
- `vidgear`: *библиотека для захвата видео с камеры или видеофайла. Используется для получения видеопотока для дальнейшей обработки.*
- `zipfile`: *модуль Python для работы с ZIP-архивами. Используется для создания, чтения и распаковки ZIP-файлов.*
- `sys`: *модуль Python, предоставляющий доступ к некоторым переменным и функциям, связанным с интерпретатором Python. Используется для работы с аргументами командной строки и другими системными функциями.*
- `ultralytics`: *библиотека для работы с YOLOv9.*
- `google`: *библиотека для загрузки файлов с гугл драйв диска.*

### Расписание задач проекта.
**07.02-21.02.2024:**
1) Зарегистрироваться в github (trello -опционально). - **Выполнено.**
2) Создать план реализации проекта. Анализ сторонних проектов. - **Выполнено.**
3) Приступить к программе стажировки. - **Выполнено.**

**21.02-28.02.2024:**
1) Поиск сторонних датасетов. - **Выполнено.**

*Все кроме курения: https://universe.roboflow.com/personal-protective-equipment/ppes-kaxsi*

*Детекция курения: [ссылка 1](https://www.kaggle.com/datasets/ahmedgamal12/smoke-detection?select=smoke), [ссылка 2](https://www.kaggle.com/datasets/sujaykapadnis/smoking).*

**28.02-06.03.2024:**
1) [Тест Yolo на стороннем массиве данных roboflow. Без сохранений.](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/testing_SIZ_yolov8_model.ipynb)  - **Выполнено.**
2) Анализ SAM и Grounding DINO - **Выполнено.**

**06.03-13.03.2024:**
1) Анализ API парсинга. LabelImg, Roboflow, CVAT. - **Выполнено.**
2) Анализ датасета заказчика. Разбиение на клипы. - **Выполнено.**

**13.03-20.03.2024:**
1) Анализ фреймворков парсинга кадров. Terra, OpenCV, MoviePy, Vidgear. - **Выполнено.**

**20.03-27.03.2024:**
1) Анализ методов интеграции в production. Requests, FastApi, Веб-сервер.  - **Выполнено.**

**27.03-03.04.2024:**
1) Анализ готовых решений huggingface.co и Pytorch Hub. - **Выполнено.**
2) Анализ двухэтапных Faster R-CNN, R-CNN и 
   одноэтапных моделей YOLO (You Only Look Once), SSD (Single Shot Multibox Detector), RetinaNet. - **Выполнено.**

*Сторонние проекты (для сравнения):
[ссылка 1](https://www.kaggle.com/code/alincijov/safety-helmets-startup-notebook),
[ссылка 2](https://www.kaggle.com/code/harpdeci/yolo-nas-safety-helmet-and-vest-detection),
[ссылка 3](https://www.kaggle.com/code/muhammetzahitaydn/yolo-helmet-vest-detection),
[ссылка 4](https://www.kaggle.com/code/ilhansevval/yolo-nas-safety-helmet-and-jacket-detection),
[ссылка 5](https://www.kaggle.com/code/muhammetzahitaydn/yolo-helmet-vest-detection),
[ссылка 6](https://www.kaggle.com/code/plasticglass/yolov8-safety-helmet-detection),
[ссылка 7](https://universe.roboflow.com/digitalimage-114q6/safety-helmet-q3b8o/model/10).*

**03.04-10.04.2024:**
1) Разметка клипов в CVAT из датасета заказчика. - **Выполнено.**

**10.04-17.04.2024:**
1) Создание скриптов парсинга [`create_frames.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/create_frames.ipynb), [`sample_size_reduction.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/sample_size_reduction.ipynb), [`Augmentations.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Augmentations.ipynb) - **Выполнено.**
2) Обучение модели. [`main_Yolo9_SIZ_.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/main_Yolo9_SIZ_.ipynb) - **Выполнено.**
3) Создание демонстрационной версии. [Web_service.ipynb](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/siz_detection/Web_service.ipynb) - **Выполнено.**

*Ожидаемые результаты (демонстрационная версия) -это работающая пилотная система с удобным для тестирования интерфейсом (в виде ipynb
ноутбука с моделями и разработанным алгоритмом, пилотного консольного, десктопного
либо веб-приложения), обеспечивающая проверку работоспособности и функциональность
технологий и решений, а также позволяющая сделать оценку необходимой аппаратной
конфигурации серверной части разрабатываемой системы.*

**17.04-01.05.2024 (Крайний срок):**
1) Внесение финальных поправок. Сдача проекта. - **Выполнено.**
- - [`Видео перезентация`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/zavod_ural_dtatarintsev.mp4) работы приложений. Сохранение [`Readme в pdf`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Zavod_Ural_n_dtatarintsev.pdf)
- - [`Web_service.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Web_service.ipynb): дополнен алгоритм демо версии веб-сервиса.
- - [`app.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/app.py): дополнен код модели под Streamlit app для детекции снимков.
- - [`Dockerfile_streamlit.txt`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Dockerfile_streamlit.txt): создан блокнот-Dockerfile для создания образа на основе файлов этого репозитория.

- - [`Demoversion.ipynb`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Demoversion.ipynb): создано приложение fast api.
- - [`Dockerfile`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/Dockerfile): создан Dockerfile для создания образа на основе файлов этого репозитория.
- - [`main.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/main.py): создан код Fast API.
- - [`detection.py`](https://github.com/DmitryTatarintsev/internship/blob/main/siz_detection/detection.py): создан код модели под Fast API для детекции видео.


*Функциональные требования.*

- *формирование журнала нарушений с функцией поиска/фильтрации событий;*
- *формирование журнала событий с функцией поиска/фильтрации событий;*
- *выявление фактов нарушений (отсутствие СИЗ, корректность использования СИЗ) по
заданным критериям видеоаналитики);*
- *запись выявленного нарушения (события) в журнал событий с сохранением файла
видеофрагмента в архив;*
- *переход от журнала событий к зафиксированному нарушению с открытием файла
видеофрагмента;*
- *определение личности сотрудника;*
- *информирование различных категорий пользователей о факте нарушения по различным
каналам (электронная почта, мессенджер);*
- *хранение архивов видеофрагментов нарушений с глубиной не менее 10 календарных
дней;*
- *формирование отчетов о нарушениях за выбранный период;*
- *система должна иметь инструменты идентификации/аутентификации пользователей с
функционалом добавления/удаления пользователей;*
- *отчет о нарушениях за период с возможностью выгрузки во внешний файл.*
- *Система должна предусматривать возможность дообучения по реализации контроля
опасных зон, мониторинга вспомогательного оборудования.*
