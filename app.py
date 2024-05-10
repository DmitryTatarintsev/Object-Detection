import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLO model
wght = 'best.pt'
model = YOLO(wght)
mn = {
    0: '',
    1: '',
    2: '',
    3: '',
    4: 'НАРУШЕНИЕ',
    5: 'НАРУШЕНИЕ',
    6: 'НАРУШЕНИЕ',
    7: 'НАРУШЕНИЕ',
    8: 'НАРУШЕНИЕ',
    9: ''
}

def greet(image):
    # Run inference on an image
    results = model(image)  # results list
    # Visualize the results
    for i, r in enumerate(results):
        r.names = mn
        # Plot results image
        im_bgr = r.plot(line_width=3, font_size=22, labels=True, conf=True)  # BGR-order numpy array
        im_rgb = im_bgr[..., ::-1]  # Convert BGR to RGB
    return im_rgb

def main():
    st.markdown("### YOLOv9-c и API Streamlit 1280x1280 с весами детекции СИЗ УРАЛ ЗАВОДА для территории: склада, снаружи склада, курилка день/ночь на ближней дистанции. Детекция снимков")
    st.write("YOLOv9-c - мощная универсальная одноэтапная модель детекции. Это самая быстрая и надежная версия YOLO от Ultralytics на 2024 год. Легко обучаема и настраиваема.")
    st.write("Streamlit - это API пользовательского интерфейса для обработки изображений, но его также можно адаптировать для работы с видео. И/или заменить на технический (Fast API, Django, Flask) для взаимодействия между программами протоколом HTTP.")
    st.write("")
    st.markdown("##### Классы:")
    st.write("- ['Каска', 'Перчатка', 'Обувь', 'Одежда', 'Рабочие']")
    st.write("- ['Курение', 'Каска. Нарушение', 'Перчатка. Нарушение', 'Обувью Нарушение', 'Одежда. Нарушение']")
    st.write("")
    
    # Display instruction text
    st.markdown("##### Инструкция:")
    st.write("1. Загрузите свое изображение, используя кнопку 'Upload an image'.")
    st.write("2. После загрузки изображения результаты обработки отобразятся ниже.")
    st.write("")
    
    # Add a button to upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Process the uploaded image if available
    if uploaded_image is not None:
        
        st.write("В результате будет детектированный снимок. Все найденные объекты выделены ограничивающей рамкой (баундинг боксом). И идентифицированы опеределенным классом - цветом рамки.")
        
        image = Image.open(uploaded_image)
        st.image(greet(image), caption='Processed Image', use_column_width=True)
        
        st.write("Отсутствие средств индивидуальной защиты или курение считается нарушением и дополнительно выделяется предупреждением 'НАРУШЕНИЕ' над прямоугльником.")

    st.write("")
    st.markdown("##### Рекомендации:")
    st.write("- Размер снимка равен или более 1280x1280.")
    st.write("- Снимок с камер УРАЛ ЗАВОДА: склада, снаружи склада, курилка день/ночь.")
    st.write("- Чем ближе объект - тем выше шанс детекции. Если объект далеко, можно масштабировать зону поиска.")

    st.write("")
    st.write("Результаты теста на независимых данных про соблюдении рекомендациий.")
    # Пути к изображениям
    image1_path = 'confusion_matrix_normalized.png'
    image2_path = 'PR_curve.png'
    # Делаем две колонки
    col1, col2 = st.columns(2)
    # Отображаем изображения в каждой колонке
    with col1:
        st.image(image1_path, caption='Матрица ошибок', use_column_width=True)
    with col2:
        st.image(image2_path, caption='Соотношение точности к полноте', use_column_width=True)

    st.write("")
    st.write("Но даже при соблюдении всех рекомендаций бывают исключения, когда детекция не срабатывает, а в плохих условиях работает. Моделей с 100% точностью не бывает.")
    st.write("Спасибо за внимание!")
    st.write("Автор: https://t.me/dtatarintsev")
    st.write("GitHub проекта: https://github.com/NeuronsUII/Zavod_Ural_n/tree/main/dtatarintsev")
    
if __name__ == '__main__':
    main()
