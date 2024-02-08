from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Путь к сохраненной модели
model_path = 'path to a trained model'

# Загрузка модели
loaded_model = load_model(model_path)

# Функция для предсказания класса изображения

img_width, img_height = 128, 128
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Нормализация значений пикселей

    prediction = loaded_model.predict(img_array)
    print("Вероятность классов: ",prediction)
    predicted_class = 1 if prediction[0][1] > 0.5 else 0 
    return predicted_class



# Пример использования
image_path_to_check = 'test img path'
predicted_class = predict_image_class(image_path_to_check)
