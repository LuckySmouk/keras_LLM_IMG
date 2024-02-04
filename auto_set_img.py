from PIL import Image
import os


def resize_and_rename_images(input_folder, output_folder, target_size):
    # Создаем выходную папку, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получаем список файлов во входной папке
    input_files = os.listdir(input_folder)

    for idx, file_name in enumerate(input_files):
        # Полный путь к текущему файлу
        input_path = os.path.join(input_folder, file_name)

        # Игнорируем не изображения
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            continue

        # Открываем изображение с помощью Pillow
        img = Image.open(input_path)

        # Приводим изображение к целевому размеру
        resized_img = img.resize(target_size, Image.LANCZOS)

        # Новое имя файла с префиксом в виде номера
        new_file_name = f"{idx + 1:04d}.jpg"

        # Создаем путь для сохранения отредактированного и переименованного изображения
        output_path = os.path.join(output_folder, new_file_name)

        # Сохраняем отредактированное и переименованное изображение
        resized_img.save(output_path)

    print("Изображения успешно приведены к заданному размеру и переименованы в", output_folder)


# Пример использования:
input_folder_path = "C:/Py/data/ll_keras/Result/B"
output_folder_path = "C:/Py/data/ll_keras/Result/B/es"
target_size = (128, 128)  # Задайте необходимый размер

resize_and_rename_images(input_folder_path, output_folder_path, target_size)
