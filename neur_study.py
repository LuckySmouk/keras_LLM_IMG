import os
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Путь сохраненной модельки
model_path = 'path to a trained model'

# Размеры изображения
img_width, img_height = 128, 128

# Путь к данным для обучения
train_data_dir = 'learning data path'
nb_train_samples = len(os.listdir(train_data_dir))

# Путь к данным для валидации
validation_data_dir = 'validation data path'
nb_validation_samples = len(os.listdir(validation_data_dir))

# Количество эпох
epochs = 30
batch_size = 32

# Проверяем, существует ли уже модель
if os.path.exists(model_path):
    # Если модель существует, загружаем ее
    model = load_model(model_path)
else:
    # Если модели нет, создаем новую
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), input_shape=(
    img_width, img_height, 3), name='conv2d_0'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), input_shape=(
    img_width, img_height, 3), name='conv2d_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), name='conv2d_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), name='conv2d_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), name='conv2d_4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(2))  # assuming 2 classes (car part and non-car part)
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',      # 'rmsprop'  !'adam'!  'adagrad'  'adadelta'  'adamax'  'sgd'  'Nadam'  'ftrl'
                  metrics=['accuracy'])

# Аугментация данных
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Обучение модели
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(nb_train_samples / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=math.ceil(nb_validation_samples / batch_size))

# Сохранение модели
model.save(model_path)


print("Model saved")
