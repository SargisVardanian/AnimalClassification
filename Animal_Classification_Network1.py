import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# Путь к папке с данными обучения
train_path = 'train'

# Загрузка изображений и меток классов из папки обучения
images = []
labels = []

for class_name in os.listdir(train_path):
    class_path = os.path.join(train_path, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            print('image', image.shape, class_name)
            images.append(image)
            labels.append(class_name)

# Преобразование изображений и меток в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Преобразование меток в числовые значения
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Разделение данных на обучающий и проверочный наборы
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Преобразование меток в формат one-hot
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

# Создание модели
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
