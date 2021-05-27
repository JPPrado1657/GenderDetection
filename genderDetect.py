import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image

fold0 = pd.read_csv("fold_0_data.txt", sep="\t")
fold1 = pd.read_csv("fold_1_data.txt", sep="\t")
fold2 = pd.read_csv("fold_2_data.txt", sep="\t")
fold3 = pd.read_csv("fold_3_data.txt", sep="\t")
fold4 = pd.read_csv("fold_4_data.txt", sep="\t")

total_data = pd.concat([fold0, fold1, fold2, fold3, fold4], ignore_index=True)
print(total_data.shape)
total_data.info()

total_data.head()

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, LayerNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

imp_data = total_data[['age', 'gender', 'x', 'y', 'dx', 'dy']].copy()

img_path = []
for row in total_data.iterrows():
  path = f'./faces/{row[1].user_id}/coarse_tilt_aligned_face.{row[1].face_id}.{row[1].original_image}'
  img_path.append(path)

imp_data['img_path'] = img_path
imp_data.head()

age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'),
               ('(8, 12)', '8-13'), ('13', '8-13'), ('22', '15-20'),
               ('(8, 23)','15-20'), ('23', '25-32'), ('(15, 20)', '15-20'),
               ('(25, 32)', '25-32'), ('(27, 32)', '25-32'), ('32', '25-32'),
               ('34', '25-32'), ('29', '25-32'), ('(38, 42)', '38-43'),
               ('35', '38-43'), ('36', '38-43'), ('42', '48-53'),
               ('45', '38-43'), ('(38, 43)', '38-43'), ('(38, 42)', '38-43'),
               ('(38, 48)', '48-53'), ('46', '48-53'), ('(48, 53)', '48-53'),
               ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'),
               ('57', '60+'), ('58', '60+')
              ]
              
age_mapping_dict = {each[0]: each[1] for each in age_mapping}
drop_labels = []
for idx, each in enumerate(imp_data.age):
    if each == 'None':
        drop_labels.append(idx)
    else:
        imp_data.age.loc[idx] = age_mapping_dict[each]

imp_data = imp_data.drop(labels=drop_labels, axis=0)
imp_data.age.value_counts(dropna=False)

imp_data = imp_data.dropna()
clean_data = imp_data[imp_data.gender != 'u'].copy()
clean_data.info()

gender_to_label_map = {
    'f': 0,
    'm': 1
}

clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])
clean_data.head()

age_to_label_map = {
    '0-2'  :0,
    '4-6'  :1,
    '8-13' :2,
    '15-20':3,
    '25-32':4,
    '38-43':5,
    '48-53':6,
    '60+'  :7
}
clean_data['age'] = clean_data['age'].apply(lambda age: age_to_label_map[age])
clean_data.head()

X = clean_data[['img_path']]
y = clean_data[['gender']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

print('Train data shape {}'.format(X_train.shape))
print('Test data shape {}'.format(X_test.shape))


def resize_image_array(data_set, image_size):
  result_images = []
  for data_item in data_set.iterrows():
    image = Image.open(data_item[1].img_path)
    image = image.resize((image_size,image_size))
    data = np.asarray(image)
    result_images.append(data)
  
  return np.asarray(result_images)

train_images = resize_image_array(X_train, 277)
test_images = resize_image_array(X_test, 277)

print('Train images shape {}'.format(train_images.shape))
print('Test images shape {}'.format(test_images.shape))

model = Sequential()

model.add(Conv2D(
    input_shape=(277,277,3),
    filters=96,
    kernel_size=(7,7),
    strides=4,
    padding='valid',
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2)
))
model.add(LayerNormalization())
model.add(Conv2D(
    filters=256,
    kernel_size=(3,3),
    strides=1,
    padding='same',
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2)
))
model.add(LayerNormalization())
model.add(Flatten())
model.add(Dense(
  units=512,
  activation='relu'
))
model.add(Dropout(rate=0.25))
model.add(Dense(
  units=512,
  activation='relu'
))
model.add(Dropout(rate=0.25))
model.add(Dense(
  units=2,
  activation='softmax'
))
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, y_train, batch_size=32, epochs=25, validation_data=(test_images, y_test), callbacks=[callback])
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
model.save('gender_model25.h5')

test_loss, test_acc = model.evaluate(test_images, y_test, verbose=2)
print(test_acc)