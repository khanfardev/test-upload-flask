"""
Imports Library : => Tensorflow , Keras
"""
import tensorflow as tensor
from tensorflow.keras import layers
#Tensorflow , Kears
print("Testing")


def get_dataset(path):
    return tensor.keras.preprocessing.image_dataset_from_directory(directory=path,
                                                                   seed=42,
                                                                   image_size=(250, 250),
                                                                   batch_size=100)


train_data = get_dataset("C:\\Users\\96279\\OneDrive\\Desktop\\Plant System\\Datasets\\Plants(Dataset)\\train")
test_data = get_dataset("C:\\Users\\96279\\OneDrive\\Desktop\\Plant System\\Datasets\\Plants(Dataset)\\test")
["APPLE_He"]
class_names = train_data.class_names
print(train_data)
# ==================================================================================#
model = tf.keras.models.Sequential([
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
retVal = model.fit(train_data,validation_data=test_data,epochs = 5)

print("Accuracy = "+str(retVal.history['accuracy'][4]))


model.save('model.h5') #save model