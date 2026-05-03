import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📁 مكان الداتا
train_dir = "dataset/train"

# 🔄 تجهيز الصور
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 الموديل
model = models.Sequential([
    layers.Input(shape=(48,48,3)),   # 🔥 مهم جدًا

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 التدريب
model.fit(train_data, validation_data=val_data, epochs=10)

# 💾 حفظ الموديل
model.save("emotion_model.h5")

print("✅ DONE TRAINING")