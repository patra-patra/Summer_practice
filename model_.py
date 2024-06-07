import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt


folder = "C:\\Users\\user\\Desktop\\split_dataset"
dataset_dir = pathlib.Path(folder)

batch_size = 32
img_width = 224
img_height = 224

train = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'train',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

valid = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'val',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

test = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'test',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

class_names_len = len(train.class_names)
class_names = train.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.prefetch(buffer_size=AUTOTUNE)
valid = valid.prefetch(buffer_size=AUTOTUNE)
test = test.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train = train.map(lambda x, y: (data_augmentation(x, training=True), y))

base_model = tf.keras.applications.VGG16(input_shape=(img_height, img_width, 3),
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(class_names_len, activation='softmax')
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

epochs = 30

history = model.fit(
    train,
    validation_data=valid,
    epochs=epochs,
    callbacks=[early_stopping]
)

model.save("animal_model__.h5")

scores = model.evaluate(test, verbose=1)
print("Точность тестирования: ", round(scores[1] * 100, 4))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
