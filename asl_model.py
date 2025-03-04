import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Define paths
train_data_dir = r"C:\Users\Aananda Sagar Thapa\OneDrive\Desktop\ASL_Alphabet_Dataset\asl_alphabet_train" # Replace with the path to your dataset folder
img_size = (128, 128)  # Increased image size for better accuracy
batch_size = 32
num_classes = 29  # 26 letters (A-Z) + 1 space

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Load MobileNetV2 as the base model
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # Match the image size
    include_top=False,  # Exclude the final classification layer
    weights="imagenet",
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(num_classes, activation="softmax"),  # 27 classes for A-Z + space
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # Increase the number of epochs
    validation_data=validation_generator,
)

# Save the model
model.save("asl_transfer_mobilenetv2.h5")

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")