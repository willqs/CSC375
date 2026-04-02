### Basic FCN for semantic segmentation

import tensorflow as tf
from tensorflow.keras import layers, models

def basic_fcn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # --- ENCODER (Downsampling) ---
    # Standard CNN behavior: shrink the image, grow the features
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x) # 112 x 112
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x) # 56 x 56
    
    # --- THE BOTTLENECK ---
    # This represents the "compressed" semantic meaning of the image
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    # --- DECODER (Upsampling) ---
    # We use Conv2DTranspose to "stretch" the image back to size
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x) # 112 x 112
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x) # 224 x 224

    # --- HEAD ---
    # Final 1x1 conv to map features to the number of classes
    # If binary segmentation (e.g., Background vs Leaf), num_classes=1, activation='sigmoid'
    # If multi-class, activation='softmax'
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)

    return models.Model(inputs, outputs)

# Example usage:
# model = basic_fcn((224, 224, 3), num_classes=3)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

