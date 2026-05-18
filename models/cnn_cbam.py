import tensorflow as tf
from tensorflow.keras import layers, models

def channel_attention_module(x, ratio=8):
    """CBAM Channel Attention Module."""
    channel = x.shape[-1]
    
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    
    return layers.Multiply()([x, cbam_feature])

def spatial_attention_module(x):
    """CBAM Spatial Attention Module."""
    avg_pool = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(x)
    max_pool = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(x)
    
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    
    return layers.Multiply()([x, cbam_feature])

def cbam_block(x, ratio=8):
    """Complete CBAM Block."""
    x = channel_attention_module(x, ratio)
    x = spatial_attention_module(x)
    return x

def build_base_cnn(input_shape=(128, 128, 3)):
    """Builds a lightweight Base CNN for <2GB VRAM."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name="Base_CNN")
    return model

def build_cnn_cbam(input_shape=(128, 128, 3)):
    """Builds a lightweight CNN with CBAM for <2GB VRAM."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name="CNN_CBAM")
    return model
