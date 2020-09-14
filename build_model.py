from tensorflow import keras

def finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = keras.layers.Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = keras.layers.Dense(fc, activation='relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    # New softmax layer
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model

def simple_binary_model(base_model):
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def simple_mult_model(base_model):
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation='softmax')
    ])
    return model