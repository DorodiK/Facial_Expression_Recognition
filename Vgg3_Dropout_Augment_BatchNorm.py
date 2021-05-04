#%%import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#%%load data 

train_data_path = "train/"
val_data_path = "test/"

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
test_datagen = ImageDataGenerator(rescale=1/255.0,
    )

train_generator = train_datagen.flow_from_directory(
        train_data_path,
        batch_size=16,
        class_mode='categorical',
        target_size = (48,48),
        color_mode = 'grayscale'     
        )
validation_generator = test_datagen.flow_from_directory(
        val_data_path,
        batch_size=16,
        class_mode='categorical',
        target_size = (48,48),
        color_mode = 'grayscale'
        )


class_indices = train_generator.class_indices

#%% testing
for i in train_generator:
    x = i
    break






#%%baseline model VGG-3


#%%BatchNormalization + Increasing Dropout
#VGG-3 + BatchNormalization + Increasing Dropout 
inputs = tf.keras.Input(shape=(48, 48,1))
x = tf.keras.layers.Conv2D(32, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(32, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(64, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(128, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(128, (3, 3), 1, padding = 'same', activation='relu', kernel_initializer='he_uniform')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.summary()


#%% compile model
optimizer = keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["acc","Recall"])

#%% train the model
epochs = 50
model_history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, shuffle=True)


#%%loss graph
#%matplotlib auto

def summarize_diagnostics(history):
    plt.figure(dpi = 300)
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.grid(True) 
    plt.legend()
# plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    plt.grid(True) 
    plt.legend()

summarize_diagnostics(model_history)

#%% test data

ypred = model.predict(validation_generator)


#%% check metrics
model_metrics = model.evaluate(validation_generator)

