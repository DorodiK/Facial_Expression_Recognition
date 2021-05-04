#%%import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage import transform

#%% data generator
train_data_path = "train/"
val_data_path = "test/"

#use this for resnet

# tf.keras.applications.resnet.preprocess_input


#use this for vgg16

# tf.keras.applications.vgg16.preprocess_input

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #horizontal_flip=True
    preprocessing_function = tf.keras.applications.vgg16.preprocess_input
    )
test_datagen = ImageDataGenerator(rescale=1/255.0,
               preprocessing_function = tf.keras.applications.vgg16.preprocess_input
                                  )

train_generator = train_datagen.flow_from_directory(
        train_data_path,
        batch_size=16,
        class_mode='categorical',
        target_size = (48,48)
        )
validation_generator = test_datagen.flow_from_directory(
        val_data_path,
        batch_size=16,
        class_mode='categorical',
        target_size = (48,48)
        )



class_indices = train_generator.class_indices

#%% testing
for i in train_generator:
    x = i
    break
#%% Vgg16 model

pretrained_model = tf.keras.applications.VGG16(
                                include_top=False,
                                weights="imagenet",
                                #input_tensor=(None),
                                input_shape=(48,48,3),
                                pooling=None,
                                classes=7
                                )


#%% resnet model
'''
pretrained_model  = tf.keras.applications.ResNet50(
                                include_top=False,
                                weights="imagenet",
                                input_tensor=(None),
                                input_shape=(224,224,3),
                                pooling=True,
                                classes=2
                                )


pretrained_model .summary()
'''
#%% freeze weights
for layer in pretrained_model.layers[:-3]:
    layer.trainable = False
#%% transfer learning vgg16

pretrained_model_input = pretrained_model.input
pretrained_model_output = pretrained_model.output
#x = keras.layers.GlobalAveragePooling2D()(pretrained_model_output) #for resnet
x = keras.layers.Flatten()(pretrained_model_output) #for vgg16
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
model_output = keras.layers.Dense(7, activation="softmax")(x)
model = keras.models.Model(inputs=[pretrained_model_input], outputs=[model_output])
 
model.summary()

#%% compile vgg16
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["acc","Recall"])

#%% callbacklist

# '''
# import time
# class TimeHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.times = []

#     def on_epoch_begin(self, batch, logs={}):
#         self.epoch_time_start = time.time()

#     def on_epoch_end(self, batch, logs={}):
#         self.times.append(time.time() - self.epoch_time_start)
        
# time_callback = TimeHistory()    
        
# checkpoint_path = r'C:/Users/Ero/Desktop/DorodiCode/weights/'
# # Keras callbacks for training
# callback_list = [
#     tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_path + \
#                 "ModelWeights.e{epoch:02d}-" + \
#                 "Loss{val_loss:.4f}.h5",
#         monitor='val_loss',
#         save_best_only=True,
#         save_weights_only=True),
        
#         tf.keras.callbacks.TensorBoard(
#             update_freq='batch',
#             profile_batch=0),
#             ]
    
# '''

#%% train the model
epochs = 50
model_history = model.fit(train_generator, epochs=epochs, 
                                                      validation_data=validation_generator,
                                                      # callbacks = callback_list,
                                                      shuffle=True)

#%% graph

#%% save model
model.save('saved_model.h5')

#%%loss graph
# %matplotlib auto

def summarize_diagnostics(history):
    plt.figure(dpi = 300)
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.grid(True) 
# plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['acc'], color='blue', label='train')
    plt.plot(history.history['val_acc'], color='orange', label='test')
    plt.grid(True) 

summarize_diagnostics(model_history)

#%% test data

ypred = model.predict(validation_generator)


#%% check metrics
model_metrics = model.evaluate(validation_generator)