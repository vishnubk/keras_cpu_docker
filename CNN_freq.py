from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K
from keras import optimizers
import numpy as np
import math
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

#x, x_test, y, y_test = = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
img_width, img_height = 48, 48
input_shape = (48, 48, 1)
batch_size = 200
tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

def generator(batch_size,from_list_x,from_list_y):

    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True: 

        for i in range(0,total_size,batch_size):
            yield np.array(from_list_x[i:i+batch_size]), np.array(from_list_y[i:i+batch_size])

# Load all data
time_phase_pulsars = np.load('time_phase_data_pulsars.npy')
time_phase_nonpulsars = np.load('time_phase_data_nonpulsars.npy')

freq_phase_pulsars = np.load('freq_phase_data_pulsars.npy')
freq_phase_nonpulsars = np.load('freq_phase_data_nonpulsars.npy')

pulse_profile_pulsars = np.load('pulse_profile_data_pulsars.npy')
pulse_profile_nonpulsars = np.load('pulse_profile_data_nonpulsars.npy')

dm_curve_pulsars = np.load('dm_curve_data_pulsars.npy')
dm_curve_nonpulsars = np.load('dm_curve_data_nonpulsars.npy')




reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars] 
reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars] 

reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars] 
reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars] 


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# After these layers, we convert our 3D feature maps to 1D feature vectors ith the help of 'flatten'. We use 'dropout' layer to prevent overfitting


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#print(model.summary())
label_reshaped_time_phase_pulsars = np.ones(len(reshaped_time_phase_pulsars))
label_reshaped_time_phase_nonpulsars = np.zeros(len(reshaped_time_phase_nonpulsars))

time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
time_phase_label_combined = np.concatenate((label_reshaped_time_phase_pulsars, label_reshaped_time_phase_nonpulsars), axis = 0)
time_phase_train, time_phase_test, time_phase_label_train, time_phase_label_test = train_test_split(time_phase_data_combined, time_phase_label_combined, test_size=0.2, random_state=42)

model.fit_generator(generator(batch_size, time_phase_train, time_phase_label_train), steps_per_epoch=len(time_phase_train)//batch_size, epochs=100, callbacks=[tensor_board])
model.save_weights('first_try.h5')

#number_of_examples = len(time_phase_test)
#number_of_generator_calls = math.ceil(number_of_examples / (1.0 * batch_size))

predict = model.predict_generator(generator(batch_size, time_phase_test, time_phase_label_test), steps=math.ceil(len(time_phase_test)/batch_size))
np.save('predictions.npy', predict)
classified_results = np.rint(predict)
f_score = f1_score(time_phase_label_test, classified_results, average='binary')
precision = precision_score(time_phase_label_test, classified_results, average='binary')
recall = recall_score(time_phase_label_test, classified_results, average='binary')
print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall)

print('Start testing GBNCC data')

gbncc_pulsars = np.load('time_phase_gbncc_test_data_pulsars.npy')

gbncc_nonpulsars = np.load('time_phase_gbncc_test_data_nonpulsars_part3.npy')

reshaped_time_phase_gbncc_pulsars = [np.reshape(f,(48,48,1)) for f in gbncc_pulsars]
reshaped_time_phase_gbncc_nonpulsars = [np.reshape(f,(48,48,1)) for f in gbncc_nonpulsars]

label_reshaped_time_phase_gbncc_pulsars = np.ones(len(gbncc_pulsars), dtype=np.int8)
label_reshaped_time_phase_gbncc_nonpulsars = np.zeros(len(gbncc_nonpulsars), dtype=np.int8)
time_phase_gbncc_data_combined = np.concatenate((reshaped_time_phase_gbncc_pulsars, reshaped_time_phase_gbncc_nonpulsars), axis = 0)
time_phase_gbncc_label_combined = np.concatenate((label_reshaped_time_phase_gbncc_pulsars, label_reshaped_time_phase_gbncc_nonpulsars), axis = 0)

predict = model.predict_generator(generator(batch_size, time_phase_gbncc_data_combined, time_phase_gbncc_label_combined), steps=math.ceil(len(time_phase_gbncc_data_combined)/batch_size))
np.save('predictions_gbncc.npy', predict)
#test = np.rint(predict)
#test = np.reshape(test, (22709))
test = np.random.uniform(0,1,22709)
test = np.rint(test)
f_score = f1_score(time_phase_gbncc_label_combined, test, average='binary')

precision = precision_score(time_phase_gbncc_label_combined, test, average='binary')

recall = recall_score(time_phase_gbncc_label_combined, test, average='binary')

print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall)
accuracy = np.sum(test == time_phase_gbncc_label_combined)
print('Accuracy:', accuracy)
#generator(batch_size, time_phase_data_combined, time_phase_label_combined)

#train_datagen = ImageDataGenerator(rotation_range = 0)

#train_generator = train_datagen.flow_from_directory('train/', target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
#print(train_generator)

