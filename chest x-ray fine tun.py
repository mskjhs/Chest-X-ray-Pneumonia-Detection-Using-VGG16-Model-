from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# set dir
train_dir='./datasets1/chest_xray/chest_xray/train/'
test_dir='./datasets1/chest_xray/chest_xray/test/'
validation_dir='./datasets1/chest_xray/chest_xray/val/'

# set image generators
train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=20, shear_range=0.1,
                    width_shift_range=0.1, height_shift_range=0.1,
                    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        # target_size=(512, 512),
        batch_size=20,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        # target_size=(512, 512),
        batch_size=20,
        class_mode='binary', shuffle=False)
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        # target_size=(512, 512),
        batch_size=20,
        class_mode='binary')

# load model
model = load_model('chest_xray[None, None].h5')
conv_base = model.layers[0]
for layer in conv_base.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True

# compile
model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])

# main loop without cross-validation
import time
starttime = time.time()
num_epochs = 100
history = model.fit_generator(train_generator,
                    epochs=num_epochs, steps_per_epoch=100,
                    validation_data=validation_generator, validation_steps=50)

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print('train_loss:', train_loss)
print('test_loss:', test_loss)
print("elapsed time (in sec): ", time.time()-starttime)

# problem 5
y_test = test_generator.classes
y_pred = model.predict_generator(test_generator)
matrix = sklearn.metrics.confusion_matrix(y_test, y_pred > 0.5)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
print('matrix')
print(matrix)
print('auc', auc)
print('')
print(sklearn.metrics.classification_report(y_test, y_pred > 0.5))

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history ['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history ['loss'])
    plt.plot(h.history ['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('[None,None]afterFTwithdrop.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('[None,None]afterFTwithdrop.accuracy.png')