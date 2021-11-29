from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# img_path = './datasets1/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-1282-0001.jpeg'
img_path = './datasets1/chest_xray/chest_xray/train/PNEUMONIA/person18_bacteria_57.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


model = VGG16(weights='imagenet')
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print('np.argmax(preds):', np.argmax(preds))

PNEUMONIA_output = model.output[:, np.argmax(preds)]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(PNEUMONIA_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.show(heatmap)


import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('chest_xray_textbook.jpg', superimposed_img)

