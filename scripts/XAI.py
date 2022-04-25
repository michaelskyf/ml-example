import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

from dataset import TEST_DATASET_PATH
from models import MODELS_PATH

model = keras.models.load_model(os.path.join(MODELS_PATH, "inception3.h5"))

datagen = ImageDataGenerator(rescale=1.0 / 255)

image_generator = datagen.flow_from_directory(TEST_DATASET_PATH,
                                              batch_size=1,
                                              target_size=(224, 224))

for image in image_generator:
    image = image[0].reshape(224, 224, 3)

    plt.imshow(image)
    plt.show()

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(image.astype('double'),
                                             model.predict,
                                             top_labels=2,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=True,
    )
    plt.imshow(mark_boundaries(temp, mask))
    plt.show()

    print(model.predict(image.reshape(1, 224, 224, 3)))

    temp, mask = explanation.get_image_and_mask(
        label=0,
        positive_only=False,
        num_features=10,
        hide_rest=False,
    )
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    temp, mask = explanation.get_image_and_mask(
        label=0,
        positive_only=False,
        num_features=10,
        hide_rest=False,
        min_weight=0.2,
    )
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()
    dict_heatmap = dict(explanation.local_exp[0])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    plt.imshow(
        heatmap,
        cmap='RdBu',
        vmin=-heatmap.max(),
        vmax=heatmap.max(),
    )
    plt.colorbar()
    plt.show()
