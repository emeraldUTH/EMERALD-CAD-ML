# EMERALD - ML
[![GitHub stars](https://img.shields.io/github/stars/emeraldUTH/EMERALD-CAD-ML.svg?style=flat&label=Star)](https://github.com/emeraldUTH/EMERALD-CAD-ML/)
[![Readme](https://img.shields.io/badge/README-green.svg)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official <b>ML</b> implementation for CAD ML classification work showcased on 
["Classification models for assessing coronary artery disease instances using clinical and biometric data: an explainable man-in-the-loop approach"](https://www.nature.com/articles/s41598-023-33500-9) and 
["Uncovering the Black Box of Coronary Artery Disease Diagnosis: The Significance of Explainability in Predictive Models"](https://www.mdpi.com/2076-3417/13/14/8120).


## Papers' Abstract
These studies aim to develop and enhance computer-aided classification models for the prediction and diagnosis of coronary artery 
disease (CAD) by incorporating clinical data and expert opinions. Traditionally diagnosed through Invasive Coronary Angiography (ICA), 
CAD diagnosis has become a focal point for Medical Decision Support Systems (MDSS) utilizing machine learning (ML) and deep 
learning (DL) algorithms. These systems, however, often function as "black boxes," offering little transparency in how features 
are weighted and decisions are made. To address this, our research introduces a "man-in-the-loop" approach, integrating expert 
input into the ML prediction process to improve both accuracy and explainability.

Using a dataset of biometric and clinical data from 571 patients, including 43% ICA-confirmed CAD cases, multiple ML algorithms 
were applied with three different feature selection methods. The models were evaluated with and without expert diagnostic 
input using common metrics such as accuracy, sensitivity, and specificity. Stratified ten-fold cross-validation was performed, 
yielding a maximum accuracy of 83.02%, sensitivity of 90.32%, and specificity of 85.49% with expert input, 
compared to 78.29%, 76.61%, and 86.07% without it.

Additionally, we applied state-of-the-art explainability techniques to explore the feature-weighting process of a CatBoost algorithm, 
comparing the findings to established medical literature and common CAD risk factors. This dual focus on performance and transparency 
enhances trust and confidence in the diagnostic process, illustrating the importance of human expertise in the development of ML-driven 
CAD diagnosis tools. The study demonstrates the potential of combining machine learning with expert opinion to both improve CAD diagnosis 
accuracy and shed light on the "black-box" nature of prediction models.

## Usage

**EPU-CNN**, to the best of our knowledge, is the first framework based on **Generalized Additive Models** for the construction of **Inherently
Interpretable CNN models**, regardless of the base CNN architecture used and the application domain.
Unlike current approaches, the models constructed by EPU-CNN enables interpretable classification based both
on perceptual features and their spatial expression within an image; thus, it enables a more thorough and intuitive
interpretation of the classification results.

EPU-CNN is capable of providing both **qualitative and quantitative classification interpretations**. An example of 
image-specific interpretations provided by the EPU-CNN is shown below:

![Interpretation Example](assests/interpretation_example.png)

The quantitative interpretations on each image can be used to construct dataset level interpretations, which can be used
to identify the most important perceptual features for the classification task. An example of such an interpretation is
shown below:

![Dataset Interpretation Example](assests/dataset_interpretation_example.png)

To use EPU-CNN, you need to provide a **base CNN architecture** and a **perceptual feature extractor**. In this repository
we provide exemplary base CNN architectures and perceptual feature extractors used in our experimentations. An example of usage
is shown below:

```python
from models.epu import EPUNet
from models.subnetworks import Subnet

# Initialize the EPU-CNN model
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", 
             subnet=Subnet, num_classes=1, num_pfms=4, 
             fc_hidden_units=512)
```

The `subnet` defines the base CNN architecture. In our implementation `init_size` defines the number of 
features in the first convolutional layers that is incremented  by a factor of **2** in each subsequent convolutional 
block of the base CNN architecture. `hidden_units` defines the number of hidden units in the fully connected layers of
the base CNN architecture. `num_pfms` defines the number of perceptual feature maps used as input for a particular 
application.

The `subnet_act` defines the output activation function of the base CNN architecture and the `epu_act` defines the inverse
of the link function **g** used to provide the final output of the EPU-CNN model.

Currently, the `EPUNet` class has an implementation of `get_pfm` method that can be used to extract the PFMs of 
__green-red, blue-yellow, coarse-fine and light-dark__. The `get_pfm` takes as input a list of images (`np.ndarray`) and
outputs a tuple of PFMs. In its current form, `EPUNet` takes as input for training and inference a `tuple` of PFMs where
each position of the tuple is an `np.ndarray` of shape `(batch_size, height, width, 1)`.

### Training

An example of the training process is provided in the `train.py` script. The `train.py` script can be used to train an
EPU-CNN model given a set of training images and target labels. The `train.py` script can be used as follows:

```python
images_train, images_validation = ...
labels_train, labels_validation = ...
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
             subnet=Subnet, fc_hidden_units=512, classes=1)
epu.set_name("example-model")

optimizer = ...

epu.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
epu.fit(x=EPUNet.get_pfms(images_train, 128, 128), y=labels_train, epochs=1,
        validation_data=(EPUNet.get_pfms(images_validation, 128, 128), labels_validation),
        batch_size=1, callbacks=[])
```

It is recomended to save the trained model using either the `save_model` function or save the weights using the `np.save`
function. For example. 

```python
epu.save_model("example-model")
# or
np.save("example-model-weights", epu.get_weights())
```

### Interpretations

Currently `EPUNet` provides its interpretations using the `get_interpretation` and `get_prm` methods. The 
`get_interpretation` returns the qunatitative contributions of each **PFM** used whereas `get_prm` returns the
spatial expression of each **PFM** used on the image. To get the exact results with the paper the `get_prm` results need 
to be propagated to the `refine_prm` method of the `EPUNet` class.

To work, call after the loading of the model or the weights to the initialized object of epu don't use the `.predict` method
but call the model as `epu(...)` on a PFM tensor instead.

```python
images = ...
epu = EPUNet(init_size=32, subnet_act="tanh", epu_act="sigmoid", features_num=4,
            subnet=Subnet, fc_hidden_units=512, classes=1)
epu.load_model("example-model")
epu(EPUNet.get_pfm(images[0], 128, 128))

# Get Relevance Similarity Scores 
rss = epu.get_interpret_output()

# Get Perceptual Relevance Maps
prms = epu.refine_prm(epu.get_prm())
```

## Datasets

The datasets below have been used for training and evaluating the EPU-CNN models.

* [Banapple](https://github.com/innoisys/Banapple)
* [KID Dataset](https://mdss.uth.gr/datasets/endoscopy/kid/)
* [Kvasir](https://datasets.simula.no/kvasir/)
* [ISIC-2019](https://challenge2019.isic-archive.com/)
* [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [iBeans](https://github.com/AI-Lab-Makerere/ibean/)

The **Banapple**, **KID Dataset**, **Kvasir** and **ISIC-2019** have been downloaded from their respective sources and 
have been curated manually for the training and evaluation of EPU-CNN models. The **CIFAR-10**, **MNIST**, 
**Fashion-MNIST** and **iBeans** datasets have been used via the [Tensorflow Datasets API](https://www.tensorflow.org/datasets). 

## Citation
If you find this work useful, please cite our papers:

```
@article{Samaras2023classification,
  title = {Classification models for assessing coronary artery disease instances using clinical and biometric data: an explainable man-in-the-loop approach},
  author = {Samaras, Agorastos-Dimitrios and Moustakidis, Serafeim and Apostolopoulos, Ioannis D. and Papandrianos, Nikolaos and Papageorgiou, Elpiniki},
  journal = {Scientific Reports},
  volume = {13},
  number = {1},
  pages = {6668},
  year = {2023},
  publisher = {Nature Publishing Group UK London},
  doi = {10.1038/s41598-023-33500-9},
  url = {https://doi.org/10.1038/s41598-023-33500-9}
}
```

```
@article{Samaras2023uncovering,
  title = {Uncovering the Black Box of Coronary Artery Disease Diagnosis: The Significance of Explainability in Predictive Models},
  author = {Samaras, Agorastos-Dimitrios and Moustakidis, Serafeim and Apostolopoulos, Ioannis D. and Papageorgiou, Elpiniki and Papandrianos, Nikolaos},
  journal = {Applied Sciences},
  volume = {13},
  number = {14},
  year = {2023},
  doi = {10.3390/app13148120},
  url = {https://doi.org/10.3390/app13148120},
  issn = {2076-3417}
}
```
