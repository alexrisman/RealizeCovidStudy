import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import os
from collections import defaultdict
import numpy as np


def get_covidnet():
    sess = tf.Session()
    tf.get_default_graph()
    weightspath = 'COVID-Net/models/COVIDNet-CXR4-A'
    metaname = 'model.meta'
    ckptname = 'model-18540'
    in_tensorname = 'input_1:0'
    out_tensorname = 'norm_dense_1/Softmax:0'
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name(in_tensorname)
    pred_tensor = graph.get_tensor_by_name(out_tensorname)
    return sess, image_tensor, pred_tensor


def get_covidnet_s(geo_ind=True):
    sess = tf.Session()
    tf.get_default_graph()
    if geo_ind:
        weightspath = 'COVID-Net/models/COVIDNet-S-GEO'
    else:
        weightspath = 'COVID-Net/models/COVIDNet-S-OPC'
    metaname = 'model.meta'
    ckptname = 'model'
    model = MetaModel(os.path.join(weightspath, metaname),
                              os.path.join(weightspath, ckptname))
    return model


def get_chexnet():
    # need to do this to avoid ValueError: Input size must be at least 221x221; got `input_shape=(224, 224, 3)`
    # note: keras and tf.keras don't mix, https://github.com/keras-team/keras/issues/10907#issuecomment-413987821
    K.clear_session()
    K.set_image_data_format('channels_last')
    base_model_name = "DenseNet121"
    class_names = "Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia".split(",")
    model_weights_path = 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)
    pneumonia_index = class_names.index('Pneumonia')
    return model, pneumonia_index


class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]

    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None, input_shape=None):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name)

        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model


def score_prediction(softmax, step_size):
    vals = np.arange(3) * step_size + (step_size / 2.)
    vals = np.expand_dims(vals, axis=0)
    return np.sum(softmax * vals, axis=-1)


class MetaModel:
    def __init__(self, meta_file, ckpt_file):
        self.meta_file = meta_file
        self.ckpt_file = ckpt_file

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.meta_file)
            self.input_tr = self.graph.get_tensor_by_name('input_1:0')
            self.phase_tr = self.graph.get_tensor_by_name('keras_learning_phase:0')
            self.output_tr = self.graph.get_tensor_by_name('MLP/dense_1/MatMul:0')

    def infer(self, image):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, self.ckpt_file)

            outputs = defaultdict(list)
            outs = sess.run(self.output_tr,
                            feed_dict={
                                self.input_tr: np.expand_dims(image, axis=0),
                                self.phase_tr: False
                            })
            outputs['logits'].append(outs)

            for k in outputs.keys():
                outputs[k] = np.concatenate(outputs[k], axis=0)

            outputs['softmax'] = np.exp(outputs['logits']) / np.sum(
                np.exp(outputs['logits']), axis=-1, keepdims=True)
            outputs['score'] = score_prediction(outputs['softmax'], 1 / 3.)

        return outputs['score']
