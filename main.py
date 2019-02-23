# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:18:29 2019

@author: wmy
"""

from keras.preprocessing.image import load_img, img_to_array
import keras.applications.vgg19
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import math
import time

class NeuralStyleTransferVGG19(object):
    
    def __init__(self, content_image_path, style_image_path, nH=320, nW=480):
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.nH = nH
        self.nW = nW
        self.nC = 3
        # 内容图像
        self.C = K.constant(self.preprocess_image(content_image_path))
        # 风格图像
        self.S = K.constant(self.preprocess_image(style_image_path))
        # 生成图像
        self.G = K.placeholder((1, nH, nW, 3))
        # 创建模型
        self.creat_model()
        self.content_layer = 'block5_conv2'
        self.style_layers = ['block1_conv1', 
                             'block2_conv1', 
                             'block3_conv1', 
                             'block4_conv1', 
                             'block5_conv1']
        self.compute_loss()
        self.compute_grads()
        pass
    
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.nH, self.nW))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = keras.applications.vgg19.preprocess_input(img)
        return img
    
    def deprocess_image(self, img):
        # 加上ImageNet训练集的图像平均值
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        # BGR转RGB
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8')
        return img
    
    def creat_model(self):
        input_tensor = K.concatenate([self.C, self.S, self.G], axis=0)
        self.model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        return self.model
    
    def compute_content_loss(self, C_features, G_features):
        loss = K.sum(K.square(G_features-C_features))
        return loss

    def compute_style_loss(self, S_features, G_features):
        def gram_matrix(X):
            features = K.batch_flatten(K.permute_dimensions(X, (2, 0, 1)))
            gram = K.dot(features, K.transpose(features))
            return gram
        S_gram_matrix = gram_matrix(S_features)
        G_gram_matrix = gram_matrix(G_features)
        size = self.nH * self.nW
        channals = self.nC
        loss = K.sum(K.square(S_gram_matrix-G_gram_matrix)) / (4.0 * (channals**2) * (size**2))
        return loss

    def compute_total_variation_loss(self):
        a = K.square(self.G[:, :self.nH-1, :self.nW-1, :] - self.G[:, 1:, :self.nW-1, :])
        b = K.square(self.G[:, :self.nH-1, :self.nW-1, :] - self.G[:, :self.nH-1, 1:, :])
        loss = K.sum(K.pow(a + b, 1.25))
        return loss
    
    def compute_loss(self, content_weight = 0.025, style_weight = 1.0, total_variation_weight = 1e-4):
        layers_outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])
        self.loss = K.variable(0.0)
        layer_features = layers_outputs_dict[self.content_layer]
        C_features = layer_features[0, :, :, :]
        G_features = layer_features[2, :, :, :]
        self.loss += content_weight * self.compute_content_loss(C_features, G_features)
        for layer_name in self.style_layers:
            layer_features = layers_outputs_dict[layer_name]
            S_features = layer_features[1, :, :, :]
            G_features = layer_features[2, :, :, :]
            self.loss += (style_weight/len(self.style_layers) * self.compute_style_loss(S_features, G_features))
            pass
        self.loss += total_variation_weight * self.compute_total_variation_loss()
        self.loss_function = K.function([self.G], [self.loss])
        return self.loss
    
    def get_loss(self, X):
        X = X.reshape((1, self.nH, self.nW, 3))
        return self.loss_function([X])[0]
    
    def compute_grads(self):
        self.grads = K.gradients(self.loss, self.G)[0]
        self.grads_function = K.function([self.G], [self.grads])
        return self.grads
    
    def get_grads(self, X):
        X = X.reshape((1, self.nH, self.nW, 3))
        return self.grads_function([X])[0]
    
    def generate(self, iterations=20):
        X = self.preprocess_image(self.content_image_path)
        X = X.flatten()
        def get_grads_flatten(X):
            grads = self.get_grads(X)
            return grads.flatten().astype('float64')
        for i in range(iterations):
            X, min_val, info = fmin_l_bfgs_b(self.get_loss, X, fprime=get_grads_flatten, maxfun=20)
            img = X.copy().reshape((self.nH, self.nW, 3))
            img = self.deprocess_image(img)
            print('After iteration ' + str(i+1) + ': ' + str(min_val) + '.')
            imsave(r'./outputs/G_' + str(i+1) + '.jpg', img)
            pass
        pass
    
    pass
    
nst = NeuralStyleTransferVGG19('images/C_10.jpg', 'images/S_10.jpg')
nst.generate()