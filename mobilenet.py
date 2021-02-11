#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020. Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo           #
# Pellegrini, Davide Maltoni. All rights reserved.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2020                                                             #
# Authors: Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo Pellegrini, Davide   #
# Maltoni.                                                                     #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" This file contains the model Class used for the exps """

import torch
import torch.nn as nn

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except:
    from pytorchcv.models.common import DwsConvBlock
from pytorchcv.model_provider import get_model

import torchvision


def remove_sequential(network, all_layers):

    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


# ALSO REPLACE RELU with RELU6
def remove_DwsConvBlock(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
           #  print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers

def remove_DwsConvBlock_relu6(cur_layers):

    all_layers = []
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
            for ch in layer.children():
                all_layers.append(ch)
                for attr_str in dir(ch):
                    target_attr = getattr(ch, attr_str)
                    if type(target_attr) == torch.nn.ReLU:
                        print("WITH RELU 6 !!!")
                        new_relu6 = torch.nn.ReLU6()
                        setattr(ch, attr_str, new_relu6)
        else:
            all_layers.append(layer)

    print(all_layers)
    return all_layers

class MyMobilenetV1(nn.Module):

    def __init__(self, pretrained=True, latent_layer_num=20):

        super().__init__()

        #model = models.resnet101(pretrained=False)
        #model_ft.load_state_dict(torch.load(PATH))

        #model = MyMobilenetV1()
        model = get_model("mobilenet_w1", pretrained=pretrained)
        # model.load_state_dict(torch.load("/home/lravaglia/work/ar1-pytorch-master/models/mobilenet_w1-0895-7e1d739f.pth"))

        #pretrained_state = torch.load("/home/lravaglia/work/ar1-pytorch-master/models/mobilenet_w1-0895-7e1d739f.pth")
        #model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        model.load_state_dict(pretrained_state)

        #model = get_model("mobilenet_w1", pretrained=pretrained)
        model.features.final_pool = nn.AvgPool2d(4)

        #print(model)

        all_layers = []
        remove_sequential(model, all_layers)
        #replace_relu(model, all_layers)
        #print("ALL LAYERS AFTER REMOVE SEQ")
        #print(all_layers)
        all_layers = remove_DwsConvBlock(all_layers)
        #print("ALL LAYERS AFTER REMOVE DW")
        #print(all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                #print("LAYER ",i)
                lat_list.append(layer)
            else:
                #print("LAYER ",i)
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        self.output = nn.Linear(1024, 1000, bias=False)


    def forward(self, x, latent_input=None, return_lat_acts=False, lat_acts_requires_grad=False):

        orig_acts = self.lat_features(x)
        if latent_input is not None:
            lat_acts = torch.cat((orig_acts, latent_input), 0)
        else:
            lat_acts = orig_acts

        if lat_acts_requires_grad:
            lat_acts.requires_grad = True

        x = self.end_features(lat_acts)
        x = x.view(x.size(0), -1)
        logits = self.output(x)

        if return_lat_acts:
            return logits, orig_acts
        else:
            return logits


if __name__ == "__main__":

    model = MyMobilenetV1(pretrained=False)
    for name, param in model.named_parameters():
        print(name)
