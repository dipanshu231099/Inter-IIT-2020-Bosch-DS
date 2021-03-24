# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:54:23 2021

@author: prajw
"""

import keras
import tensorflow as tf
import numpy as np
import os

import innvestigate
import innvestigate.utils as iutils
import matplotlib.pyplot as plt

import imp
import innvestigate
import innvestigate.utils as iutils

base_dir = os.getcwd()

def investigate(path):
    model=tf.keras.models.load_model(path)

    eutils = imp.load_source("utils", base_dir+"/classification/utils_ma.py")
    utils = imp.load_source("utils", base_dir+"/classification/utils.py")

    data_not_preprocessed = eutils.fetch_data()

    input_range = [-1, 1]
    preprocess, revert_preprocessing = eutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

    data = (
        preprocess(data_not_preprocessed[0]), data_not_preprocessed[1],
        preprocess(data_not_preprocessed[2]), data_not_preprocessed[3]
    )

    num_classes = 43
    label_to_class_name = [str(i) for i in range(num_classes)]

    # Scale to [0, 1] range for plotting.
    def input_postprocessing(X):
        return revert_preprocessing(X) / 255

    noise_scale = (input_range[1]-input_range[0]) * 0.1
    ri = input_range[0]  # reference input


    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

        # Show input
        ("input",                 {},                       input_postprocessing,      "Input"),

        # Function
        ("gradient",              {"postprocess": "abs"},   eutils.graymap,        "Gradient"),
        ("smoothgrad",            {"noise_scale": noise_scale,
                                   "postprocess": "square"},eutils.graymap,        "SmoothGrad"),

        # Signal
        ("deconvnet",             {},                       eutils.bk_proj,        "Deconvnet"),
        ("guided_backprop",       {},                       eutils.bk_proj,        "Guided Backprop",),
        ("pattern.net",           {"pattern_type": "relu"}, eutils.bk_proj,        "PatternNet"),

        # Interaction
        ("pattern.attribution",   {"pattern_type": "relu"}, eutils.heatmap,        "PatternAttribution"),
        ("deep_taylor.bounded",   {"low": input_range[0],
                                   "high": input_range[1]}, eutils.heatmap,        "DeepTaylor"),
        ("input_t_gradient",      {},                       eutils.heatmap,        "Input * Gradient"),
        ("integrated_gradients",  {"reference_inputs": ri}, eutils.heatmap,        "Integrated Gradients"),
        ("deep_lift.wrapper",     {"reference_inputs": ri}, eutils.heatmap,        "DeepLIFT Wrapper - Rescale"),
        ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
                                                            eutils.heatmap,        "DeepLIFT Wrapper - RevealCancel"),
        ("lrp.z",                 {},                       eutils.heatmap,        "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           eutils.heatmap,        "LRP-Epsilon"),
    ]

    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    # Create analyzers.
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(data[0], batch_size=256, verbose=1)
        analyzers.append(analyzer)

    n = 10
    test_images = list(zip(data[2][:n], map(np.argmax,(data[3][:n]))))

    analysis = np.zeros([len(test_images), len(analyzers), 32, 32, 3])
    text = []


    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :, :]

        # Predict final activations, probabilites, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % label_to_class_name[y],    # ground truth label
                     "%.2f" % presm.max(),             # pre-softmax logits
                     "%.2f" % prob.max(),              # probabilistic softmax output
                     "%s" % label_to_class_name[y_hat] # predicted label
                    ))

        for aidx, analyzer in enumerate(analyzers):
            try:
                # Analyze.
                a = analyzer.analyze(x)

                # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                a = eutils.postprocess(a)
                # Apply analysis postprocessing, e.g., creating a heatmap.
                a = methods[aidx][2](a)
                # Store the analysis.
                analysis[i, aidx] = a[0]
            except :
                print('')


    grid = [[analysis[i, j] for j in range(analysis.shape[1])]
            for i in range(analysis.shape[0])]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method[3]) for method in methods]

    # Plot the analysis.
    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                           file_name="classification/model")



