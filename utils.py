from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize
import numpy as np
import random

import pdb

distributions = tf.contrib.distributions


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.cast(tf.pack(mean_arr), tf.float32)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.cast(tf.pack(sampled_arr), tf.float32)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]
  return logll


def dys_to_label(dys_fns):
    labels = []
    if dys_fns:
        #pos_sp = dys_fns.find(" ")
        #dys_str = str(dys_fns[0:pos_sp])
        dys_str = dys_fns
        if dys_str.lower() == 'Normal/None':
            labels = 0
        if dys_str.lower() == 'Mild':
            labels = 1
        if dys_str.lower() == 'Moderate':
            labels = 2
        if dys_str.lower() == 'Severe/Restrictive':
            labels = 3

            # print(labels)
    if labels==[]:
        print(" this label does not exist " +dys_fns)
        pdb.set_trace()
    return labels


def label_to_dys(labels):
    dys = []
    for cnt, labels in enumerate(labels):
        if labels == 0:
            dys[cnt] = 'None or Normal'
        if labels == 1:
            dys[cnt] = 'Mild'
        if labels == 2:
            dys[cnt] = 'Moderate'
        if labels == 3:
            dys[cnt] = 'Severe or Restrictive'
    return dys


def compute_multiClassRoc(gt,scores,num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for classes in range(num_classes):
        binary_gt = label_binarize(gt, classes=range(num_classes))
        fpr[classes], tpr[classes], _ = \
            roc_curve(binary_gt[:, classes], scores[:, classes])
        roc_auc[classes] = auc(fpr[classes], tpr[classes])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"], tpr["macro"], fpr["macro"], tpr, fpr, roc_auc

def augmentData(case,labels, n):
    tmp_indices = random.sample(range(len(case)), n)
    aug_case = [np.roll(case, indx).tolist() for indx in tmp_indices]
    aug_labels = [labels] * n
    return aug_case, aug_labels







