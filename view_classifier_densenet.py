from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
import pdb
import sys
import os
import random
import time
from  tensorflow.contrib.layers import batch_norm
import scipy.io as scio
from scipy.misc import imresize
from scipy.misc import imsave
from cv2 import imread
from itertools import cycle
import glob

from sklearn import metrics
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import loglikelihood, compute_multiClassRoc
from utils import augmentData
from config import Config
from sklearn.preprocessing import label_binarize

#from network import DenseNet
from tensorflow.contrib import rnn
#from processDysData import readDysfunctionExcelFile
#from processDysData_test import readDysfunctionExcelFile



np.set_printoptions(precision=3)

def load_data():
    dataset_path = {'ap4':[os.path.join(os.getcwd(),'AP4'),0],
                'ap2':[os.path.join(os.getcwd(),'AP2'),1],
                'doppler':[os.path.join(os.getcwd(), 'doppler'),2],
                'none':[os.path.join(os.getcwd(), 'none'),2]}
    dataset = []

    for items in dataset_path:
        label = dataset_path[items][1]
        label_name = items
        label_path = dataset_path[items][0]
        for study_dir in os.listdir(label_path):
            files = glob.glob(os.path.join(label_path,study_dir, '*.png'))
            if len(files)>=10:
                study = {'id':study_dir,
                         'view':label_name,
                         'image_files':files,
                         'label':label}
                dataset.append(study)

    """split the data in train and validation"""
    random.shuffle(dataset)
    train_end = int(np.ceil(len(dataset)*0.8))
    val_start = int(np.ceil(len(dataset)*0.8)+1)
    train_cases = dataset[0:train_end]
    valid_cases  = dataset[val_start:]

   #normalize the images in terms of mean and standard deviation
    try:
        with np.load('cardiacStats.npz') as mammoStats:
            mean_images = mammoStats['mean_images']
    except:
        mean_images, std_images = getImageStats(train_cases)
        imsave('mean_cardiac_image.png', mean_images)
        np.savez('cardiacStats.npz',
                 mean_images = mean_images)

    return dict(
        train_cases = train_cases,
        valid_cases = valid_cases,
        mean_images = mean_images)


def getImageStats(cases):
    bs = 30  # batch size
    stdData = []

    for t in range(0, len(cases), bs):
        batch = cases[t:min((t + bs, len(cases)))]
        for cnt_case,case in enumerate(batch):
            image_files = case['image_files']
            print('collecting image stats: batch starting with image {}\n'.format(t))
            tmp = getBatch(image_files)
            data = tmp
            meanBatch = np.mean(data, axis=0)
            if t == 0:
                meanData = meanBatch
            else:
                meanData = np.mean((meanData, meanBatch), axis=0)

    return meanData, stdData



def getBatch(images, *args):
    if len(args) == 1:
        meanData = args[0]
        meanData = np.float32(meanData)
    else:
        meanData = []

    imo = []
    for cine_images in images:
        oimt = np.float32(imread(cine_images))
        oimt = imresize(oimt[:, :, 1], (Config.image_size, Config.image_size), interp='cubic')
        #clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #imt=  clahe.apply(oimt)
        imt = oimt
        if meanData!=[]:
            norData = np.float32(imt)

            norData -= meanData
            # norData/= stdData
        else:
            norData = imt
        imo.append(norData)
    imageData = np.asarray(imo)
    return np.float32(imageData)


def iterate_minibatches(cases,
                        mean_images,
                        batchsize,
                        shuffle=True,
                        augment=False):
    if shuffle:
        np.random.shuffle(cases)
    for start_idx in range(0, len(cases) - batchsize + 1, batchsize):
        batch_cases = cases[start_idx:start_idx + batchsize]

        for indx_cases, batch_case in enumerate(batch_cases):
            image_filenames = batch_case['image_files']
            sample_filenames = random.sample(image_filenames, Config.num_frames)
            data_label =batch_case['label']
            image_data = getBatch(sample_filenames, mean_images)
            image_labels = [data_label]*len(image_data)
            if indx_cases==0:
                data_shape = image_data.shape
                image_data = image_data.reshape(data_shape[0],data_shape[1], data_shape[2],1)
                batch_data = image_data
                batch_labels = image_labels

            else:
                data_shape = image_data.shape
                image_data = image_data.reshape(data_shape[0],data_shape[1], data_shape[2],1)
                batch_data = np.concatenate([batch_data, image_data])
                batch_labels = np.concatenate([batch_labels, image_labels])
        yield batch_data, label_binarize(batch_labels, classes=[0,1,2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(input,in_features,out_features,kernel_size,with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + self.bias_variable([out_features])
    return conv


def batch_activ_conv(current,in_features,out_features,kernel_size):
    current = tf.nn.relu(current)
    current = conv2d(current,in_features,out_features,kernel_size)
    return current

def block(input,layers,in_features, growth):
    current = input
    features = in_features
    for idx in xrange(layers):
      tmp = batch_activ_conv(current, features, growth,3)
      current = tf.concat((current, tmp),3)
      features += growth
    return current, features

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')

def maxpool2d(x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


def cnn_modal(x):
    layers = int((Config.depth-4) /3)
    current = conv2d(x,1,16,3)

    current, features = block(current, layers, 16, 12)
    current = batch_activ_conv(current, features, features, 1)
    current = avg_pool(current, 2)

    current, features = block(current, layers, features, 12)
    current = batch_activ_conv(current, features, features, 1)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12)
    current = batch_activ_conv(current, features, features, 1)
    current = avg_pool(current, 2)
    current, features = block(current, layers, features, 12)
    current = tf.nn.relu(current)
    current = avg_pool(current, 8)
    final_dim = features
    current = tf.reshape(current, [-1, final_dim])
    return current


def main(mode=None):

    #make the folder to save the training model

    model_folder = 'multi_view_model'
    fig_folder = 'multi_view_figure'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    if not os.path.isdir(fig_folder):
        os.mkdir(fig_folder)

    #load the cardiac data
    if (mode.lower() =='--train') or (mode.lower() =='--validate'):
        print("Loading training and validation data...")
        data = load_data()
        train_cases = data['train_cases']
        valid_cases = data['valid_cases']
        mean_images = data['mean_images']

    #
    logging.getLogger().setLevel(logging.INFO)


    images_ph = tf.placeholder(tf.float32,
                                   [None,
                                    Config.image_size,
                                    Config.image_size,
                                    Config.num_channels])

    labels_ph = tf.placeholder(tf.int64, [None, Config.num_classes])
    outputs_cnn = cnn_modal(images_ph)

    with tf.variable_scope('cls'):
        w_logit = weight_variable((outputs_cnn[-1].get_shape().as_list()[0],Config.num_classes))
        b_logit = bias_variable((Config.num_classes,))
    logits = tf.matmul(outputs_cnn, w_logit)+ b_logit
    # cross-entropy.
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
    xent = tf.reduce_mean(xent)
   # var_list  = tf.trainable_variables()
    #l2 = tf.add_n([tf.nn.l2_loss(var) for var in var_list])
    loss = xent #+ l2*Config.weight_decay
    #grads = tf.gradients(loss, var_list)
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0),
        trainable=False)
    starter_learning_rate = Config.lr_start
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        Config.num_epoch,
        0.97,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, Config.lr_min)
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)

        #call the saver operator to save the variables
    if mode.lower() == "--train":
        best_acc = 0
        saver = tf.train.Saver()
        #config_device = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_acc = np.zeros(Config.num_epoch)
            train_loss = np.zeros(Config.num_epoch)
            #
            val_acc = np.zeros(Config.num_epoch)
            val_loss = np.zeros(Config.num_epoch)

            for epoch in range(Config.num_epoch):
                training_batch = iterate_minibatches(train_cases,
                                             mean_images,
                                             Config.batch_size,
                                             shuffle=True,
                                             augment=False)
                cnt_batch = 0
                tr_loss_tot = 0

                for tr_images, tr_labels in training_batch:
                    # create the mirror images
                    cnt_batch += 1

                    softmax_val, loss_val, lr_val, _ = sess.run(
                        [logits, loss, learning_rate, train_op],
                        feed_dict={
                            images_ph: tr_images,
                            labels_ph: tr_labels,
                        })

                    logging.info('Batch {} Epoch {}: '
                                 'lr = {:3.6f}'.format(cnt_batch, epoch, lr_val))
                    logging.info('Batch {} Epoch {}:\tloss = {:3.4f}'
                                 .format(cnt_batch, epoch, loss_val))

                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    tr_loss_tot += loss_val

                    if cnt_batch == 1:
                        train_scores = softmax_val
                        train_predid = pred_labels_val
                        train_gt = np.argmax(tr_labels,1)
                    else:
                        train_scores = np.concatenate([train_scores, softmax_val])
                        train_predid = np.concatenate([train_predid, pred_labels_val])
                        train_gt = np.concatenate([train_gt, np.argmax(tr_labels,1)])

                mean_train_loss = tr_loss_tot/float(cnt_batch)
                mean_train_acc = np.mean(train_gt == train_predid)
                train_confusion_matrix = metrics.confusion_matrix(train_gt, train_predid)
                train_confusion_matrix = np.float32(train_confusion_matrix) / \
                                         train_confusion_matrix.sum(axis=1)[:, np.newaxis]

                cnt_batch = 0
                val_loss_tot = 0

                val_batch = iterate_minibatches(valid_cases,
                                                mean_images,
                                                Config.batch_size,
                                                shuffle=True,
                                                augment=False)

                for val_images, val_labels in val_batch:
                    cnt_batch += 1
                    softmax_val, loss_val = sess.run([logits, loss],
                                                     feed_dict={
                                                        images_ph: val_images,
                                                        labels_ph: val_labels})

                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    val_loss_tot += loss_val

                    if cnt_batch == 1:
                        val_scores = softmax_val
                        val_predid = pred_labels_val
                        val_gt = np.argmax(val_labels,1)
                    else:
                        val_scores = np.concatenate([val_scores, softmax_val])
                        val_predid = np.concatenate([val_predid, pred_labels_val])
                        val_gt = np.concatenate([val_gt, np.argmax(val_labels,1)])

                mean_val_loss = val_loss_tot / float(cnt_batch)
                mean_val_acc = np.mean(val_gt == val_predid)
                val_confusion_matrix = metrics.confusion_matrix(val_gt, val_predid)
                val_confusion_matrix = np.float32(val_confusion_matrix) / \
                                       val_confusion_matrix.sum(axis=1)[:, np.newaxis]

                #
                train_acc[epoch] = mean_train_acc
                train_loss[epoch] = mean_train_loss

                #
                val_acc[epoch] = mean_val_acc
                val_loss[epoch] = mean_val_loss

                #save the best model
                if mean_val_acc>best_acc:
                    best_acc = mean_val_acc
                    saver.save(sess, model_folder+"/"+"best_model"+".ckpt")
                    logging.info("Best Model saved in file:  %s "
                                 %model_folder+"/"+"best_model"+".ckpt")

                np.savetxt('train_acc.csv', train_acc[0:epoch + 1], delimiter='\n', fmt='%.3f')
                np.savetxt('test_acc.csv', val_acc[0:epoch + 1], delimiter='\n', fmt='%.3f')


                #display the result on the command line
                print("\n***************************************")
                logging.info('Epoch {} train accuracy = {:3f}'
                             .format(epoch, mean_train_acc))
                logging.info('Epoch {} train loss = {:3f}'
                             .format(epoch, mean_train_loss))
                print("Confusion Matrix:\n",train_confusion_matrix)
                print("-----------------------------------------")
                logging.info('Epoch {} val accuracy = {:3f}'
                             .format(epoch, mean_val_acc))
                logging.info('Epoch {} val loss = {:3f}'
                             .format(epoch, mean_val_loss))
                print("Confusion Matrix:\n",val_confusion_matrix)
                print("***************************************\n")

                # plot and save the acc
                fig = Figure(figsize=(8, 8))
                fig.clear()
                canvas = FigureCanvas(fig)
                acc_plot = fig.add_subplot(111)
                train_line, = acc_plot.plot(train_acc[0:epoch + 1], 'g')
                val_line, = acc_plot.plot(val_acc[0:epoch + 1], 'r')
                acc_plot.legend((train_line, val_line), ('Training', 'Validation'))
                acc_plot.title.set_text('')
                acc_plot.set_xlabel('Num. Epoch')
                acc_plot.set_ylabel('Accuracy')
                canvas_name = fig_folder + '/' + 'ram_dys_acc.png'
                canvas.print_figure(canvas_name)

                # plot and save the loss
                fig = Figure(figsize=(8, 8))
                fig.clear()
                canvas = FigureCanvas(fig)
                loss_plot = fig.add_subplot(111)
                train_line, = loss_plot.plot(train_loss[0:epoch + 1], 'g')
                val_line, = loss_plot.plot(val_loss[0:epoch + 1], 'r')
                loss_plot.legend((train_line, val_line), ('Training', 'Validation'))
                loss_plot.title.set_text('')
                loss_plot.set_xlabel('Num. Epoch')
                loss_plot.set_ylabel('Loss')
                canvas_name = fig_folder + '/' + 'ram_dys_loss.png'
                canvas.print_figure(canvas_name)

    if mode.lower() == "--validate":
        # calculate the accuracy for the validation data
        #config_device = tf.ConfigProto(device_count={'CPU': 0})
        model = model_folder +"/"+"best_model.ckpt"

        config_device = tf.ConfigProto(log_device_placement=True)
        saver = tf.train.Saver()
        with tf.Session(config=config_device) as sess:
            saver.restore(sess, model)
            for indx_dataset in range(0, 2, 1):
                if indx_dataset == 0:
                    cardiac_cases = train_cases
                    tag = "Train"

                if indx_dataset == 1:
                    cardiac_cases = valid_cases
                    tag = "Validate"

                # loc_net.sampling = True
                num_batches = 0

                cardiac_batch = iterate_minibatches(cardiac_cases,
                                             mean_images,
                                             Config.eval_batch_size,
                                             shuffle=True,
                                             augment=False)

                for val_images, val_labels in cardiac_batch:


                    start_time = time.time()
                    num_batches += 1
                    labels_bak = val_labels
                    softmax_val, _ = sess.run([logits, loss],
                                                     feed_dict={
                                                        images_ph: val_images,
                                                        labels_ph: val_labels})

                    pred_labels_val = np.argmax(softmax_val, 1)
                    pred_labels_val = pred_labels_val.flatten()
                    nor_scores = (softmax_val - np.min(softmax_val)) /(np.max(softmax_val)-np.min(softmax_val))
                    global_scores  = np.mean(nor_scores,axis=0)
                    global_scores.reshape(3,1)
                    global_predid = np.argmax(global_scores)
                    if num_batches == 1:
                        scores = nor_scores
                        predid = pred_labels_val
                        gscores = global_scores
                        gt = np.argmax(labels_bak,1)
                        ggt = np.unique(np.argmax(labels_bak,1))

                    else:
                        scores = np.concatenate([scores, softmax_val])
                        gscores = np.vstack([gscores, global_scores])
                        predid = np.concatenate([predid, pred_labels_val])
                        gt = np.concatenate([gt, np.argmax(labels_bak,1)])
                        ggt = np.concatenate([ggt, np.unique(np.argmax(labels_bak,1))])
                    elapsed_time = time.time()-start_time
                    print("Evaluating {} case {}, Time Taken = {:3f} sec".
                          format(tag,num_batches,elapsed_time))
                """Local Evaluation"""
                mean_acc = np.mean(gt == predid)
                mean_auc, mean_tpr, mean_fpr, tpr, fpr, roc_auc= \
                    compute_multiClassRoc(gt, scores, Config.num_classes)
                cm = metrics.confusion_matrix(gt, predid)
                nor_cm = np.float32(cm) / cm.sum(axis=1)[:, np.newaxis]

                """Global Evaluation"""
                gpredid = np.argmax(gscores,1)
                mean_gacc = np.mean(ggt == gpredid)
                mean_gauc, mean_gtpr, mean_gfpr, gtpr, gfpr, groc_auc= \
                    compute_multiClassRoc(ggt, gscores, Config.num_classes)
                gcm = metrics.confusion_matrix(ggt, gpredid)
                gnor_cm = np.float32(gcm) / gcm.sum(axis=1)[:, np.newaxis]


                #plot the local Roc curve for each class
                fig = Figure(figsize=(8, 8))
                fig.clear()
                canvas = FigureCanvas(fig)
                roc_plot = fig.add_subplot(111)
                roc_plot.plot(mean_fpr, mean_tpr,
                                          label="Mean ROC (AUC = {:0.2f})".format(mean_auc),
                                          color='deeppink',
                                          linestyle='--',
                                          linewidth=2)
                colors = cycle(['yellow', 'blue', 'green'])


                for i, color in zip(range(Config.num_classes), colors):
                    clabel = "ROC Class {} (AUC = {:0.2f})".format(Config.classes[i], roc_auc[i])
                    roc_plot.plot(fpr[i], tpr[i],
                                  label=clabel,
                                  color=color,
                                  linestyle='-',
                                  linewidth=2)
                roc_plot.axes.set_xlim([0.0, 1.0])
                roc_plot.axes.set_ylim([0.0, 1.05])
                roc_plot.set_xlabel('False Positive Rate')
                roc_plot.set_ylabel('True Positive Rate')
                roc_plot.set_title('Receiver operating characteristic for view classification')
                roc_plot.legend(loc="lower right")

                canvas_name = fig_folder + '/' + tag + '_ram_roc_curves_local.png'
                canvas.print_figure(canvas_name)


                #plot the global roc curve for each class
                fig = Figure(figsize=(8, 8))
                fig.clear()
                canvas = FigureCanvas(fig)
                roc_plot = fig.add_subplot(111)
                roc_plot.plot(mean_gfpr, mean_gtpr,
                              label="Mean ROC (AUC = {:0.2f})".format(mean_auc),
                              color='deeppink',
                              linestyle='--',
                              linewidth=2)
                colors = cycle(['yellow', 'blue', 'green'])
                for i, color in zip(range(Config.num_classes), colors):
                    clabel = "ROC Class {} (AUC = {:0.2f})".format(Config.classes[i], groc_auc[i])
                    roc_plot.plot(gfpr[i], gtpr[i],
                                  label=clabel,
                                  color=color,
                                  linestyle='-',
                                  linewidth=2)
                roc_plot.axes.set_xlim([0.0, 1.0])
                roc_plot.axes.set_ylim([0.0, 1.05])
                roc_plot.set_xlabel('False Positive Rate')
                roc_plot.set_ylabel('True Positive Rate')
                roc_plot.set_title('Receiver operating characteristic for view classification')
                roc_plot.legend(loc="lower right")

                canvas_name = fig_folder + '/' + tag + '_ram_roc_curves_global.png'
                canvas.print_figure(canvas_name)

                #display the final results
                print("\n***************************************")
                logging.info('Final {} average accuracy on individual images = {:3f}'
                             .format(tag, mean_acc))
                logging.info('Final {} mean AUC on individual images = {:3f}'
                             .format(tag, mean_auc))
                print("Local Confusion Matrix:\n", nor_cm)
                print("\n----------------------------------------")
                logging.info('Final {} average accuracy on individual study = {:3f}'
                             .format(tag, mean_gacc))
                logging.info('Final {} mean AUC individual study= {:3f}'
                             .format(tag, mean_gauc))
                print("Global Confusion Matrix:\n", gnor_cm)
                print("***************************************\n")
    if mode.lower() == "--test":

        src_test_folder = Config.src_test_folder
        model = Config.model_folder +"/"+"best_model.ckpt"


        #load the mean image
        with np.load('cardiacStats.npz') as cardiacStats:
            mean_images = cardiacStats['mean_images']

        if not os.path.isdir(src_test_folder):
            print('The Source Folder for mat files does not exist.. :-)')
            exit(0)

        dest_test_folder = Config.dest_test_folder
        if not os.path.isdir(dest_test_folder):
            os.mkdir(dest_test_folder)
        ap4_dest = os.path.join(dest_test_folder,'AP4')
        ap2_dest = os.path.join(dest_test_folder, 'AP2')
        dopp_dest = os.path.join(dest_test_folder, 'Doppler')
        others_dest = os.path.join(dest_test_folder, 'Others')
        if not os.path.isdir(ap4_dest):
            os.mkdir(ap4_dest)
        if not os.path.isdir(ap2_dest):
            os.mkdir(ap2_dest)
        if not os.path.isdir(dopp_dest):
            os.mkdir(dopp_dest)
        if not os.path.isdir(others_dest):
            os.mkdir(others_dest)

        """Read the individual cases from the source folder and cluster them using learned model"""
        import matlab.engine
        eng = matlab.engine.start_matlab()
        list_cases_dir = [os.path.join(src_test_folder, case) for case in os.listdir(src_test_folder)
                          if os.path.isdir(os.path.join(src_test_folder, case))]
        if Config.cpu:
            config_device = tf.ConfigProto(device_count = {'GPU':0})
        else:
            config_device = tf.ConfigProto(log_device_placement=False)
        saver = tf.train.Saver()
        with tf.Session(config=config_device) as sess:
            saver.restore(sess, model)
            for cnt_case in range(Config.start_idx,len(list_cases_dir)):
                case_study = list_cases_dir[cnt_case]
                list_mat_files = [os.path.join(case_study, matfile) for matfile in os.listdir(case_study)
                                  if matfile.endswith('.mat')]
                cnt_ap2 = 0;cnt_ap4 = 0;cnt_others=0;cnt_doppler=0
                for cnt_mat,study in enumerate(list_mat_files):
                    start_time = time.time()
                    """process the data from the matlab engine"""
                    #remove the temorary directory
                    if os.path.isdir('tmp_folder'):
                        os.system("rm -r tmp_folder")
                    id, study_date, isdoppler, test_case = eng.morpho_crop_test(study, nargout=4)
                    id = str(id)
                    study_date = str(study_date)
                    list_images = [os.path.join(os.getcwd(), 'tmp_folder', image_filename)
                                   for image_filename in os.listdir('tmp_folder')
                                   if image_filename.endswith('.png')]

                    if list_images:

                        if isdoppler == 1:
                            cnt_doppler += 1
                            dest_study_path = os.path.join(dopp_dest,id + '_' + study_date + '_' + '%03d' % cnt_doppler)
                            if not os.path.isdir(dest_study_path):
                                os.mkdir(dest_study_path)
                            for idx,filename in enumerate(list_images):
                                image = imread(filename)
                                """save the frames as png file in the study folder"""
                                imsave(os.path.join(dest_study_path, 'doppler_image' + '%03d' % idx + '.png'), image)
                        else:
                            test_images = getBatch(list_images, mean_images)
                            data_shape = test_images.shape
                            test_images = test_images.reshape(data_shape[0], data_shape[1], data_shape[2], 1)

                            test_scores= sess.run([logits],feed_dict={images_ph: test_images})
                            nor_scores = (test_scores - np.min(test_scores)) / (np.max(test_scores) - np.min(test_scores))
                            global_scores = np.mean(nor_scores, 1)
                            predid_label = np.argmax(global_scores, 1)

                            if predid_label==0:
                                cnt_ap4+=1
                                dest_study_path = os.path.join(ap4_dest,
                                                               id + '_' + study_date + '_' + '%03d' % cnt_ap4)
                                if not os.path.isdir(dest_study_path):
                                    os.mkdir(dest_study_path)
                                for idx, filename in enumerate(list_images):
                                    image = imread(filename)
                                    """save the frames as png file in the study folder"""
                                    imsave(os.path.join(dest_study_path, 'ap4_image' + '%03d' % idx + '.png'),
                                           image)
                            if predid_label==1:
                                cnt_ap2+=1
                                dest_study_path = os.path.join(ap2_dest,
                                                               id + '_' + study_date + '_' + '%03d' % cnt_ap2)
                                if not os.path.isdir(dest_study_path):
                                    os.mkdir(dest_study_path)
                                for idx, filename in enumerate(list_images):
                                    image = imread(filename)
                                    """save the frames as png file in the study folder"""
                                    imsave(os.path.join(dest_study_path, 'ap2_image' + '%03d' % idx + '.png'),
                                           image)

                            if predid_label == 2:
                                cnt_others += 1
                                dest_study_path = os.path.join(others_dest,
                                                               id + '_' + study_date + '_' + '%03d' % cnt_others)
                                if not os.path.isdir(dest_study_path):
                                    os.mkdir(dest_study_path)
                                for idx, filename in enumerate(list_images):
                                    image = imread(filename)
                                    """save the frames as png file in the study folder"""
                                    imsave(os.path.join(dest_study_path, 'others_image' + '%03d' % idx + '.png'),
                                           image)

                    print('Finished clusterring data case {:.0f} study {:.0f} in {:.2f} sec'
                          .format(cnt_case, cnt_mat,time.time()-start_time))


            print("done.")




if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DenseNet for view classification problem \n "
              "Model for diastolic dysfunction classification.\n")
        print("Network architecture using multiview CNN.")
        print()
        print("Mode: {--train, --test}, --train for training, --test for inference, Default is --train")
        print("Model: {model path} for finetuning if mode == {--train}, inference if mode == {--test}")

    else:
        kwargs = {}
        kwargs['mode'] = str(sys.argv[1])
        main(**kwargs)
