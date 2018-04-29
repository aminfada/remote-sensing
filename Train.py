import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from glob import glob
import time
from os import listdir
from os.path import isfile, join
import os
import matplotlib.image as readimg
from resizeimage import resizeimage
from tensorflow.contrib.layers import dropout
from Reading_Dataset import read_labeled_image_list as RD
from Reading_Dataset import read_images_from_disk as RI
import Model

mypath = '../training-set/'
mypath_val = '../validation-set/'
epoch = 500
dropout_keep_prob = 0.6
learning_rate = 0.0001
interval_saver = 3


labels_list = []
for folder_index_vec, folder_vec in enumerate(listdir(mypath)):
    labels_list.append(folder_vec)
countlabel = folder_index_vec + 1

print(countlabel)

input_images = tf.placeholder(tf.float32, [256, 256,3], name="input")
input_labels = tf.placeholder(tf.float32, [None, countlabel], name="labels")

is_training = tf.placeholder(tf.bool, name='is_training')
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False)
#he_init = tf.contrib.layers.xavier_initializer()
epsilon = 1e-3
parameter = 0

images_normalized = tf.image.per_image_standardization(input_images)
images = tf.reshape(images_normalized, shape=[-1, 256, 256, 3])




with tf.variable_scope('conv1'):
    conv1_output = Model.convolutional_layer(images,1,[3, 3, 3, 60],2,False)


with tf.variable_scope('pool1'):
    pool1_output = Model.pooling_layer(conv1_output, 1, 4, 2)


with tf.variable_scope('conv2'):
    conv2_output = Model.convolutional_layer(pool1_output, 2, [1, 1, 60, 20], 1, False)


with tf.variable_scope('conv3'):
    kernel = tf.get_variable('kernel', [3, 3, 20, 90], initializer=he_init)
    biases = tf.get_variable('biases', 90, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv2_output, kernel, strides=[1, 1, 1, 1], padding='SAME')

    kernelres = tf.get_variable('kernelres', [1, 1, 60, 90], initializer=he_init)
    resinput = tf.nn.conv2d(pool1_output, kernelres, strides=[1, 1, 1, 1], padding='SAME')

    conv = conv + resinput
    conv3_output = tf.nn.relu(conv + biases)


with tf.variable_scope('pool2'):
    pool2_output = Model.pooling_layer(conv3_output, 2, 3, 2)


with tf.variable_scope('conv4'):
    conv4_output = Model.convolutional_layer(pool2_output, 4, [1, 1, 90, 40], 1, False)


with tf.variable_scope('conv5'):
    kernel = tf.get_variable('kernel', [3, 3, 40, 130], initializer=he_init)
    biases = tf.get_variable('biases', 130, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv4_output, kernel, strides=[1, 1, 1, 1], padding='SAME')

    kernelres = tf.get_variable('kernelres', [1, 1, 90, 130], initializer=he_init)
    resinput = tf.nn.conv2d(pool2_output, kernelres, strides=[1, 1, 1, 1], padding='SAME')

    conv = conv + resinput
    conv5_output = tf.nn.relu(conv + biases)


with tf.variable_scope('pool3'):
    pool3_output = Model.pooling_layer(conv5_output, 3, 3, 2)

with tf.variable_scope('conv6'):
    conv6_output = Model.convolutional_layer(pool3_output, 6, [1, 1, 130, 60], 1, False)

with tf.variable_scope('conv7'):
    conv7_output = Model.convolutional_layer(conv6_output, 7, [1, 3, 60, 190], 1, False)

with tf.variable_scope('conv8'):
    kernel = tf.get_variable('kernel', [3, 1, 190, 190], initializer=he_init)
    biases = tf.get_variable('biases', 190, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv7_output, kernel, strides=[1, 1, 1, 1], padding='SAME')

    kernelres = tf.get_variable('kernelres', [1, 1, 130, 190], initializer=he_init)
    resinput = tf.nn.conv2d(pool3_output, kernelres, strides=[1, 1, 1, 1], padding='SAME')

    conv = conv + resinput
    conv8_output = tf.nn.relu(conv + biases)


with tf.variable_scope('pool4'):
    pool4_output = Model.pooling_layer(conv8_output, 4, 3, 2)

with tf.variable_scope('conv9'):
    conv9_output = Model.convolutional_layer(pool4_output, 9, [1, 1, 190, 60], 1, False)


with tf.variable_scope('conv10'):
    conv10_output = Model.convolutional_layer(conv9_output, 10, [1, 3, 60, 220], 1,False)

with tf.variable_scope('conv11'):
    kernel = tf.get_variable('kernel', [3, 1, 220, 220], initializer=he_init)
    biases = tf.get_variable('biases', 220, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv10_output, kernel, strides=[1, 1, 1, 1], padding='SAME')

    kernelres = tf.get_variable('kernelres', [1, 1, 190, 220], initializer=he_init)
    resinput = tf.nn.conv2d(pool4_output, kernelres, strides=[1, 1, 1, 1], padding='SAME')

    conv = conv + resinput
    conv11_output = tf.nn.relu(conv + biases)


with tf.variable_scope('pool5'):
    pool5_output = Model.pooling_layer(conv11_output, 5, 3, 2)

with tf.variable_scope('conv12'):
    conv12_output =  Model.convolutional_layer(pool5_output, 12, [1, 1, 220, 90], 1,False)

with tf.variable_scope('conv13'):
    conv13_output = Model.convolutional_layer(conv12_output, 13, [1, 3, 90, 260], 1,False)

with tf.variable_scope('conv14'):
    kernel = tf.get_variable('kernel', [3, 1, 260, 260], initializer=he_init)
    biases = tf.get_variable('biases', 260, initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv13_output, kernel, strides=[1, 1, 1, 1], padding='SAME')

    kernelres = tf.get_variable('kernelres', [1, 1, 220, 260], initializer=he_init)
    resinput = tf.nn.conv2d(pool5_output, kernelres, strides=[1, 1, 1, 1], padding='SAME')

    conv = conv + resinput
    conv14_output = tf.nn.relu(conv + biases)


with tf.variable_scope('pool6'):
    pool6_output = Model.pooling_layer(conv14_output, 6, 3, 2)

with tf.variable_scope('neural_network'):
    w = tf.get_variable('weights', [2 * 2 * 260, 1024], initializer=he_init)
    b = tf.get_variable('biases', [1024], initializer=tf.zeros_initializer())

    pool6 = tf.reshape(pool6_output, [-1, 2 * 2 * 260])
    nn_mult = tf.matmul(pool6, w) + b

    neural_network = tf.nn.relu(nn_mult)


with tf.variable_scope('softmax'):
    w = tf.get_variable('weights', [1024, countlabel], initializer=he_init)
    b = tf.get_variable('biases', [countlabel], initializer=tf.zeros_initializer())

    logits = tf.matmul(neural_network, w) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_labels))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
print("number of parameter :")





image_list, label_list, datasample = RD(mypath,countlabel,labels_list)
images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
input_queue = tf.train.slice_input_producer([images, labels])
image, label = RI(input_queue)


image_list_val, label_list_val, datasample_val = RD(mypath_val,countlabel,labels_list)
images_val = tf.convert_to_tensor(image_list_val, dtype=tf.string)
labels_val = tf.convert_to_tensor(label_list_val, dtype=tf.int32)
input_queue_val = tf.train.slice_input_producer([images_val, labels_val])
image_val, label_val = RI(input_queue_val)

saver = tf.train.Saver()
step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
stepplt = 0

iteration_accuracies_list = []
iteration_accuracies_list_val = []
iteration_losses_list = []
iteration_losses_list_val = []
kernel_1_death_neurons = 0


#config = tf.ConfigProto()
#config.log_device_placement = True
mysess = tf.Session()

with mysess as sess:

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    total_correct_preds = 0

    start_time = time.time()

    plot_range = 0



    for z in range(epoch):
        step = step + 1
        stepplt = stepplt + 1
        sample = 0
        iteration_total_correct_preds = 0
        iteration_loss = 0
        for i in range(datasample):
            image_lis, label_lis= sess.run([image,label])

            label_vec = np.zeros((1, countlabel))

            label_vec[0,label_lis-1]= 1
            tf.cast(label_vec, tf.int32)

            #plt.imshow(image_lis)
            #plt.title(label_vec)
            #plt.show()

            _,loss_score,softmax_score = sess.run([optimizer,loss,logits] ,feed_dict={input_images:image_lis , input_labels:label_vec ,is_training: True})

            preds = tf.nn.softmax(softmax_score)

            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label_vec, 1))
            correct_preds_run = sess.run(correct_preds)
            iteration_total_correct_preds += tf.cast(correct_preds, tf.float32)
            iteration_loss += loss_score

            sample = sample+1
            print('%d th smaple has been readed and its prediction is %d with loss of: %f' %(sample,correct_preds_run,loss_score))


        iteration_accuracy = sess.run(iteration_total_correct_preds)/datasample

        iteration_losses_list.append(iteration_loss)


        finished_time = time.time() - start_time

        print('%d th itereation has been completed  in %d seconds with the overal accuracy of %f and loss of: %f with learning rate of: %f' %(z+1,finished_time,iteration_accuracy,iteration_loss,learning_rate))
        with open("log.txt", "a", encoding='utf8') as text_file:
            text_file.write('\n %d th itereation has been completed  in %d seconds with the overal accuracy of %f and loss of: %f with learning rate of: %f' %(z+1,finished_time,iteration_accuracy,iteration_loss,learning_rate))

        if (z+1)%interval_saver == 0:
            kernel1o = ['hello','world']
            for dea in kernel1o:
                 print(dea)

            iteration_accuracies_list.append(iteration_accuracy)

            iteration_total_correct_preds_val = 0
            sample_val = 0
            iteration_loss_val=0
            for v in range(40):
                image_lis_val, label_lis_val = sess.run([image_val, label_val])

                label_vec = np.zeros((1, countlabel))

                label_vec[0, label_lis_val - 1] = 1
                tf.cast(label_vec, tf.int32)

                loss_score, softmax_score = sess.run([loss, logits],
                                                     feed_dict={input_images: image_lis_val, input_labels: label_vec,
                                                                is_training: False})

                preds = tf.nn.softmax(softmax_score)
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label_vec, 1))
                prediction_status = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

                prediction_status_run = sess.run(prediction_status)
                iteration_total_correct_preds_val += prediction_status_run

                sample_val = sample_val + 1
                print('%d th validation smaple has been readed and its prediction is %d and loss of:%d' % (sample_val, prediction_status_run, loss_score))

            finished_time = time.time() - start_time
            iteration_accuracy = iteration_total_correct_preds_val / 40

            print('validation has been completed  in %d seconds with the overal accuracy of %f' % (finished_time, iteration_accuracy))
            with open("log.txt", "a", encoding='utf8') as text_file:
                text_file.write('\n validation has been completed  in %d seconds with the overal accuracy of %f' % (finished_time, iteration_accuracy))

            iteration_accuracies_list_val.append(iteration_accuracy)
            iteration_loss_val += loss_score
            iteration_losses_list_val.append(iteration_loss_val)

            print(iteration_accuracies_list)
            print(iteration_accuracies_list_val)

            saver.save(sess, './saved_graph/remote_sensing', global_step=step)
            epoch_for_plot_loss = np.arange(0, z + 1)
            epoch_for_plot_other = np.arange(0, z + 1 , interval_saver)
            print(epoch_for_plot_loss)
            print(epoch_for_plot_other)
            print(iteration_losses_list)
            print(iteration_accuracies_list)
            print(iteration_accuracies_list_val)
            plt.subplot(221)
            plt.plot(epoch_for_plot_loss, iteration_losses_list, 'g--',label='training loss')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.subplot(222)
            plt.plot(epoch_for_plot_other, iteration_accuracies_list_val, 'b',label='validation accuracies')
            plt.plot(epoch_for_plot_other, iteration_accuracies_list, 'r--',label='training accuracies')
            plt.xlabel('Epoch')
            plt.ylabel('accuracies')
            plt.subplot(223)
            plt.plot(epoch_for_plot_other, iteration_losses_list_val, 'b', label='validation losses')
            plt.xlabel('Epoch')
            plt.ylabel('losses')

            plt.grid()
            plt.savefig('epoch' + str(z+1) + '.png')
            print(iteration_accuracies_list)

    coord.request_stop()
    coord.join(threads)
