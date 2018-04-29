from os import listdir
import tensorflow as tf


def read_labeled_image_list(mypath,countlabel,labels_list):
    filenames = []
    labels = []
    for folder_index, folder in enumerate(listdir(mypath)):
        for file in listdir(mypath + '/' + folder):
            filenames.append(mypath + '/' + folder + '/' + file)
            labels.append(folder)

    countsample = len(filenames)
    for ite_r in range(countlabel):
        for ite_index, ite in enumerate(labels):
            if (ite == labels_list[ite_r]):
                labels[ite_index] = ite_r +1

    print(countsample)
    return filenames, labels, countsample


def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])

    example = tf.image.decode_png(file_contents, channels=3)
    return example, label
