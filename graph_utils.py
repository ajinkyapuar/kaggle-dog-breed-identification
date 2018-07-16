import numpy as np
import tensorflow as tf


def load_graph(path):
    with tf.gfile.GFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def run_graph(image, path, input_layer, output_layer):
    graph = load_graph(path)

    x = graph.get_tensor_by_name(input_layer + ':0')
    y = graph.get_tensor_by_name(output_layer + ':0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [image]
        })
    return y_out


def load_labels(labels_file):
    labels = []
    proto_as_ascii_lines = tf.gfile.GFile(labels_file).readlines()
    for l in proto_as_ascii_lines:
        labels.append(l.rstrip())
    return labels


def convert_tensors_to_labels(tensors, labels, n):
    results = np.squeeze(tensors)
    top_k = results.argsort()[-n:][::-1]
    # template = "{} (score={:0.5f})"
    l = []
    for i in top_k:
        # print(template.format(labels[i], results[i]))
        l.append(labels[i])
    return l

