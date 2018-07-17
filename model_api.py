


import argparse
import json
import time

import tensorflow as tf
import numpy as np

from flask import Flask, request
from flask_cors import CORS

from graph_utils import *

BreedsGraph = {
        "path": "tf_poets/tf_files/breedCeption.pb",
        "inputLayer": "import/ResizeBilinear",
        "outputLayer": "import/final_result"
    }

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)


@app.route("/api/breeds", methods=['POST'])
def predict():
    print("POST Request Received")
    start = time.time()
    data = request.data.decode("utf-8")
    # print(data, type(data))
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        # print("PARAMS")
        # x_in = params['x']
        # print(np.array(list(params)), type(np.array(list(params))), np.array(list(params)).shape)
        x_in = np.array(list(params))
        # print(x_in.shape)

    # print(x_in)
    ##################################################
    # Tensorflow part
    ##################################################
    y_out = persistent_sess.run(y, feed_dict={
        x: x_in
    })
    # print(y_out)
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'y': y_out.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default=BreedsGraph['path'], type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    x = graph.get_tensor_by_name(BreedsGraph['inputLayer'] + ':0')
    y = graph.get_tensor_by_name(BreedsGraph['outputLayer'] + ':0')

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run(port=8000, debug=True)
