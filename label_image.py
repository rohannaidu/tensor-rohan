import tensorflow as tf
import sys
import numpy as np
import os

# input folder
image_path = sys.argv[1]

# Write to CSV
def writeCSV(image_path_fn, predicted_label_for_test_fn):
    filename, file_extension = os.path.splitext(os.path.basename(image_path_fn))
    line_to_write = '%s,%s \n' % (filename, predicted_label_for_test_fn) 
    fd = open('./tf_submission.csv', 'a')
    fd.write(line_to_write)
    fd.close()

def getPrediction(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("./retrained_labels_v3.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("./retrained_graph_v3.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
#            print('%s (score = %.5f)' % (human_string, score))

        # To write to CSV
        top_k = top_k.astype(np.int32)
        if top_k[0] == 3 or top_k[0] == 0:
	    predicted_label_for_test = top_k[0] + 1
            print 'Prediction for this image %s is %s' % (os.path.basename(image_path), predicted_label_for_test)
	    writeCSV(image_path, predicted_label_for_test)
        elif top_k[0] == 2:
	    predicted_label_for_test = 2
	    print 'Prediction for this image %s is %s' % (os.path.basename(image_path), predicted_label_for_test)
            writeCSV(image_path, predicted_label_for_test)
        elif top_k[0] == 1:
	    predicted_label_for_test = 3
	    print 'Prediction for this image %s is %s' % (os.path.basename(image_path), predicted_label_for_test)
            writeCSV(image_path, predicted_label_for_test)


##################-MAIN-#########################

getPrediction(image_path)
