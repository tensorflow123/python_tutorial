import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

index=1

def compute_accuracy(sess, prediction, input_x, keep_prob, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={input_x: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={input_x: v_xs, input_y: v_ys, keep_prob: 1})
    return result

signature_key = 'test_signature'
input_key_x = 'input_x'
input_key_y = 'input_y'
input_key_keep_prob = 'keep_prob'
output_key_prediction = 'prediction'

saved_model_dir='./model'
with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, ['model_final'], saved_model_dir)

    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name 
    x_tensor_name = signature[signature_key].inputs[input_key_x].name
    y_tensor_name = signature[signature_key].inputs[input_key_y].name
    keep_prob_tensor_name = signature[signature_key].inputs[input_key_keep_prob].name
    prediction_tensor_name = signature[signature_key].outputs[output_key_prediction].name

    # 获取tensor 并inference
    input_x = sess.graph.get_tensor_by_name(x_tensor_name)
    input_y = sess.graph.get_tensor_by_name(y_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(keep_prob_tensor_name)
    prediction = sess.graph.get_tensor_by_name(prediction_tensor_name)

    x = mnist.test.images[index].reshape(1, 784)
    y = mnist.test.labels[index].reshape(1, 10)  # 转为one-hot形式
    print (y)

    pred_y = sess.run(prediction, feed_dict={input_x: x, keep_prob : 1 })
    print (pred_y)

    print("Actual class: ", str(sess.run(tf.argmax(y, 1))), \
          ", predict class ",str(sess.run(tf.argmax(pred_y, 1))), \
          ", predict ", str(sess.run(tf.equal(tf.argmax(y, 1), tf.argmax(pred_y, 1))))
          )

    print(compute_accuracy(sess, prediction, input_x, keep_prob,
	mnist.test.images[:1000], mnist.test.labels[:1000]))
