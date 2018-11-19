import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/my_model.meta')
    new_saver.restore(sess, './model/my_model')

    graph = tf.get_default_graph()
    prediction = tf.get_collection('prediction')[0]
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    prob = graph.get_operation_by_name("prob").outputs[0]

    x = mnist.test.images[0].reshape(1, 784)
    y = mnist.test.labels[0].reshape(1, 10)  # 转为one-hot形式

    res = sess.run(prediction, feed_dict={input_x: x, prob : 1 })

    print("Actual class: ", str(sess.run(tf.argmax(y, 1))), \
          ", predict class ",str(sess.run(tf.argmax(res, 1))), \
          ", predict ", str(sess.run(tf.equal(tf.argmax(y, 1), tf.argmax(res, 1))))
          )
