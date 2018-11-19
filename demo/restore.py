import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

index=2

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={input_x: v_xs, prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={input_x: v_xs, input_y: v_ys, prob: 1})
    return result

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/my_model.meta')
    new_saver.restore(sess, './model/my_model')

    graph = tf.get_default_graph()
    prediction = tf.get_collection('prediction')[0]
    input_x = graph.get_operation_by_name("input_x").outputs[0]
    input_y = graph.get_operation_by_name("input_y").outputs[0]
    prob = graph.get_operation_by_name("prob").outputs[0]

    x = mnist.test.images[index].reshape(1, 784)
    y = mnist.test.labels[index].reshape(1, 10)  # 转为one-hot形式
    print (y)

    res = sess.run(prediction, feed_dict={input_x: x, prob : 1 })
    print (res)

    print("Actual class: ", str(sess.run(tf.argmax(y, 1))), \
          ", predict class ",str(sess.run(tf.argmax(res, 1))), \
          ", predict ", str(sess.run(tf.equal(tf.argmax(y, 1), tf.argmax(res, 1))))
          )

    print(compute_accuracy(
        mnist.test.images[:1000], mnist.test.labels[:1000]))
