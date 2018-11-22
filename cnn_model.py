from  __future__  import division, print_function, absolute_import

import tensorflow as tf

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

tf.test.gpu_device_name()

# Some numbers
batch_size = 128
display_step = 5
num_input = 784
num_classes = 10

def conv_layer(inputs, channels_in, channels_out, strides=1):

    # Create variables
    # first two numbers 5, 5 are filter size (width height), Channel_out is number of filters, channel_in is number of channels in previous input

    w = tf.Variable(tf.random_normal([5, 5, channels_in, channels_out]))
    b = tf.Variable(tf.random_normal([channels_out]))

    # We can double check the device that this variable was placed on
    print("Variable w is placed on device :", w.device)
    print("Variable b is placed on device :", b.device)

    # Define Ops
    x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    # Non-linear activation
    return tf.nn.softmax(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def CNN(x, devices):

    #Put first half of network on device 0

    with tf.device(devices[0]):

        x = tf.reshape(x, shape=[-1, 28, 28, 1])                                               #[num_images, img_height, img_width, number_of_channels_image_has]

        # Convolution Layer
        conv1 = conv_layer(x, 1, 32, strides=1)
        pool1 = maxpool2d(conv1)

        # Convolution Layer
        conv2 = conv_layer(pool1, 32, 96, strides=1)
        pool2 = maxpool2d(conv2)

        #convolution Layer
        conv3 = conv_layer(pool2, 96, 136, strides=1)

        #convolution Layer
        conv4 = conv_layer(conv3, 136, 196, strides=1)


    #Put second half of network on device 1

    with tf.device(devices[1]):

        # Fully connected layer
        fc1 = tf.reshape(conv4, [-1, 7 * 7 * 196])                                                #[num_images, img_height*img_width*num_channels] after downsampling
        w1 = tf.Variable(tf.random_normal([7 * 7 * 196, 1024]))                                   #[num_inputs, num_outputs]
        b1 = tf.Variable(tf.random_normal([1024]))
        fc1 = tf.add(tf.matmul(fc1, w1), b1)
        fc1 = tf.nn.softmax(fc1)

        # Output layer
        w2 = tf.Variable(tf.random_normal([1024, num_classes]))
        b2 = tf.Variable(tf.random_normal([num_classes]))
        out = tf.add(tf.matmul(fc1, w2), b2)

        # Check devices for good measure
        print("w1 is placed on device : ",w1.device)
        print("b1 is placed on device : ",b1.device)
        print("w2 is placed on device : ",w2.device)
        print("b2 is placed on device : ",b2.device)

    return out

def main():

    # device names and ports
    IP_ADDRESS1 = '198.168.1.136'
    PORT1 = '2222'
    IP_ADDRESS2 = '198.168.1.121'
    PORT2 = '2224'

    # Define devices that we wish to split our graph over
    device0 = '/job:worker/task:0'
    device1 = '/job:worker/task:1'
    devices = (device0, device1)

    # reset graph
    tf.reset_default_graph()

    # Construct model
    with tf.device(devices[0]):

        X = tf.placeholder(tf.float32, [None, num_input])                                              #Input images feedable
        Y = tf.placeholder(tf.float32, [None, num_classes])

    logits = CNN(X, devices)                                                                           #Unscaled probabilities

    with tf.device(devices[1]):
        # following function gives probability for classes
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()


    # Set up cluster

    cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

    task_idx=0 # We have chosen this machine to be our chief (The first IPaddress:Port combo), so task_idx=0
    server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

    # Check the server definition
    server.server_def

    #Start training
    #IMPORTANT: Pass the server target to the session definition

    with tf.Session(server.target) as sess:

        # Run the initializer
        sess.run(init)

        for step in range(100):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
                    loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

        # Get test set accuracy
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256]}))

if __name__=='__main__':

    main()