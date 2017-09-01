# /usr/bin/env python
# -*- coding: utf-8 -*-
import hello_cnn.embed_factory as e_fac
import hello_cnn.vectorizer as vec
import hello_cnn.cnn as txt_cnn
import hello_cnn.label_factory as l_fac
import tensorflow as tf
import os
import time
import datetime
import pandas as pd
import numpy as np


def read_test_data(src, binarizer, vectorizer):
    df = pd.read_csv(src)
    x = np.array([text_vec for text_vec
                  in df.iloc[:, 2].map(vectorizer.vectorize)])
    y = binarize(binarizer, df.iloc[:, 1])
    return x, y


def binarize(binarizer, labels):
    """
    Returns:
        It can be feeded to the below placeholder.

        input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
    """
    return binarizer.transform(labels)


def train(FLAGS):
    vectorizer = vec.build_vectorizer(FLAGS.w2v_model)
    embed_factory = e_fac.EmbedFactory(vectorizer)
    label_binarizer = l_fac.create_label_binarizer(FLAGS.data, 1)
    x_test, y_test = read_test_data(
        FLAGS.test_data, label_binarizer, vectorizer)

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            # Allow device soft device placement
            # If you would like TensorFlow to automatically choose an existing
            # and supported device to run the operations
            # in case the specified one doesn't exist,
            # you can set allow_soft_placement
            allow_soft_placement=FLAGS.allow_soft_placement,
            # Log placement of ops on devices
            # To find out which devices your operations and
            # tensors are assigned to,
            # create the session with log_device_placement configuration
            # option set to True.
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = txt_cnn.Cnn(
                sequence_length=vectorizer.length,
                num_classes=label_binarizer.classes_.shape[0],
                embedding_size=vectorizer.embedding_size,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters)
            # Define Training procedure
            # global_step refer to the number of batches seen by the graph.
            # Everytime a batcnh is provided,
            # the weights are updated in the direction that minimizes the loss.
            # global_step just keeps track of the number of batches seen so far
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # minimize simply combine
            # calls compute_gradients() and apply_gradients()
            optimizer = tf.train.AdamOptimizer(1e-3)
            # List of (gradient, variable) pairs
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g)

                    # tf.nn.zero_fraction
                    # Returns the fraction of zeros in value.
                    # If value is empty, the result is nan.
                    # This is useful in summaries to
                    # measure and report sparsity.
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name),
                        tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(
                dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes
            # this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 0.5
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op,
                     cnn.loss, cnn.accuracy], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))

                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            # Training loop. For each batch...
            for x_batch, y_batch in embed_factory.create_epoch_batch_gen(
                    FLAGS.train_data, FLAGS.batch_size, FLAGS.num_epochs):

                train_step(x_batch, binarize(label_binarizer, y_batch))

                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_test, y_test, writer=dev_summary_writer)
                    print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        # checkpoint_prefix: save destination
                        path = saver.save(
                            sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(argv):

    dir = os.path.dirname(__file__) + '/../data'
    # Training parameters
    tf.flags.DEFINE_integer(
        "num_epochs", 200,
        "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer(
        "evaluate_every", 100,
        "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer(
        "num_checkpoints", 5,
        "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer(
        "checkpoint_every", 100,
        "Save model after this many steps (default: 100)")
    # Model Hyperparameters
    tf.flags.DEFINE_integer(
        "num_filters", 128,
        "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_string(
        "filter_sizes", "3,4,5",
        "Comma-separated filter sizes (default: '3,4,5')")
    # Misc Parameters
    tf.flags.DEFINE_boolean(
        "allow_soft_placement", True,
        "Allow device soft device placement")
    tf.flags.DEFINE_boolean(
        "log_device_placement", False,
        "Log placement of ops on devices")
    tf.flags.DEFINE_string(
        "train_data", f'{dir}/train.csv',
        "a csv file of training data")
    tf.flags.DEFINE_string(
        "test_data", f'{dir}/test.csv',
        "a csv file for test")
    tf.flags.DEFINE_string(
        "data", f'{dir}/data.csv',
        "a csv file of data")
    tf.flags.DEFINE_string(
        "w2v_model", f'{dir}/GoogleNews-vectors-negative300.bin',
        "The path of a Word2Vec model")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
