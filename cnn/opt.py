def _loss(logits,labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    loss_op = tf.reduce_mean(cross_entropy, name='xentropy')
    return loss_op

def _minimize(loss, learning_rate):
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def _evaluate(logits, labels):
    predicted = tf.argmax(tf.nn.softmax(logits))
    expected = tf.argmax(labels)
    correct_pred = tf.equal(predicted, expected)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def _eval(sess, accuracy, x, y_true, data_set, eval_log):
    true_count = 0
    steps_per_epoch = data_set.num_examples // 32
    num_examples = steps_per_epoch * 32
    for step in range(steps_per_epoch):
        x_valid_batch, valid_cls_batch, y_valid_batch = data_set.next_batch(32)
        x_valid_batch = x_valid_batch.reshape(32, img_size_flat)
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}
        true_count += sess.run(accuracy, feed_dict=feed_dict_validate)
    precision = true_count / num_examples
    result = ('  Num examples: %d  Num correct: %d  Precision : %0.04f\n' %
        (num_examples, true_count, precision))
    eval_log.write(result)