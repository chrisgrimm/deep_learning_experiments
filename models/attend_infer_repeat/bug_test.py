import tensorflow as tf

sess =  tf.Session()
init_state = tf.zeros([32, 6])
init_state2 = tf.zeros([32, 6])
input = tf.placeholder(tf.float32, [32, 10])
input2 = tf.placeholder(tf.float32, [32, 10])
print init_state.get_shape()
rnn = tf.nn.rnn_cell.BasicRNNCell(3)
output, state = rnn(input, init_state)
output, state = rnn(input, state)
#output2, state2 = tf.nn.rnn_cell.BasicRNNCell(3)(input2, init_state2)
