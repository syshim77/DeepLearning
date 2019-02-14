#### Lecture12
## RNN

###### Theory
1. Sequence data
  - We don't understand one word only
  - We understand based on the previous words + this word(time series)
  - NN/CNN cannot do this
  - *[for more information](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*
2. Recurrent Neural Network
  - We can process a sequence of vectors x by applying a recurrence formula at every time step:
    + Ht = Fw(Ht-1,Xt)
    + Ht: new state, Fw: some function with parameters W, Ht-1: old state, Xt: input vector at some time step
  - notice: the same function and the same set of parameters are used at every time step
3. (Vanilla) Recurrent Neural Network
  - the state consists of a single "hidden" vector h:
    + Ht = Fw(Ht-1,Xt)
    + Ht = tanh(Whh*Ht-1 + fxh*Xt)
    + Yt = Why*Ht
  - weight 전체 똑같은 값을 사용함
4. character-level language model example
  - vocabulary: [h,e,l,o]
  - example training sequence: "hello"
  - h 입력했을때 다음에 올 단어를 예측
  - input chars -> input layer(vector): use one-hot
    + h: [1 0 0 0], e: [0 1 0 0], l: [0 0 1 0], o: [0 0 0 1]
  - Ht = tanh(Whh*Ht-1 + fxh*Xt)
    + input chars -> input layer -> hidden layer(input layer * W_xh)
    + h -> [1 0 0 0] -> [0.3 -0.1 0.9]
    + e -> [0 1 0 0] -> [1.0 0.3 0.1] (= [0.3 -0.1 0.9] * W_hh + [0 1 0 0] * W_xh)
  - Yt = Why*Ht
    + hidden layer -> output layer(hidden layer * W_hy) -> target chars
    + output layer에서 max값 찾으면 그 위치에 해당하는 글자가 예측하는 글자가 됨
5. [RNN applications](https://github.com/TensorFlowKR/awesome_tensorflow_implementations)
  - **Language Modeling**
  - Speech Recognition
  - Machine Translation
  - Conversation modeling/question Answering
  - Image/Video Captioning
  - Image/Music/Dance Generation
  - *[for more information](http://jiwonkim.org/awesome-rnn/)*
6. Recurrent Networks offer a lot of flexibility
  - one to one
    + e.g. Vanilla Neural Networks
  - one to many
    + e.g. image captioning(image -> sequence of words)
  - many to one
    + e.g. sentiment classification(sequence of words -> sentiment)
  - many to many
    + e.g. machine translation(seq of words -> seq of words)
  - many to many
    + e.g. video classification on frame level
7. Multi-layer RNN
  - 더 복잡한 학습 가능
8. Training RNNs is challenging
  - several advanced models
    + **long short term memory(LSTM)**
    + GRU by Cho et al. 2014

###### Laboratory
1. RNN in TensorFlow
  ```python
  cell = tf.contrib.nn.BasicRNNCell(num_units=hidden_size)  # num_units: output size, 잘 사용하지 않음
  cell = tf.contrib.nn.BasicLSTMCell(num_units=hidden_size) # cell만 바꾸면 나머지는 그대로 사용 가능
  ...
  outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
  ```
  - 1) One node: 4 (input_dim) in 2 (hidden_size)
  ```python
  # one hot encoding
  h = [1,0,0,0]
  e = [0,1,0,0]
  l = [0,0,1,0]
  o = [0,0,0,1]

  # input(Xt):
  # [[[1,0,0,0]]]
  shape = (1,1,4)

  hidden_size = 2

  # output(Ht):
  # [[[x,x]]]
  shape = (1,1,2) # output dimension은 hidden_size에 따라 달라짐

  # one cell RNN input_dim(4) -> output_dim(2)
  hidden_size = 2
  cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

  x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
  outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

  sess.run(tf.global_variables_initializer())
  pp.pprint(outputs.eval())
  ```
  ```
  result:
  array([[[-0.42409304, 0.64651132]]])
  ```
  - 2) Unfolding to n sequences
  ```python
  # hidden_size = 2, sequence_length = 5
  # shape = (1,5,2): [[[x,x], [x,x], [x,x], [x,x], [x,x]]]
  # shape = (1,5,4): [[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]] # hello

  # one cell RNN input_dim(4) -> output_dim(2). sequence: 5
  hidden_size = 2
  cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
  x_data = np.array([[h,e,l,l,o]], dtype=np.float32)
  print(x_data.shape)
  pp.pprint(x_data)
  outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32))
  sess.run(tf.global_variables_initializer())
  pp.pprint(outputs.eval())
  ```
  - 3) Bathing input
  ```python
  # hidden_size = 2, sequence_length = 5, batch = 3
  # shape = (3,5,2): [x,x]가 5개짜리 1줄이 3줄
  # shape = (3,5,4): [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]] 와 같은 1줄이 3줄(hello, eolll, lleel)

  # one cell RNN input_dim(4) -> output_dim(2). sequence: 5, batch: 3
  # 3 batches 'hello', 'eolll', 'lleel'
  x_data = np.array([[h,e,l,l,o], [e,o,l,l,l], [l,l,e,e,l]], dtype=np.float32)
  pp.pprint(x_data)

  cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
  outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
  sess.run(tf.global_variables_initializer())
  pp.pprint(outputs.eval())
  ```
  - 4) Cost: sequence_loss
  ```python
  # [batch_size, sequence_length]
  y_data = tf.constant([[1,1,1]])

  # [batch_size, sequence_length, emb_dim]
  prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

  # [batch_size * sequence_length]
  weights = tf.constant([[1,1,1]], dtype=tf.float32)

  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
  sess.run(tf.global_variables_initializer())
  print("Loss: ", sequence_loss.eval())
  ```
  ```
  result:
  Loss: 0.596759
  ```
  ```python
  prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7],[0.3, 0.7]]], dtype=tf.float32)
  prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

  weights = tf.constant([[1,1,1]], dtype=tf.float32)

  sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
  sequense_loss2(prediction2)

  sess.run()
  print("Loss1: ", sequence_loss1.eval(), "Loss: ",sequence_loss2.eval())
  ```
  ```
  result:
  Loss1: 0.513015
  Loss2: 0.371101
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-0-rnn_basics.ipynb)*
2. Teach RNN 'hihello'
  - 일반적인 forward net으로는 힘듬(어떨땐 h 다음에 예측값이 i이고, 어떨땐 예측값이 e이어야 함)
  ```
  text: 'hihello'
  unique chars(vocabulary, voc): h,i,e,l,o
  voc index(dic): h 0, i 1, e 2, l 3, o 4
  one-hot encoding
  [1,0,0,0,0] # h 0
  [0,1,0,0,0] # i 1
  [0,0,1,0,0] # e 2
  [0,0,0,1,0] # l 3
  [0,0,0,0,1] # o 4
  ```
  - 1) Creating RNN cell
  ```python
  # RNN model
  rnn_cell = rnn_cell.BasicRNNCell(rnn_size)

  rnn_cell = rnn_cell.BasicLSTMCell(rnn_size)
  rnn_cell = rnn_cell.GRUCell(rnn_size)
  ```
  - 2) Execute RNN
  ```python
  # rnn model
  rnn_cell = rnn_cell.BasicRNNCell(rnn_size)
  outputs, _states = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=initial_state, dtype=tf.float32)
  ```
  - 3) RNN parameters
  ```python
  hidden_size = 5 # output from the LSTM
  input_dim = 5 # one-hot size
  batch_size = 1  # one sentence
  sequence_length = 6 # |ihello| == 6
  ```
  - 4) Data creation
  ```python
  idx2char = ['h', 'i', 'e', 'l', 'o']  # h=0, i=1, e=2, l=3, o=4
  x_data = [[0,1,0,2,3,3]]  # hihell
  x_one_hot = [[[1,0,0,0,0,0] # h 0
              , [0,1,0,0,0,0] # i 1
              , [1,0,0,0,0,0] # h 0
              , [0,0,1,0,0,0] # e 2
              , [0,0,0,1,0,0] # l 3
              , [0,0,0,1,0,0]]] # l 3

  y_data = [[1,0,2,3,3,4]]  # ihello

  X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y Label
  ```
  - 5) Feed to RNN
  ```python
  X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y Label

  cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
  ```
  - 6) cost: sequence_loss
  ```python
  outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
  weights = tf.ones([batch_size, sequence_length])

  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights) # 원래는 바로 output을 넣으면 안됨, 일단은 이렇게 하는 것
  loss = tf.reduce_mean(sequence_loss)
  train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
  ```
  - 7) Training
  ```python
  prediction = tf.argmax(outputs, axis=2)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
      l, _ = sess.run([loss,train], feed_dict={X: x_one_hot, Y: y_data})
      result = sess.run(prediction, feed_dict={X: x_one_hot})
      print(i, "loss: ", l, "prediction: ", result, "true Y: ", y_data)

      # print char using dic
      result_str = [idx2char[c] for c in np.squeeze(result)]
      print("\tPrediction str: ", ''.join(result_str))
  ```
  ```
  results:
  0 loss: 1.55474 prediction: [[3 3 3 3 4 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: lllloo
  1 loss: 1.55081 prediction: [[3 3 3 3 4 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: lllloo
  2 loss: 1.54704 prediction: [[3 3 3 3 4 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: lllloo
  3 loss: 1.54342 prediction: [[3 3 3 3 4 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: lllloo
  ...
  1998 loss: 0.75305 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: ihello
  1999 loss: 0.752973 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2,3,3,4]] Prediction str: ihello
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-1-hello-rnn.py)*
3. Manual data creation
  ```python
  idx2char = ['h', 'i', 'e', 'l', 'o']
  x_data = [[0,1,0,2,3,3]]  # hihell
  x_one_hot = [[[1,0,0,0,0,0] # h 0
              , [0,1,0,0,0,0] # i 1
              , [1,0,0,0,0,0] # h 0
              , [0,0,1,0,0,0] # e 2
              , [0,0,0,1,0,0] # l 3
              , [0,0,0,1,0,0]]] # l 3

  y_data = [[1,0,2,3,3,4]]  # ihello
  ```
4. Better data creation
  ```python
  sample = "if you want you"
  idx2char = list(set(sample))  # index -> char
  char2idx = {c: i for i, c in enumerate(idx2char)} # char -> idx

  sample_idx = [char2idx[c] for c in sample] # char to index
  x_data = [sample_idx[:-1]] # X data sample (0 ~ n-1) hello: hell
  y_data = [sample_idx[1:]] # Y Label sampel (1 ~ n) hello: ello

  X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

  X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0, num_classes: indexed char의 크기와 같음, shape 유의
  ```
  - 1) Hyper parameters
  ```python
  dic_size = len(char2idx)  # RNN input size(one hot size)
  rnn_hidden_size = len(char2idx) # RNN output size
  num_classes = len(char2idx) # final output size(RNN or softmax, etc...)
  batch_size = 1 # one sample data, one batch
  sequence_length = len(sample)-1 # number of LSTM unfolding (unit #)
  ```
  - 2) LSTM and Loss
  ```python
  X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

  X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

  cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

  weights = tf.ones([batch_size, sequence_length])
  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
  loss = tf.reduce_mean(sequence_loss)
  train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

  prediction = tf.argmax(outputs, axis=2)
  ```
  - 3) Training and results
  ```python
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
      l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
      result = sess.run(prediction, feed_dict={X: x_data})
      # print char using dic
      result_str = [idx2char[c] for c in np.squeeze(result)]
      print(i, "loss: ", l, "Prediction: ", ''.join(result_str))
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-2-char-seq-rnn.py)*
5. Really long sentence?
  ```
  sentence = ("if you want to build a ship, don't drum up people together to "
              "collect wood and don't assign them tasks and work, but rather "
              "teach them to long for the endless immensity of the sea.")

  # training dataset
  0 if you wan -> f you want
  1 f you want ->  you want
  2  you want -> you want t
  3 you want t -> ou want to
  ...
  168 of the se -> of the sea
  169 of the sea -> f the sea.
  ```
  - 1) Making dataset
  ```python
  char_set = list(set(sentence))
  char_dic = {W: i for i, w in enumerate(char_set)}

  dataX = []
  dataY = []

  for i in range3(0, len(sentence) - seq_length):
    x_str = sentence[i:i+seq_length]
    y_str = sentence[i+1:i+seq_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)
  ```
  - 2) RNN parameters
  ```python
  data_dim = len(char_set)
  hidden_size = len(char_set)
  num_classes = len(char_set)
  seq_length = 10 # Any arbitrary number

  batch_size = len(dataX)
  ```
  - 3) LSTM and loss
  ```python
  X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

  X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

  cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
  initial_state = cell.zero_state(batch_size, tf.float32)
  outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

  weights = tf.ones([batch_size, sequence_length])
  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
  loss = tf.reduce_mean(sequence_loss)
  train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

  prediction = tf.argmax(outputs, axis=2)
  ```
  - 4) Results
    + Exercise
      * run long sequence RNN
      * why it does not work? (logit이 매끄럽지 않고 RNN이 깊지 않음)
6. Wide & Deep
  - *[for more information](https://www.tensorflow.org/versions/r0.11/tutorials/wide_and_deep/index.html)*
7. Stacked RNN
  ```python
  X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
  Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

  # one -hot encoding
  X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
  print(X_one_hot)

  # make a LSTM cell with hidden_size(each unit output vector size)
  # cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
  cell = rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

  # outputs: unfolding size x hidden size, state = hidden size
  outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
  ```
  - 1) Softmax(FC) in Deep CNN
    + softmax
  ```python
  outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])  # output: softmax output

  X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

  # (optional) softmax layer
  X_for_softmax = tf.reshape(outputs, [-1, hidden_size])  # RNN output

  softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])  # input size, output size

  softmax_b = tf.get_variable("softmax_b", [num_classes])
  outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

  outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])  # RNN output shape same
  ```
  - 2) Loss
  ```python
  # reshape out for sequence_loss
  outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])  # reshape한 이 output을 넘겨줘야함

  # All weights are 1 (equal weights)
  weights = tf.ones([batch_size, seq_length])

  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
  loss = tf.reduce_mean(sequence_loss)
  train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

  prediction = tf.argmax(outputs, axis=2)
  ```
  - 3) Training and print results
  ```python
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for i in range(500):
    _ , l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})

    for j, result in enumerate(results):
      index = np.argmax(result, axis=1)
      print(i, j, ''.join([char_set[t] for t in index]), l)

  # Let's print the last char of each result to check it works
  results = sess.run(outputs, feed_dict={X: dataX})
  for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
      print(''.join([char_set[t] for t in index]), end='')
    else:
      print(char_set[index[-1]], end='')
  ```
  ```
  result:
  f you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-4-rnn_long_char.py)*
8. char-RNN
  - Shakespeare, linux source code 등 학습해서 새로운 문구나 코드를 만들 수 있음
9. char/word RNN (char/word level n to n model)
  - *[for more information_1](https://github.com/sherjilozair/char-rnn-tensorflow)*
  - *[for more information_2](https://github.com/hunkim/word-rnn-tensorflow)*
10. Different sequence length
  ```python
  sequence_length = [5,2,3] # hello, hi, why
  ```
11. Dynamic RNN
  ```python
  # 3 batches 'hello', 'eolll', 'lleel'
  x_data = np.array([[[...]]], dtype=np.float32)

  hidden_size = 2
  cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=true)

  outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5,3,4], dtype=tf.float32)

  sess.run(tf.global_variables_initializer())
  pp.pprint(outputs.eval())
  ```
  - result에서 확실하게 없는 데이터의 값은 0으로 만들어줘서 loss가 잘 동작하도록 만듬
12. Time series data
  - 시간이 지나면서 값이 변하는 데이터
  - ex) 주식시장
13. Many to one
  - 하루 전의 데이터만 가지고는 예측할 수 없고 이전의 데이터가 예측값에 영향을 미친다
  - 1) Reading data
  ```python
  timesteps = seq_length = 7
  data_dim = 5
  output_dim = 1

  # open, high. low, close, volume
  xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
  xy = xy[::-1] # reverse order(chronically ordered)
  xy = MinMaxScaler(xy)
  x = xy
  y = xy[:,[-1]]  # close as label

  dataX=[]
  dataY=[]
  for i in range(0, len(y)-sequence_length):
    _x = x[i:i+sequence_length]
    _y = y[i+seq_length] # next close price
    print(_ x, "->", y)
    dataX.append(_x)
    dataY.append(_y)
  ```
  - 2) Training and test datasets
  ```python
  # split to train and testing
  train_size = int(len(dataY)*0.7)
  test_size = len(dataY) - train_size
  trainX, textX = np.array(dataX[0: train_size]), np.array(dataX[train_size: len(dataX)])
  trainY, textY = np.array(dataY[0: train_size]), np.array(dataY[train_size: len(dataY)])

  # input placeholders
  X = tf.placeholder(tf.float32, [None, seq_length, data_dim])]
  Y = tf.placeholder(tf.float32, [None, 1])
  ```
  - 3) LSTM and loss
  ```python
  # input placeholders
  X = tf.placeholder(tf.int32, [None, seq_length, data_dim])
  Y = tf.placeholder(tf.int32, [None, 1])

  cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
  outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
  Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)  # we use the last cell's output, fully-connected 한 번 거쳐서 y로 간다

  # cost/loss
  loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
  # optimizer
  optimizer = tf.train.AdamOptimizer(0.01)
  train = optimizer.miimize(loss)
  ```
  - 4) Training and results
  ```python
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    print(i,l)

  testPredict = sess.run(Y_pred, feed_dict={X: testX})

  import matplotlib.pyplot as plt
  plt.plot(testY)
  plt.plot(testPredict)
  plt.show()
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py)*
14. Exerciese
  - implement stock prediction using linear regression only
    + linear vs LSTM
  - improve results using more features such as keywords and/or sentiments in top news
