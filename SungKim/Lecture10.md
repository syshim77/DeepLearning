#### Lecture10
## ReLU: Better non-linearity

###### Theory
1. NN for XOR
  - activation function(각 unit에 있는 sigmoid functions)
    + network이 서로 연결 되어있어서 어느 값 이상이면 active, 이하면 non-active
  ```python
  W1 = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0))
  W2 = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))

  b1 = tf.Variable(tf.zeros([2]), name="Bias1")
  b2 = tf.Variable(tf.zeros([1]), name="Bias2")

  # Our hypothesis
  L2 = tf.sigmoid(tf.matmul(X,W1)+b1)
  hypothesis = tf.sigmoid(tf.matmul(L2,W2)+b2)
  ```
2. Let's go deep & wide!
  ```python
  W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0))
  W2 = tf.Variable(tf.random_uniform([5,4], -1.0, 1.0))
  W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0))

  b1 = tf.Variable(tf.zeros([5]), name="Bias1")
  b2 = tf.Variable(tf.zeros([4]), name="Bias2")
  b3 = tf.Variable(tf.zeros([1]), name="Bias3")

  # Our hypothesis
  L2 = tf.sigmoid(tf.matmul(X,W1)+b1)
  L3 = tf.sigmoid(tf.matmul(L2,W2)+b2)
  hypothesis = tf.sigmoid(tf.matmul(L3,W3)+b3)
  ```
  - 1st: input layer, 2nd: hidden layer, 3rd: output layer
3. 9 hidden layers!
  ```python
  W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0), name="Weight1")
  W2 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight2")
  W3 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight3")
  W4 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight4")
  W5 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight5")
  W6 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight6")
  W7 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight7")
  W8 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight8")
  W9 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight9")
  W10 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name="Weight10")

  W11 = tf.Variable(tf.random_uniform([5,1], -1.0, 1.0), name="Weight11")

  b1 = tf.Variable(tf.zeros([5]), name="Bias1")
  b2 = tf.Variable(tf.zeros([5]), name="Bias2")
  b3 = tf.Variable(tf.zeros([5]), name="Bias3")
  b4 = tf.Variable(tf.zeros([5]), name="Bias4")
  b5 = tf.Variable(tf.zeros([5]), name="Bias5")
  b6 = tf.Variable(tf.zeros([5]), name="Bias6")
  b7 = tf.Variable(tf.zeros([5]), name="Bias7")
  b8 = tf.Variable(tf.zeros([5]), name="Bias8")
  b9 = tf.Variable(tf.zeros([5]), name="Bias9")
  b10 = tf.Variable(tf.zeros([5]), name="Bias10")

  b11 = tf.Variable(tf.zeros([1]), name="Bias11")

  # TensorBoard
  # Our hypothesis
  with tf.name_scope("layer1") as scope:
    L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
  with tf.name_scope("layer12") as scope:
    L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)
  with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2,W3)+b3)
  with tf.name_scope("layer4") as scope:
    L4 = tf.sigmoid(tf.matmul(L3,W4)+b4)
  with tf.name_scope("layer5") as scope:
    L5 = tf.sigmoid(tf.matmul(L4,W5)+b5)
  with tf.name_scope("layer6") as scope:
    L6 = tf.sigmoid(tf.matmul(L5,W6)+b6)
  with tf.name_scope("layer7") as scope:
    L7 = tf.sigmoid(tf.matmul(L6,W7)+b7)
  with tf.name_scope("layer8") as scope:
    L8 = tf.sigmoid(tf.matmul(L7,W8)+b8)
  with tf.name_scope("layer9") as scope:
    L9 = tf.sigmoid(tf.matmul(L8,W9)+b9)
  with tf.name_scope("layer10") as scope:
    L10 = tf.sigmoid(tf.matmul(L9,W10)+b10)

  with tf.name_scope("last") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10,W11)+b11)
  ```
4. TensorBoard visualization
  - 확인하기 편함
5. Poor results?
  - cost 떨어지지 않고 accuracy 0.5
  - TensorBoard에서 cost & accuracy 그래프로 한눈에 확인 가능
  - backpropagation(1986)
    + 2 layers, 3 layers는 잘됨
    + But 9 layers, 10 layers 넘어가면서 학습이 안됨
    + lec 9-2: Backpropagation(chain rule)
  - vanishing gradient(NN winter2: 1986-2006)
    + 경사 기울기가 사라지는 문제
    + 기울기가 사라진다 = 학습하기가 굉장히 어렵다 = 예측이 어렵다
6. Geoffrey Hinton's Summary of findings up to today
  - our labeled datasets were thousands of times too small
  - our computers were millions of times too slow
  - we initialized the weights in a stupid way
  - **we used the wrong type of non-linearity**
    + sigmoid!
      * 입력값이 항상 1보다 작은 값
      * 1보다 작은 값들을 자꾸 곱해나가니까 최종값은 너무 작은 값이다
    + 그래서 만들어진게 ReLU
      * 0보다 작은 값: 꺼버림, 0보다 큰 값: linear하게 갈때까지 감
7. ReLU: Rectified Linear Unit
  - sigmoid 대신 집어넣으면 됨
    + sigmoid: L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
    * ReLU: L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
8. ReLu
  ```python
  # Our hypothesis
  with tf.name_scope("layer1") as scope:
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
  with tf.name_scope("layer2") as scope:
    L1 = tf.nn.relu(tf.matmul(L1,W2)+b2)
  with tf.name_scope("layer3") as scope:
    L1 = tf.nn.relu(tf.matmul(L2,W3)+b3)
  with tf.name_scope("layer4") as scope:
    L1 = tf.nn.relu(tf.matmul(L3,W4)+b4)
  with tf.name_scope("layer5") as scope:
    L1 = tf.nn.relu(tf.matmul(L4,W5)+b5)
  with tf.name_scope("layer6") as scope:
    L1 = tf.nn.relu(tf.matmul(L5,W6)+b6)
  with tf.name_scope("layer7") as scope:
    L1 = tf.nn.relu(tf.matmul(L6,W7)+b7)
  with tf.name_scope("layer8") as scope:
    L1 = tf.nn.relu(tf.matmul(L7,W8)+b8)
  with tf.name_scope("layer9") as scope:
    L1 = tf.nn.relu(tf.matmul(L8,W9)+b9)
  with tf.name_scope("layer10") as scope:
    L1 = tf.nn.relu(tf.matmul(L9,W10)+b10)

  # 마지막의 출력은 0과 1 사이어야하므로 sigmoid 사용
  with tf.name_scope("last") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10,W11)+b11)
  ```
  - works very well
    + accuracy 1.0, cost 거의 없음
9. Activation functions
  - sigmoid
    + σ(x) = 1/(1+e^-x)
  - tanh = tanh(x)
    + sigmoid를 0을 중심으로 내려서 1, -1 사이 값을 갖게 됨
  - ReLU = max(0,x)
  - Leaky ReLU = max(0.1x, x)
  - Maxout = mas(w1^T+b1, w2^T+b2)
  - ELU
    + f(x) = x (if x>0), α(exp(x)-1) (if x≤0)
10. Geoffrey Hinton's summary of finding s up to today
  - our labeled datasets were thousands of times too small
  - our computers were millions of times too slow
  - **we initialized the weights in a stupid way**
    + weight을 좋은 값으로 설정하는 것이 중요
  - we used the wrong type of non-linearity
11. Set all initial weights to 0
  - 어떤 문제가 발생할까?
    + W가 chain rule에서 사용되는데 W=0이면 기울기가 0이 됨
    + gradient 사라지는 문제
12. Need to set the initial weight values wisely
  - not all 0's
  - challenging issue
  - Hinton et al. (2006) "A Fast Learning Algorithm for Deep Belief Nets"
    + **Restricted Boatman Machine(RBM)**
13. RBM structure
  - restriction = no connections within a layer
  - recreate input
    + forward, backward(= encoder, decoder)
    + KL divergence = compare actual to recreation
14. How can we use RBM to initialize weights?
  - Apply the RBM idea on adjacent two layers as a pre-training step
  - Continue the first process to all layers
  - This will set weights
  - example: Deep Belief Network
    + weight initialized by RBM
    + pre-training, fine tuning(실제 labels을 가지고 학습을 시킴)
15. Good news
  - No need to use complicated RBM for weight initializations
  - Simple methods are ok
    + Xavier initialization
    + He's initialization
16. Xavier/He initialization
  - makes sure the weights are 'just right', not too small, not too big
  - using number of input (fan_in) and output (fan_out)
  ```python
  # Xavier initialization
  # Glorot et al. 2010
  W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)

  # He et al. 2015
  W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2)
  ```
17. prettytensor implementation
  - *[for more information](http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow)*
18. Still an active area of research
  - We don't know how to initialize perfect weight values, yet
  - Many new algorithms
    + batch normalization
    + layer sequential uniform variance
19. Overfitting
  - fit: linear하게 나눌 수 있는 것, overfit: W에 심한 왜곡을 주는 것
20. Am i Overfitting?
  - very high accuracy on the training dataset(e.g. 0.99)
  - poor accuracy on the test data set(0.85)
  - *[for more information](http://cs224d.stanford.edu/syllabus.html)*
21. Solutions for Overfitting
  - more training data!
  - **regularization**
22. Regularization
  - Let's not have too big numbers in the weight
    + cost + λΣW^2
    + l2reg = 0.001*tf.reduce_sum(tf.square(W))
23. Dropout: a simple way to prevent neural networks from overfitting [Srivastava et al. 2014]
  - 학습할 때 인위적으로 연결을 몇 개 끊어서 몇몇의 노드들을 없애버리자
  - regularization: dropout
    + randomly set some neurons to zero in the forward pass
    + how could this possibly be a good idea?
      * forces the network to have a redundant representation
      * 랜덤하게 쉬게해서 나머지 애들만 가지고 훈련시키는 것
      * 마지막에 총동원해서 예측하면 더 잘될수도 있다
  - TensorFlow implementation is as below
  ```python
  dropout_rate = tf.placeholder("float")
  _L1 = tf.nn.relu(tf.add(tf.matmul(X,W1)+B1))
  L1 = tf.nn.dropout(_L1, dropout_rate)

  # dropout_rate is important
  TRAIN:
    sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
  EVALUATION:
    print "Accuracy: ", accuracy.eval({X: mnidt.test.images, Y: mnist.test.labels, dropout_rate: 1})
  ```
24. What is ensemble?
  - data set -> training set -> learning model -> combiner -> ensemble prediction
  - *[for more information](http://www.slideshare.net/sasasiapacific/ipb-improving-the-models-predictive-power-with-ensemble-approaches)*
25. Feedforward neural network
  - 레고처럼 쌓아올리는 네트워크 구조
26. Fast forward
  - signal을 앞으로 밀어줘서 앞에꺼랑 더해져서 들어가게 하는 구조
  - He's initialization에서 사용한 구조
  - 오차율 3% 이하 네트워크 구조
27. split & merge
  - 2개로 나눠져서 가다가 원하면 만나고 또 갈라지고 하는 구조
  - 처음부터 입력을 여러개를 나누어서 받아서 각각을 처리하다가 다 모아서 하나로 나가고 하는 구조
28. Recurrent network
  - 앞으로만 쭉 나가는게 아니라 옆으로도 뻗어나가고 input 값도 받고 하는 구조
  - RNN
29. 'The only limit is your imagination'
  - *[for more information](http://itchyi.squarespace.com/thelatest/2012/5/17/the-only-limit-is-your-imagination.html)*

###### Laboratory
1. Softmax classifier for MNIST
  - lab 7-2: MNIST data
  - Accuracy: 0.9035
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-1-mnist_softmax.py)*
2. NN for MNIST
  ```python
  # input place holders
  X = tf.placeholder(tf.float32, [None,784])
  Y = tf.placeholder(tf.float32, [None,10])

  # weights & bias for nn layers
  W1 = tf.Variable(tf.random_normal([784,256])) # shape 유의
  b1 = tf.Variable(tf.random_normal([256]))
  L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

  W2 = tf.Variable(tf.random_normal([256,256])) # shape 유의
  b2 = tf.Variable(tf.random_normal([256]))
  L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

  W3 = tf.Variable(tf.random_normal([256,10])) # shape 유의
  b3 = tf.Variable(tf.random_normal([10]))
  hypothesis = tf.matmul(L2,W3)+b3

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  ```
  - Accuracy: 0.9455
  - *[full code]( https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-2-mnist_nn.py)*
3. Xavier for MNIST
  ```python
  # input place holders
  X = tf.placeholder(tf.float32, [None,784])
  Y = tf.placeholder(tf.float32, [None,10])

  # weights & bias for nn layers
  # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  W1 = tf.get_variable("W1", shape=[784,256], initializer = tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.random_normal([256]))
  L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

  W2 = tf.get_variable("W2", shape=[256,256], initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.random_normal([256]))
  L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

  W3 = tf.get_variable("W3", shape=[256,10], initializer = tf.contrib.layers.xavier_initializer())
  b3 = tf.Variable(tf.random_normal([10]))
  hypothesis = tf.matmul(L2,W3)+b3

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  ```
  - Accuracy: 0.9783
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-3-mnist_nn_xavier.py)*
4. Deep NN for MNIST
  ```python
  W1 = tf.get_variable("W1", shape=[784,512], initializer = tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.random_normal([512]))
  L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

  W2 = tf.get_variable("W2", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.random_normal([512]))
  L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

  W3 = tf.get_variable("W3", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
  b3 = tf.Variable(tf.random_normal([512]))
  L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)

  W4 = tf.get_variable("W4", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
  b4 = tf.Variable(tf.random_normal([512]))
  L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)

  W5 = tf.get_variable("W5", shape=[512,10], initializer = tf.contrib.layers.xavier_initializer())
  b5 = tf.Variable(tf.random_normal([10]))
  hypothesis = tf.matmul(L4,W5)+b5

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  ```
  - Accuracy: 0.9742
  - 여기서는 아마 overfitting이 일어나서 결과가 위에꺼보다 안좋은 것 같음
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-4-mnist_nn_deep.py)*
5. Dropout for MNIST
  ```python
  # dropout (keep_prob) rate 0.7 on training, but should be 1 for testing(테스트할 때는 모든 것을 총동원 해야하기 때문에 1)
  keep_prob = tf.placeholder(tf.float32)  # feed_dict를 통해 학습할 때랑 테스트할 때 값을 다르게 넘겨주게 됨

  W1 = tf.get_variable("W1", shape=[784,512])
  b1 = tf.Variable(tf.random_normal([512]))
  L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
  L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

  W2 = tf.get_variable("W2", shape=[512,512])
  b2 = tf.Variable(tf.random_normal([512]))
  L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
  L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
  ...
  # train my model
  for epoch in range(training_epochs):
    ...
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
      c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
      avg_cost += c/total_batch

  # test model and check accuracy
  correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
  ```
  - Accuracy: 0.9804
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-5-mnisdt_nn_dropout.py)*
6. Optimizers
  ```python
  train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
  ```
  - GradientDescentOptimizer 외에도 여러가지가 있음
  - 뭐가 좋은지 [여기](https://www.tensorflow.org/api_guides/python/train)에서 보고 하나씩 테스트해보는 것도 좋음
  - 뭐가 좋은지 [여기](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)에서 시뮬레이션해서 비교해놓음
  - 처음 사용하는 사람들은 ADAM을 먼저 쓰기를 권장
    + ADAM: a method for stochastic optimization [Kingma et al. 2015]
7. Use ADAM optimizer
  ```python
  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  ```
8. Summary
  - Softmax vs neural nets for MNIST: 90% and 94.5%
  - Xavier initialization: 97.8%
  - Deep neural nets with dropout: 98%
    + overfitting 발생해서 dropout으로 해결
  - Adam and other optimizers
  - Exercise: batch normalization
    + *[for more information]((https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-6-mnist_nn_batchnorm.ipynb)*
