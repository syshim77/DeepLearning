#### Lecture09
## Neural Nets(NN) for XOR

###### Theory
1. One logistic regression unit cannot separate XOR
  - Neural Nets first depression
2. Multiple logistic regression units
  - "No one on earth had found a viable way to train" by Marvin Minsky
3. XOR using NN
  - linear x
  - Neural Net
    + 임의의 가정을 이용한 증명을 통해 NN으로 XOR 계산 가능함을 알 수 있음
  - Forward propagation
    + Can you find another W and b for the XOR?
      * 숙제
  - NN
    + Recap: Lec6-1 Multinomial Classification
    + K(X) = sigmoid(XW1+B1), Y의 hat = H(X) = sigmoid(K(X)W2+b2)
    ```python
    # NN
    K = tf.sigmoid(tf.matmul(X,W1)+b1)  # K(X) = sigmoid(XW1+B1)
    hypothesis = tf.sigmoid(tf.matmul(K,W2)+b2) # H(X) = sigmoid(K(X)W2+b2)
    ```
    + How can we learn W1,W2,b1,b2 from trading data?
      * to be continued in number 6
4. Basic derivative
  - 미분(derivative): 순간 변화율, 그 지점에서의 기울기
  - *[for more information](https://ko.wikipedia.org/wiki/%EB%AF%B8%EB%B6%84)*
5. Partial derivative: consider other variables as constants
  - 편미분(partial derivative): 해당 변수에 대해서만 미분하고 다른 변수는 상수 취급
  - *[for more information](https://ko.wikipedia.org/wiki/%EB%AF%B8%EB%B6%84)*
6. How can we learn W1,W2,B1,b2 from trading data?
  - Derivation
  - Perceptrons(1969)
    + We need to use MLP, multiplayer perceptrons(multilayer neural nets)
    + No one on earth had found a viable way to train MLPs good enough to learn such simple functions
  - Backpropagation(chain rule)
    + Back propagation 이용하면 미분값을 쉽게 구할 수 있다
    + chain rule: f(g(x)) partial derivative -> f/g*g/x
    ```
    f = wx+b, g = wx 이라고 하면 f = g+b

    1) forward(w=-2,x=5,b=3)
    2) backward(use partial derivative)
    ```
    + partial derivation mean: f의 출력에 미치는 영향(비율)
    + *[for more information](http://cs231n.stanford.edu/)*
    + ex) sigmoid
      * g(z)=1/(1+e^-2)
      * make function to graph: z -> * (-1) -> exp -> + 1 -> 1/x -> g
      * 위의 그래프를 뒤에서부터(<- 방향) 편미분으로 계산해나가면 미분값 구할 수 있음
    + back propagation in TensorFlow
      * 각각을 다 그래프로 만들어서 구현
      * TensorBoard

###### Laboratory
1. XOR data set
  - Boolean expression(X=A⊕B), Logic diagram symbol, Truth table(진리표)
  ```python
  x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
  y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
  ```
  - XOR with logistic regression?
    ```
    result:
    Hypothesis:
    [[0.5]
    [0.5]
    [0.5]
    [0.5]]
    Correct:
    [[0.]
    [0.]
    [0.]
    [0.]]
    Accuracy: 0.5
    ```
    + *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-1-xor.py)*
    + But it doesn't work!
  - Neural Net(NN for XOR)
    ```python
    W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

    W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)
    ```
    ```
    2 layers result:
    Hypothesis:
    [[0.01338218]
    [0.98166394]
    [0.98809403]
    [0.01135799]]
    Correct:
    [[0.]
    [1.]
    [1.]
    [0.]]
    Accuracy: 1.0
    ```
    + *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-2-xor-nn.py)*
  - Wide NN for XOR
    ```python
    W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

    W2 = tf.Variable(tf.random_normal([10,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)
    ```
    ```
    wider 2 layers result:
    Hypothesis:
    [[0.00358802]
    [0.99366933]
    [0.99204296]
    [0.0095663]]
    Correct:
    [[0.]
    [1.]
    [1.]
    [0.]]
    Accuracy: 1.0
    ```
    + cost가 더 작아짐 = 더 잘 학습되었다
  - Deep NN for XOR
    ```python
    W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
    b1 = tf.Variable(tf.random_normal([10]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

    W2 = tf.Variable(tf.random_normal([10,10]), name='weight2')
    b2 = tf.Variable(tf.random_normal([10]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1,W1)+b1)

    W3 = tf.Variable(tf.random_normal([10,10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2,W1)+b1)

    W4 = tf.Variable(tf.random_normal([10,1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3,W2)+b2)
    ```
    ```
    4 layers result:
    Hypothesis:
    [[8.90e-04]
    [9.99e-01]
    [9.98e-01]
    [1.55e-03]]
    Correct:
    [[0.]
    [1.]
    [1.]
    [0.]]
    Accuracy: 1.0
    ```
    + cost가 더 작아짐 = 더 잘 학습되었다
2. Exercise
  - Wide and Deep NN for MNIST
3. TensorBoard: TF logging/debugging tool
  - Visualize your TF graph
  - Plot quantitative metrics
  - Show additional data
  - *[for more information](https://www.tensorflow.org/get_started/summaries_and_tensorboard)*
4. 5 steps of using TensorBoard
  ```python
  # 1) From TF graph, decide which tensors you want to log
  w2_hist = tf.summary.histogram("weights2", w2)
  cost_summ = tf.summary.scalar("cost", cost)

  # 2) Merge all summaries
  summary = tf.summary.merge_all()

  # 3) Create writer and add graph
  # Create summary writer
  writer = tf.summary.FileWriter('.logs')
  writer.add_graph(sess.graph)

  # 4) Run summary merge and add_summary
  s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
  writer.add_summary(s, global_step=global_step)

  # 5) Launch TensorBoard
  tensorboard --logdir = ./logs
  ```
  - 1) From TF graph, decide which tensors you want to log
    + scalar tensors
    ```python
    cost_summ = tf.summary.scalar("cost", cost)
    ```
    + histogram(multi-dimensional tensors)
    ```python
    W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)

    w2_hist = tf.summary.histogram("weights2", w2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
    ```
    + add scope for better graph hierarchy
    ```python
    with tf.name_scope("layer1") as scope:
      W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
      b1 = tf.Variable(tf.random_normal([2]), name='bias1')
      layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

      w1_hist = tf.summary.histogram("weights1", w1)
      b1_hist = tf.summary.histogram("biases1", b1)
      layer1_hist = tf.summary.histogram("layer1", layer1)


    with tf.name_scope("layer2") as scope:
      W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
      b2 = tf.Variable(tf.random_normal([1]), name='bias2')
      hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)

      w2_hist = tf.summary.histogram("weights2", w2)
      b2_hist = tf.summary.histogram("biases2", b2)
      hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
    ```
  - 2) Merge summaries and create writer
  - 3) After creating session
    ```python
    # Summary
    summary = tf.summary.merge_all()

    # Initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create summary writer
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)  # Add graph in the TensorBoard
    ```
  - 4) Run merged summary and write (add summary)
    ```python
    s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
    writer.add_summary(s, global_step=global_step)
    global_step += 1
    ```
  - 5) Launch TensorBoard
    + local
    ```python
    writer = tf.summary.FileWriter("./logs/xor_logs")
    ```
    ```
    $ tensorboard -logdir = ./logs/xor_logs

    Starting TensorBoard b'41' on port 6006
    (You can navigate to http://127.0.0.1:6006)
    ```
    + remote server
      * port forwarding 이라는 기법 사용
    ```
    ssh -L local_port:127.0.0.1:remote_port username@server.com

    local> $ ssh -L 7007:121.0.0.0:6006 hunkim@server.com
    server> $ tensorboard -logdir = ./logs/xor_logs
    (You can navigate to http://127.0.0.1:7007)
    ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-09-4-xor_tensorboard.py)*
5. Multiple runs
  - learning_rate=0.1 vs learning_rate=0.01
    ```
    tensorboard -logdir = ./logs/xor_logs
      train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
      ...
      writer = tf.summary.FileWriter("./logs/xor_logs")

    tensorboard -logdir = ./logs/xor_logs_r0_01
      train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
      ...
      writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")

    tensorboard -logdir = ./logs
    ```
    + 여러개의 하위 dir을 만들어서 상위 dir을 실행시킴
6. Exercise
  - Wide and Deep NN for MNIST
  - **Add TensorBoard**
