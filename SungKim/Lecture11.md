#### Lecture11
## CNN introduction

###### Theory
1. 'The only limit is your imagination'
  - forward = fully connected network
  - most of this lecture is from [this site](http://cs231n.stanford.edu/)
2. Convolutional neural networks
  - a bit of history:
    + Hubel & Wiesel, 1959 1962 1968...
    + 고양이 실험에서 시작됨
  - CONV(convolution) + ReLU + POOL => FC(Fully Connected)
3. Start with an image(width x hight x depth)
  - 32x32x3 image
4. Let's focus on a small area only
  - 5x5x3 filter
  - 마지막의 3은 항상 같은 값
5. Get one number using the filter
  - 한 점만 뽑아낸다(한 개의 값으로 만들어낸다) -> filter의 역할
  - Wx+b(= ReLU(Wx+b)) 사용
  - Y의 hat처럼 하나의 값으로 만들어냄
6. Let's look at other areas with the same filter(w)
  - filter를 옆으로 넘기면서 그림을 점차적으로 보듯이 각각의 그 지역에서 값을 가져옴
  - How many numbers can we get?
    + 이게 중요한 것
    + a closer look at spatial dimensions
      * 7x7 input(spatially) assume 3x3 filter => 5x5 output
      * stride의 크기가 1이다 = 1칸씩 움직인다, stride의 크기가 2이다 = 2칸씩 움직인다
    + applied with stride 2 => 3x3 output!
    + output size: (N-F)/stride + 1 (NxN size image, FxF size filter)
      * e.g. N=7, F=3: stride 1 => (7-3)/1 + 1 = 5, stride 2 => (7-3)/2 + 1 = 3, stride 3 => (7-3)/3 + 1 = 2.33...(할수없음)
    + in practice: common to zero pad the border
      * e.g. input 7x7, 3x3 filter, applied with stride 1, pad with 1 pixel border => what is the output?
      * 7x7 output
    + in general, common to see CONV layers with stride 1, filters of size FxF, and zero-padding with (F-1)/2.(will preserve size spatially)
      * e.g. F=3 => zero pad with 1, F=5 => zero pad with 2, F=7 => zero pad with 3
    + padding을 사용하는 이유
      * 그림의 크기가 급격하게 작아지는 것을 방지하기 위해
      * 이 부분이 모서리라는 것을 어떤 형태로든 network에 알려주기 위함
7. Swiping the entire image
  - filter 2
  - 6 filters(5x5x3) -> (convolutional layer) activation maps(?,?,6)(마지막꺼는 filter의 수와 같고 앞에꺼는 filter의 크기에 따라 달라짐, 여기선 패딩 안했다고 하면 28)
8. Convolutional layers
  - 32x32x3 image conv, relu
    + e.g. 6 5x5x3 filters -> conv relu e.g. 10 5x5x6 filters -> ...
  - How many weight variables? How to set them?
    + 처음에는 랜덤하게 초기화, 우리가 가진 데이터로 학습을 하게 함
9. Pooling layer(Sampling)
  - conv layer -> resize(sampling)
  - 한 layer씩 뽑아서 sampling하고 다시 넘기는걸 반복
10. Max pooling
  - max pool with 2x2 filters and stride 2
  - single depth slice(4x4) -> 2x2
  - 해당 filter에서 가장 큰 값을 뽑는 것
  - 전체 값들 중 한개만 뽑기 때문에 sampling이라고 부름
11. Fully Connected Layer(FC layer)
  - contains neurons that connect to the entire input volume, as in ordinary Neural Networks
12. ConvNetJS demo: training on CIFAR-10
  - *[for more information](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)*
13. case study: LeNet-5
  - [LeCun et al. 1998]
  - conv filters were 5x5, applied at stride 1
  - subsampling(pooling) layers were 2x2 applied at stride 2
    + i.e. architecture is [CONV-POOL-CONV-POOL-CONV-FC]
14. case study: AlexNet
  - [Krizhevsky et al. 2012]
  - input: 227x227x3 images
  - first layer (CONV1): 96 11x11 filters applied at stride
    * output volume [55x55x96]
    * parameters: (11*11*3)* 96=35K
  - second layer(POOL1): 3x3 filters applied at stride 2
    * output volume: 27x27x96
    * parameters: 0
  - CONV1 -> MAXPOOL1 -> NORM1 -> CONV2 -> MAXPOOL2 -> NORM2 -> CONV3 -> CONV4 -> CONV5 -> MAXPOOL3 -> FC6 -> FC7 -> FC8
  - details/retrospectives:
    + first use of ReLU
    + used Norm layers(not common anymore)
    + heavy data augmentation
    + dropout 0.5
    + batch size 128
    + SGD momentum 0.9
    + learning rate 1e-2, reduced by 10 manually when val accuracy plateaus
    + L2 weight decay 5e-4
    + 7 CNN ensemble3: 18,2% -> 15.4%
15. case study: GoogLeNet
  - [Szegedy et al. 2014]
  - inception module
  - ILSVRC 2014 winner(6.7% top 5 error)
16. case study: **ResNet**
  - [He et al. 2015]
  - ILSVRC 2015 winner (3.6% top 5 error)
  - [slide](https://www.youtube.com/watch?v=1PGLj-uKT1w) from Kaiming He's recent presentation
  - 2 ~ 3 weeks of training on 8 GPU machine
  - at runtime: faster than a VGGNet!(even though it has 8x more layers)
  - fast forward 라는 개념 사용
    + plaint net vs residual net
17. Convolutional Neural Networks for sentence classification
  - [Yoon Kim, 2014]
  - syntax classifier(text)
18. case study bonus: DeepMind's AlphaGo
  - policy network: [19x19x48] input
    + conv1: 192 5x5 filters, stride 1, pad 2 => [19x19x192]
    + conv2..12: 192 3x3 filters, stride 1, pad 1 => [19x19x192]
    + conv: 1 1x1 filter, stride 1, pad 0 => [19x19] (probability map of promising moves)

###### Laboratory
1. CNN
  - level 1: convolution
  - level 2: feature extraction(subsampling)
  - level 3: classification(fully connected)
2. CNN for CT images
  - Asan medical center & Microsoft medical BigData contest winner by GeunYoung Lee and Alex Kim
3. Convolution layer and Max pooling
  - Simple convolution layer
    + stride: 1x1
    + 3x3x1 image
    + 2x2x1 filter W
    + 1 number
  - Toy image
  ```python
  # In[2]:
  sess = tf.InteractiveSession()
  image = np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]], dtype=np.float32)
  print(image.shape)
  plt.imshow(image.reshape(3,3), cmap='Grays')

  # Out[2]:
  <matplotlib.image.AxesImage at 0x10db67dd8>
  ```
  ```
  result:
  (1,3,3,1)
  ```
  - Simple convolution layer
    + Image: 1,3,3,1 image, Filter: 2,2,1,1, Stride: 1x1, Padding: VALID
  ```python
  # print("img:\n", image)
  print("image.shape ", image.shape)
  weight = tf.constant([[[[1.]], [[1.]]], [[1.]], [[1.]]])
  print("weight.shape ", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID')
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape ", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img,0,3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
  ```
  ```
  result:
  image.shape (1,3,3,1)
  weight.shape (2,2,1,1)
  conv2d_img.shape (1,2,2,1)
  [[12. 16.]
   [24. 28.]]
  ```
  - Simple convolution layer
    + Image: 1,3,3,1 image, Filter: 2,2,1,1, Stride: 1x1, Padding: SAME
    + padding 'SAME': zero padding으로 입출력 convolution image 크기를 같게 만드는 것
  ```python
  # print("img:\n", image)
  print("image.shape ", image.shape)
  weight = tf.constant([[[[1.]], [[1.]]], [[1.]], [[1.]]])
  print("weight.shape ", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape ", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img,0,3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
  ```
  ```
  result:
  image.shape (1,3,3,1)
  weight.shape (2,2,1,1)
  conv2d_img.shape (1,3,3,1)
  [[12. 16. 9.]
   [24. 28. 15.]
   [15. 17. 9.]]
  ```
  - 3 filters(2,2,1,3)
  ```python
  # print("img:\n", image)
  print("image.shape ", image.shape)
  weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]], [[1., 10., -1.]], [[1., 10., -1.]]])
  print("weight.shape ", weight.shape)
  conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
  conv2d_img = conv2d.eval()
  print("conv2d_img.shape ", conv2d_img.shape)
  conv2d_img = np.swapaxes(conv2d_img,0,3)
  for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
  ```
  ```
  result:
  image.shape (1,3,3,1)
  weight.shape (2,2,1,3)
  conv2d_img.shape (1,3,3,3)
  [[12. 16. 9.]
   [24. 28. 15.]
   [15. 17. 9.]]
  [[120. 160. 90.]
   [240. 280. 150.]
   [150. 170. 90.]]
  [[-12. -16. -9.]
   [-24. -28. -15.]
   [-15. -17. -9.]]
  ```
4. Max pooling
  ```python
  # In[19]:
  image = np.array([[[[4],[3]],[[2],[1]]]], dtype=np.float32)
  pool = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
  print(pool.shape)
  print(pool.eval())
  ```
  ```
  result:
  (1,2,2,1)
  [[[[4.]
     [3.]]

    [[2.]
     [1.]]]]
  ```
  - 1) MNIST image loading
  ```python
  # In[6]:
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  # Check out https://www.tensorflow.org/get_started/mnist/beginners
  # for more information about the mnist dataset
  ```
  ```
  result:
  Extracting MNIST_data/train-images-idx3-ubyte.gz
  Extracting MNIST_data/train-labels-idx1-ubyte.gz
  Extracting MNIST_data/t10k-images-idx3-ubyte.gz
  Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
  ```
  ```python
  # In [7]:
  img = mnist.train.image[0].reshape(28,28)
  plt.imshow(img, cmap='gray')

  # Out [7]:
  <matplotlib.image.AxesImage at 0x115029ac8>
  ```
  - 2) MNIST convolution layer
  ```python
  # In[8]:
  sess = tf.InteractiveSession()

  img = img.reshape(-1,28,28,1)
  W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
  conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
  print(conv2d)
  sess.run(tf.global_variables_initializer())
  conv2d_img = conv2d.eval()
  conv2d_img = np.swapaxes(conv2d_img,0,3)
  for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
  ```
  ```
  result:
  Tensor("Conv2D_1:0", shape=(1,14,14,5), dtype=float32)
  ```
  - 3) MNIST max pooling
  ```python
  # In[9]:
  pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  print(pool)
  sess.run(tf.global_variables_initializer())
  pool_img = pool.eval()
  pool_img = np.swapaxes(pool_img,0,3)
  for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')
  ```
  ```
  result:
  Tensor("MaxPool_2:0", shape=(1,7,7,5), dtype=float32)
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-0-cnn_basics.ipynb)*
5. Simple CNN
  - Input layer -> Convolution layer 1 -> Pooling layer 1 -> Convolutional layer 2 -> Pooling layer2 -> Fully-Connected layer
  - 1) CONV layer 1
  ```python
  # input placeholders
  X = tf.placeholder(tf.float32, [None,784])
  X_img = tf.reshape(X, [-1,28,28,1]) # img 28x28x1 (black/white)
  Y = tf.placeholder(tf.float32, [None,10])

  # L1 ImgIn shape=(?,28,28,1)
  W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01)) # 마지막 값: # of filters
  # Conv -> (?,28,28,32)
  # Pool -> (?,14,14,32)
  L1 =tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
  L1=tf.nn.relu(L1)
  L1=tf.nn.max_popol(L1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
  ```
  ```
  ...
  Tensor("Conv2D:0", shape=(?,28,28,32), dtype=float32)
  Tensor("Relu:0", shape=(?,28,28,32), dtype=float32)
  Tensor("MaxPool:0", shape=(?,14,14,32), dtype=float32)
  ...
  ```
  - 2) CONV layer2
  ```python
  # L2 ImgIn shape=(?,14,14,32)
  W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
  # Conv -> (?,14,14,64)
  # Pool -> (?,7,7,64)
  L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
  L2 = tf.nn.relu(L2)
  L2 = tf.nn.max_popol(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # stride 중요
  L2 = tf.reshape(L2,[-1, 7*7*64]) # important
  ```
  ```
  ...
  Tensor("Conv2D_1:0", shape=(?,14,14,64), dtype=float32)
  Tensor("Relu_1:0", shape=(?,14,14,64), dtype=float32)
  Tensor("MaxPool_1:0", shape=(?,7,7,64), dtype=float32)
  Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
  ...
  ```
  - 3) Fully Connected(FC, Dense) layer
  ```python
  L2 = tf.reshape(L2,[-1, 7*7*64])

  # Final FC 7x7x64 inputs -> 10 outputs
  W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer = tf.contrib.layers.xavier_initializer())
  b = tf.Variable(tf.random_normal([10]))
  hypothesis = tf.matmul(L2,W3)+b

  # define cost/loss & optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  ```
  - 4) Training and Evaluation
  ```python
  # initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train my model
  print('Learning started. It takes sometime.')
  for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      feed_dict = {X: batch_xs, Y: batch_ys}
      c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
      avg_cost += c/total_batch
    print('Epoch: ', '%04d'%(epoch+1), 'cost= ', '{:.9f}'.format(avg_cost))

  print('Learning Finished!')

  # Test model and check accuracy
  correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
  ```
  - Accuracy: 0.9885
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-1-mnist_cnn.py)*
6. Deep CNN
  - Accuracy: 0.9938
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-2-mnist_deep_cnn.py)*
7. python class
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-3-mnist_cnn_class.py)*
8. [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
  ```python
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding="SAME", strides=2)
  dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
  ...
  flat = tf.reshape(dropout3, [-1, 128*4*4])
  dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
  dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-4-mnidst_cnn_layers.py)*
9. [Ensemble](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)
  - 여러개를 조합해서 조화롭게 만든다
  - classification models, predictions -> meta-classifier => final prediction
10. Ensemble training
  ```python
  class Model:
    def __init__(self, sess, name):
      self.sess = sess
      self.name = name
      self._build_net()

    def _build_net(self):
      with tf.variable_scope(self.name):

    ...

    models = []
    num_models = 7
    for m in range(num_models):
      models.append(Model(sess, "model"+str(m)))

    sess.run(tf.global_variables_initializer())
    print('Learning Started!')

    # train my model
    for epoch in range(training_epochs):
      avg_cost_list = np.zeros(len(models))
      total_batch = int(mnist.train.num_examples/batch_size)
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
          c, _ = m.train(batch_xs, batch_ys)
          avg_cost_list[m_idx] += c/total_batch

      print('Epoch: ', '%04d'%(epoch+1), 'cost= ', avg_cost_list)

    print('Learning Finished!')
  ```
  - *[full code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-5-mnist_cnn_ensemble_layers.py)*
11. Ensemble prediction
  - 각각의 모델들한테 예측을 해보라고 하고 예측이 끝나면 sum(= prediction)에 다 합을 해서 넣고 argmax를 이용해서 최댓값을 찾는 것
  ```python
  # Test model and check accuracy
  test_size = len(mnist.test.labels)
  predictions = np.zeros(test_size*10).reshape(test_size,10)

  for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy: ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

  ensemble_correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(mnist.test.labels,1))
  ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
  print('Ensemble accuracy: ', sess.run(ensemble_accuracy))
  ```
  ```
  result:
  0 Accuracy: 0.9933
  1 Accuracy: 0.9946
  2 Accuracy: 0.9934
  3 Accuracy: 0.9935
  4 Accuracy: 0.9935
  5 Accuracy: 0.9949
  6 Accuracy: 0.9941

  Ensemble accuracy: 0.9952
  ```
12. Exercise
  - Deep & Wide?
  - CIFAR 10
  - ImageNet
