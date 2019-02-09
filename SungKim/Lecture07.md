#### Lecture07
## Application & Tips: Learning rate, Data preprocessing, **Overfitting**

###### Theory
1. Gradient descent
  ```{.python}
  # Minimize error using cross entropy
  # -α*(L(w1,w2) 미분한 것(= derivative))
  learning_rate = 0.001 # α = learning_rate, 지금까지는 임의의 값을 지정
  cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))  # cross entropy
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # gradient descent
  ```
2. Large learning rate: Overshooting
  - step이 굉장히 크면 global minimum을 찾지 못하고 그래프 양쪽을 왔다갔다만 반복 or 그래프 밖으로 튕겨나갈 수 있음(숫자가 아닌 값이 출력되게 됨)
  - cost가 줄어들지 않고 계속 커지다가 튕겨나감
  - 이런 현상을 overshooting 이라고 함
3. Small learning rate: takes too long, stops at local minimum
  - 산을 내려갈때 굉장히 작은 보폭으로 내려가면 해가 다 질 때까지 하산을 못하는 것과 같음
  - training을 stop한 지점이 global minimum까지 못미치고 local minimum이 됨
4. Try several learning rates
  - observe the cost function
  - check it goes down in a reasonable rate
  - normally start to try from 0.01
5. Data(X) preprocessing for Gradient descent
  - 어떤 데이터 사이에 값 차이가 너무 크게 나면 α값(= learning rate)이 좋음에도 불구하고 튀어 나가는 경우가 생김
  - normalize 할 필요가 있음
  - zero-centered data: 중심을 0으로 가도록 바꾸어주는 것
  - **normalized data**: 모든 데이터가 어떤 범위 내에 항상 들어가도록 바꾸어주는 것
6. [Standardization](http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html)
  - **x'(j) = (x(j)- μ(j))/σ(j)**
  - X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
7. Overfitting
  - machine learning의 가장 큰 문제점 중 하나
  - our model is very good with training data set(with memorization)
  - not good at test dataset or in real use
8. Solutions for overfitting
  - graph 자체를 너무 구부리는 것을 overfitting이라고 함
  - more training data!
  - reduce the number of features
  - **regularization**
    + graph를 구부리지 말고 펴지도록 하는게 일반화
    + let's not have too big numbers in the weight
      * L = 1/NΣD(S(WX(i)+b),L(i)) + λΣW2 (제곱)(λ: regularization strength, 값이 크면 엄청 중요하다는 뜻이고 작으면 안쓰겠다는 뜻)
      * l2reg = 0.001*tf.reduce_sum(tf.square(W))
9. Summary
  - learning rate
  - data preprocessing
  - **overfitting**
    + more training data
    + regularization
10. Performance evaluation: is this good?
  - evaluation using training set?
    + 100% correct(accuracy)
    + can memorize
  - training and test sets
    + data를 나누어서 앞부분은 training, 뒷부분은 test set으로 하고 뒷부분은 무조건 숨겨야 함
    + training set을 가지고 학습을 시키고 test set을 가지고 답과 비교하여 확인을 해보면 accuracy를 알 수 있음
    + training = 교과서, test set = 실전 시험
  - training, validation and test sets
    + training set을 또 2개로 나눌 수 있음
      * 완벽한 training set
      * validation set
    + training set으로 학습시키고 validation set을 가지고 tuning
    + validation = 모의 시험
11. Online learning
  - training 방법 중 하나
  - 전체 data가 100만개가 있다고 가정하면, 10만개씩 model을 학습시킴
  - 이전 학습 결과가 model에 남아있어야 함
  - 지금까지의 학습 결과에 이후 학습 결과가 추가되는 방식
  - 나중에 새로운 데이터가 또 들어오면 이전 데이터를 처음부터 다시 학습시키는 것이 아니라 새로운 데이터만 추가로 학습시킴
  - *[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)*
12. Accuracy
  - how many of your predictions are correct?
  - 95% ~ 99%?(최근 이미지 인식 관련 분야의 정확성)
  - check out the lab video

###### Laboratory
1. [training and test datasets](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-1-learning_rate_and_evalution.py)
  - 이전까지는 학습시킨 데이터로 확인했지만 지금부터는 아님
  - test set은 모델 입장에서 한번도 본 적이 없는 것
  - 학습이 완료된 시점에서 테스트
  ```{.python}
  # training data
  x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
  y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
  # evaluation our model using this test dataset
  x_test = [[2,1,1],[3,1,2],[3,3,4]]
  y_test = [[0,0,1],[0,0,1],[0,0,1]]
  ```
  - placeholder가 유용 -> training할 때는 그 데이터를 던져주면 되고 테스트할때는 그 데이터를 던져주면 됨
2. learning rate: NaN!
  - large learning rate: overshooting
  - small learning rate: many iterations until convergence and trapping in local minima
  - accuracy가 100%였던 1번과 똑같은 모델로 learning_rate=1.5로 주면 accuracy 0%이 됨
    + nan이 나오면 learning rate이 너무 큰거 아닌가 의심
  - 반대로 learning_rate=1e-10로 주어도 accuracy 0%이 됨
    + learning_rate이 너무 작은 값이라 cost가 처음 값에 계속 머물게 됨
  - *[for more information](http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html)*
3. [Non-normalized inputs](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-2-linear_regression_without_min_max.py)
  - inf, nan 등이 섞인 이상한 결과값이 나오게 됨
4. Normalized inputs(min-max scale)
  ```{.python}
  xy = MinMaxScaler(xy)
  print(xy)
  ```
  - min, max를 0, 1로 주면 그 사이의 값으로 data가 바뀜
  - 데이터의 형태가 이상하거나 데이터가 너무 들쭉날쭉하면 normalize 이용
5. MNIST dataset
  - 28x28x1 image
  ```{.python}
  # MNIST data image of shape 28 * 28 = 784
  X = tf.placeholder(tf.float32, [None, 784])
  # 0 - 9 digits recognition = 10 classes (one-hot encoder)
  Y = tf.placeholder(tf.float32, [None, nb_classes])
  ```
  - MNIST dataset using TensorFlow
  ```{.python}
  from tensorflow.examples.tutorials.mnist import input_data
  # check out https://www.tensorflow.org/get_started/mnist/beginners
  # for more information about the mnist dataset
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot=True: Y값을 우리가 원하는대로 one-hot으로 읽어온다
  ...
  batch_xs, batch_ys = mnist.train.next_batch(100)
  ...
  print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
  ```
  - reading data and set variables
  ```{.python}
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  nb_classes=10

  X = tf.placeholder(tf.float32, [None, 784])
  Y = tf.placeholder(tf.float32, [None, nb_classes])

  W = tf.Variable(tf.random_normal([784, nb_classes]))
  b = tf.Variable(tf.random_normal([nb_classes]))
  ```
  - softmax!
  ```{.python}
  # Hypothesis (using softmax)
  hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
  cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
  # Test model
  is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
  ```
6. training [epoch/batch](http://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks)
  - epoch: 전체 dataset을 한 번 학습시킨 것
  - batch size: 한 번에 몇 개씩 학습시킬지
  ```{.python}
  # parameters
  training_epochs = 15
  batch_size = 100
  with tf.Session() as sess:
    # initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # training cycle
    for epoch in range(training_epochs):
      avg_cost = 0
      total_batch = int(mnist.train.num_examples/batch_size)

      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xsw, Y: batch_ys})
        avg_cost += c/total_batch

    print('Epoch: ', '%04d'%(epoch+1), 'cost= ', '{:.9f}'.format(avg_cost))
    ```
7. [report results on test dataset](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-4-mnist_introduction.py)
  ```{.python}
  # test the model using test sets
  print("Accuracy: ", accuracy_eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))  # accuracy_eval() = sess.run()
  ```
8. sample image show and prediction
  ```{.python}
  import matplotlib.pyplot as plt
  import random
  # get one and predict
  r = random.randint(0, mnist.test.num_examples-1)
  print("Label :", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
  print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))
  plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
  plt.show()
  ```
