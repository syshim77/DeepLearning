#### Lecture02
## Linear Regression

###### Theory
1. Predicting exam score: Regression
  - x: hours, y: score(0~100) 인 데이터를 준다고 가정 -> supervised learning 중에서도 regression
  - 이 데이터를 가지고 학습시킴(training), 이 때 데이터는 training data 라고 함
  - model을 하나 만듬 -> regression
  - ex) 7시간 공부했는데 이 학생은 몇점정도 받을까?(x를 주고 y를 원함)
  - 이것을 linear regression이라고 함
2. Regression(data)
  - x: 예측을 위한 기본적인 자료(feature), y: 예측 대상
3. Regression(presentation)
  - 3개의 학습 데이터를 주고 학습시킴
4. (linear)Hypothesis
  - linear한 모델이 데이터에 맞을 것이라고 가설을 세움
  - 효과가 많음 -> 세상에 있는 많은 것들이 이런 형태로 나타남
  - ex) 공부를 많이 할수록 성적이 높아짐, 훈련을 많이 할수록 달리기 속도가 빨라짐, 집의 크기가 클수록 가격이 높아짐
  - 데이터가 있으면 거기에 맞는 linear한 선을 찾는 것 = 학습을 하는 것
  - 수학적으로 나타내면 -> H(x) = Wx + b
  - 따라서, 1차 방정식에 넣을 것이라는게 가설
5. Which hypothesis is better?
  - 실제 데이터의 점과 가설이 나타내는 점과의 거리가 가까우면 좋은 것, 멀면 나쁜 것
6. Cost Function(= Loss Function)
  - How fit the line to our (training) data: H(x) - y (y는 실제 데이터 값)
  - 그러나, 저 식은 좋지 않음(+, - 둘 다 가능하기 때문)
  - 따라서, (H(x) - y)2 (제곱을 함) -> cost function
  - cost(W, b) = 1/mΣ(H(x(i) - y(i)))2 (m: 학습 데이터 갯수, 평균을 내기 위해 m으로 나눔)
  - 가장 작은 cost 값을 갖는 W와 b를 찾는 것을 학습이라고 함
  - minimize cost(W,b) 하게 하는 W,b 찾는 것 = 학습의 목표

###### Laboratory([code](https://github.com/hunkim/DeepLearningZeroToAll/))
1. Hypothesis and Cost Function
  - 예측을 어떻게 할 것인가
  - H: 예측 값, y: 실제 값
  - W, b가 달라질수록 cost가 커질수도 있고 작아질수도 있음
  - 학습을 한다 = cost minimize
2. Tensorflow Mechanics  
  1) Build graph using Tensorflow operations  
  // x and y data  
  x_train = [1,2,3]  
  y_train = [1,2,3]  
  // Variable: Tensorflow가 사용하는 variable(자체적으로 변경시키는 값, trainable variable, 학습하는 과정에서 자기가 변경시킨다)  
  W = tf.Variable(tf.random_normal([1]), name = 'weight')  
  b= tf.Variable(tf.random_normal([1]), name='bias')  
  // our hypothesis (Wx+b)  
  **hypothesis = x_train*W + b**  
  // cost/loss function  
  cost = tf.reduce_mean(tf.square(hypothesis - y_train)) // reduce_mean: 값을 평균내주는 것    
  // minimize using GradientDescent  
  **optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  
  train = optimizer.minimize(cost)**  // train: node name(cost가 연결되어 있음)  
  2) feed data and run graph(operation) -> sess.run(op, feed_dict={x: x_data})  
  3) run/update graph and get results  
  // launch the graph in a session  
  sess = tf.Session()  
  // initializes global variables in the graph  
  **sess.run(tf.global_variables_initializer())**  
  // fit the line  
  for step in range(2001):  
    sess.run(train) // train을 실행시키게 되면 연결된 것들을 모두 실행시키게 되어 W, b까지 실행 시키게 됨  
    if step % 20 == 0  
      print(step, sess.run(cost), sess.run(W), sess.run(b))  
  - 실행 결과: W = 1, b = 0 에 수렴하게 됨
  - 실행을 많이 할수록 cost 값이 작아짐
3. Placeholders(지정해두고 필요할 때 값을 던져줌)  
  // placeholders for a tensor that will be always fed using feed_dict  
  // see code from [here](http://stackoverflow.com/questions/36693740)  
  // x_train, y_train 대신 사용  
  x = tf.placeholder(tf.float32)  // or (tf.float32, shape=[None])  
  y = tf.placeholder(tf.float32)  // or (tf.float32, shape=[None])  
  // fit the line  
  for step in range(2001):  
    cost_val, W_val, b_val, _ = \
    sess.run([cost, W, b, train], feed_dict={x: [1,2,3], y: [1,2,3]})  
    if step % 20 == 0:  
      print(step, cost_val, W_val, b_val)  
  - placeholder를 사용하는 가장 큰 이유: 만들어진 모델에 대해서 값을 따로 넘겨줄 수 있음
  - linear model을 만들어둔 다음에 학습 데이터를 나중에 줄 수 있음
  - 시간이 지날수록 cost는 작은 값으로 수렴하고 W = 1, b = 1.1에 수렴
4. Summarize
  - cost에 해당하는 그래프 만듬
  - feed_dict를 사용해서 원하는 x, y 데이터를 던져주어서 학습시킴
  - W, b 업데이트 시킴
  - hypothesis 같은걸 가져와서 출력도 해봤음
