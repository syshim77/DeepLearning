#### Lecture04
## Multivariable linear regression

###### Theory
1. Recap
  - hypothesis
    + H(x) = Wx+b
  - cost(loss) function
    + cost(W,b) = 1/mΣ(H(x(i))-y(i))2 (W와 b의 함수, H: prediction, y: true)
  - gradient descent algorithm
    + 밥그릇 모양 그래프
    + cost 최저값 찾기 위한 알고리즘
2. Predicting exam score
  - 1개의 value: linear regression
  - regression using three inputs(x1, x2, x3) 일땐?
3. Hypothesis
  - H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b
  - cost(W,b) = 1/mΣ(H(x1(i), x2(i), x3(i)) - y(i))2
4. Multi-variable
  - H(x1, x2, ... , xn) = w1x1 + w2x2 + ... + wnxn
5. Matrix
  - w1x1 + w2x2 + ... + wnxn 식을 쉽게 표현하기 위한 방법
  - *[matrix multiplication](https://www.mathsisfun.com/algebra/matrix-multiplying.html)*
6. Hypothesis using Matrix
  - H(X) = XW
  - X,W는 matrix
  - matrix에서는 x가 w보다 먼저 쓰임
7. Many x Instances
  - (x1 x2 x3) 한 세트를 instance 라고 함
  - instance 수대로 matrix를 줘버림(X만 늘리고 W는 그대로)
  - 각각 instance를 계산할 필요없이 한번에 긴 matrix에 넣으면 바로 계산이 됨
  - [5,3] [3,1] -> [5,1] (instance가 3개, x(y variable의 개수)가 3개)
  - [5,3] [?,?] -> [5,1] (이런식으로 주어지면 ? 안에 3,1이 들어간다는 걸 알 수 있어야 함)
  - [n,3] [3,1] -> [n,1] (n은 -1 or None 이라고 표시)
  - [n,3] [?,?] -> [n,2] (n output)
  - matrix를 사용하면 multi-instance, multi-variable, multi-output 일 때 쉽게 처리 가능하다는 것이 장점
8. WX vs XW
  - lecture(theory)
    + H(x) = Wx+b
  - implementation(TensorFlow)
    + **H(x) = XW**

###### Laboratory
1. [Hypothesis using Matrix](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-1-multi_variable_linear_regression.py)
  - cost = tf.reduce_mean(tf.square(hypothesis-Y))
  - 학습 시킬 때 train을 돌림(cost, hypothesis도 같이 확인하는 방식)
  - 전에 작성했던 코드에서 x, weight 부분만 많이 확장됨
  - 많이 실행시킬수록 cost가 많이 낮아짐(마지막엔 원하는 y값에 거의 수렴하는 값이 나옴)
2. [Matrix](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-2-multi_variable_matmul_linear_regression.py)
  - H(X) = XW
  - n개 예측일 땐 None이라고 표시
  - 복잡했던 전 코드가 단순해짐(matrix의 장점)
  - 결과는 전과 똑같이 나옴
3. [Loading Data from File](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-3-file_input_linear_regression.py)
  - 보통 matrix를 .csv에 저장(콤마(,)로 연결된 정보 저장 파일)
  - matrix 전체가 같은 data type 이어야 함
  - 파일 전체를 array로 읽어오게 됨
4. **Slicing**
  ```python
  nums = range(5)
  print nums  # Prints [0,1,2,3,4]
  print nums[2:4] # Get a slice from index 2 to 4 (exclusive); Prints[2,3]
  print nums[2:]  # Get a slice from index 2 to the end; Prints [2,3,4]
  print nums[:2]  # Get a slice from the start to index 2 (exclusive); Prints [0,1]
  print nums[:]   # Get a slice of the whole list; Prints [0,1,2,3,4]
  print nums[:-1] # Slice indices can be negative; Prints [0,1,2,3]
  nums[2:4] = [8,9] # Assign a new sublist to a slice
  print nums  # Prints [0,1,8,9,4]
  ```
  - *[for more information](http://cs231n.github.io/python-numpy-tutorial/)*
5. Indexing, Slicing, Iterating
  - arrays can be indexed, sliced, iterated much like lists and other sequence types in python
  - as with python lists, slicing in NumPy can be accomplished with the colon(:) syntax
  - colon instances(:) can be replaced with dots(...)
  - ex1) x_data = xy[:, 0:-1] # n행을 다 가져오고 처음부터 마지막 세로 한줄를 제외한 모든 것을 x라고 하겠다
  - ex2) y_data = xy[:, [-1]] # n행을 다 가져오고 마지막 끝에 세로 한줄만 가져오겠다
  ```python
  # 주의해야할 부분(shape 신경 쓸 것)
  W = tf.Variable(tf.random_normal([3,1]), name='weight') # 3: x 개수, 1: y 개수
  b = tf.Variable(tf.random_normal([1]), name='bias') # 1: y 개수
  # 학습을 한 다음에 예측하는 부분
  # ask my score
  print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100,70,101]]}))
  print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70,110],[90,100,80]]}))  # 2개 동시에 물어볼 때
  ```
6. [Queue Runners](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-04-4-tf_reader_linear_regression.py)
  ```python
  # 1)
  filename_queue = tf.train.string_input_producer(['data-01-test-score.csv','data-02-test-score.csv', ...], shuffle=False, name='filename_queue')
  # 2)
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  # 3)
  record_defaults = [[0.], [0.], [0.], [0.]]
  xy = tf.decode_csv(value, record_defaults=record_defaults)
  ```
7. tf.train.batch
  ```python
  # 일종의 펌프같은 역할(계속 데이터를 읽어오게 됨)
  # collect batches of csv in
  train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) # xy[0:-1]: x data, xy[-1:]: y data, batch_size: 한 번 펌프할 때마다 몇개씩 가져올까

  for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
  ```
8. shuffle_batch
  - *[for more information](https://www.tensorflow.org/programmers_guide/reading_data)*
