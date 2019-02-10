#### Lecture06
## Softmax classification: Multinomial classification

###### Theory
1. Logistic regression
  - z = H(X), 0 < g(z) < 1
  - g(z) = 1/(1+e^-z) : sigmoid or logistic
2. Multinomial classification
  - x1(hours), x2(attendance), y(grade A, B, C)
  - 3개의 binary만 가지고도 구현 가능 -> C인 것과 아닌 것, 나머지 중에서 B인 것과 아닌 것, 나머지 중에서 A인 것과 아닌 것
  - W matrix의 열을 늘림(shape을 바꿈)
  - where is sigmoid?
    + x(a)의 hat, x(b)의 hat, x(c)의 hat 각각에 대해 적용시키면 됨 but 코딩하기 복잡함
  - sigmoid?
    + logistic classifier WX = y a=2.0 -> p=0.7, b=1.0 -> p=0.2, c=0.1 -> p=0.1
    + softmax 라는 함수 사용
    + scores -> probabilities <-> One-hot encoding(argmax 사용, 해당 자리가 hot하다 = 1로 표시한다)
    + 그림은 [여기](https://www.udacity.com/course/viewer#!/c-ud730/l-6370362152/m-6379811817) 참고
3. Cost function
  - cross-entropy
    + D(S,L) = -ΣL(i)log(S(i)) (S: S(y)(= y의 hat), L: y)
    + -ΣL(i)log(S(i)) = -ΣL(i)log(y의 hat(i)) = ΣL(i)x(-log(y의 hat(i)))
      * L = [a=0, b=1] = b일 때, Y의 hat = [a=0, b=1] = b이면 예측 성공, Y의 hat = [a=1, b=0] = a면 예측 실패
      * 예측이 맞으면 값이 작고 틀리면 값이 크게 되어 벌칙을 줌
4. logistic cost vs cross entropy
  - c(h(x),y) = ylog(h(x))-(1-y)log(1-h(x))
  - D(S,L)=-ΣL(i)log(S(i))
  - 두 개는 같은 것
  - 왜 같은 것일지는 숙제로 생각해볼 것
5. cost function
  - L = 1/NΣD(S(WX(i)+b),L(i))
  - 1/NΣ -> training set, L -> loss
6. gradient descent
  - step: -α*(L(w1,w2) 미분한 것) (derivative)
  - loss(= cost) -> 밥그릇 모양의 그래프

###### Laboratory
1. softmax function
  - 여러 개의 class를 예측할 때 유리
  - 각각의 값을 softmax function을 통과시키고 나면 확률로 표현이 됨(확률을 모두 더하면 1이 됨)
  - XW = y -> tf.matmul(X,W)+b
  - S(y(i))(probabilities로 나옴) -> **hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)**
2. cost function: cross entropy
  - L = 1/NΣD(S(WX(i)+b),L(i))
  ```python
  # cross entropy cost/loss
  cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
  ```
3. test & One-hot encoding
  ```python
  hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
  # testing & one-hot encoding
  a = sess.run(hypothesis, feed_dict={X: [1,11,7,9]})
  print(a, sess.run(tf.arg_max(a,1)))
  ```
  ```
  result:
  [[1.38904958e-03 9.98601854e-01 9.06129117e-06]]  # softmax 함수를 통과하여 확률로 돌려줌
  [1]  # arg_max를 통해 1 반환
  ```
  ```python
  # 여러개 질문할 때
  all = sess.run(hypothesis, feed_dict={X: [[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
  print(all, sess.run(tf.arg_max(all,1)))
  ```
  ```
  result:
  [[1.38904958e-03 9.98601854e-01 9.06129117e-06]
  [9.31192040e-01 6.29020557e-02 5.90589503e-03]
  [1.27327668e-08 3.34112905e-04 9.99665856e-01]]
  [1 0 2]
  ```
4. softmax_cross_entropy_with_logits
  ```python
  # tf.nn.softmax computes softmax activations
  # softmax = exp(logits)/reduce_sum(exp(logits), dim)
  logits = tf.matmul(X,W)+b # logits(= scores)
  hypothesis = tf.nn.softmax(logits)  # softmax(= probabilities)

  # 1) cross entropy cost/loss
  cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
  # 2) cross entropy cost/loss
  cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
  cost = tf.reduce_mean(cost_i) # 1번의 cost와 동일
  ```
5. [Animal classification](https://kr.pinterest.com/explore/animal-classification-activity/) with softmax_cross_entropy_with_logits
  ```python
  # predicting animal type based on various features
  xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]
  ```
  - tf.one_hot and reshape
    + if the input indices is rank N, the output will have rank N+1
    + the new axis is created at dimension axis(default: the new axis is appended at the end)
  ```python
  Y = tf.placeholder(tf.int32, [None,1])  # 0~6, shape=(?,1)
  Y_one_hot = tf.one_hot(Y, nb_classes)  # one_hot shape=(?,1,7), tf.one_hot: One-hot으로 바꾸기 위해 사용하는 함수
  Y_one_hot = tf.reshape(Y_one_hot, [-1,nb_classes])  # shape=(?,7), -1: everything
  # 예측한 값이 맞는지 틀린지 확인하는 부분
  prediction = tf.argmax(hypothesis, 1) # probabilities를 0~6 값 중 하나로 만들어내는 것
  correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # Y 그 자체(one_hot으로 만들기 전)와 prediction이 맞는지
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # zip: 묶는 것, flatten: [[1], [0]] -> [1,0]
  # step이 돌면서 loss 줄어들고 accuracy는 100%가 됨
  ```
  - *[for more information](https://www.tensorflow.org/api_docs/python/tf/one_hot)*
