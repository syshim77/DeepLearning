#### Lecture05
## Logistic (regression) classification

###### Theory
1. Overall
  - 이 알고리즘은 classification 알고리즘 중 굉장히 정확도가 높은 것으로 알려져있음
  - 실생활 문제들에 바로 적용 가능
  - 이 강의의 궁극적 목표가 neural network와 deep learning을 잘 이해하는 것인데, 이 알고리즘이 중요한 component
2. Regression(HCG)
  - H(Hypothesis)
    + H(X) = WX
  - C(Cost)
    + cost(W) = 1/mΣ(WX-y)2
  - G(Gradient decent)
    + W := W - α*(cost(W) 미분한 것)
3. Classification
  - 오늘 할거는 특별히 binary classification
  - Spam detection: Spam or Ham
    + 이메일에 스팸이 많이 안오는 이유가 이 처리가 잘 되어있기 때문
  - Facebook feed: Show or Hide
    + 무수히 많은 친구들의 타임라인들을 다 보여주는 것이 아니라 내가 그동안 좋아요를 누른 것들을 가지고 학습해서 수백개의 타임라인들 중 어떤 라인은 보여주고 어떤 라인은 안보여줌
  - Credit card fraudulent transaction detection: Legitimate/Fraud
    + 도난당했을 때 내가 쓰던 항목들과 다른 곳에 지출이 되면 도난당했음을 아는 것
4. 0,1 Encoding
  - Spam detection: Spam -> 1, Ham -> 0
  - Facebook feed: Show -> 1, Hide -> 0
  - Credit card fraudulent transaction detection: Legitimate -> 0, Fraud -> 1
5. Radiology
  - malignant tumor(나쁜 것), benign tumor(좋은 것)
  - finance(주식을 살까 팔까 -> 이전의 시장 동향을 파악해서 알 수 있음)
6. Pass(1)/Fail(0) based on study hours
  - binary classification의 예
  - linear regression? 0.5와 만나는 지점을 선에서 알아내서 그 값보다 크면 pass, 작으면 fail 하면 될 것 같지 않나?
    + 학습 시간이 50시간이면 값이 엄청 올라가야하는데 실제로는 계속 값이 전과 같음
    + 합격임에도 불구하고 인식이 제대로 안되는 경우가 발생
    + We know y is 0 or 1
      * H(x) = Wx+b
    + Hypothesis can give values large than 1 or less than 0
    + ex) W = 0.5, b = 0, x = 100이면 H값이 0<H<1이 아니라 50이 되어버림
    + 따라서 linear regression 사용 불가
7. Logistic Hypothesis
  - g(z) = 1/(1+e^-z)
  - sigmoid function 이라고 부름(= logistic function)
    + curved in two directions
    + like the letter "S" or the Greek (sigma))
  - z값이 작아질수록 0에 수렴, 커질수록 1에 수렴
  - z = WX, H(X) = 1/(1+e^((-W^T)X)) or 1/(1+e^-WX)
8. Cost Function
  - in linear regression
    + cost(W,b) = 1/mΣ(H(x(i))-y(i))2 when H(x) = Wx+b
    + 밥그릇 모양의 그래프(2차 방정식 모양)로 중간 지점이 최소값을 가짐
  - in logistic (regression) classification
    + H(X) = 1/(1+e^((-W^T)X)) or 1/(1+e^-WX) (0<H<1)
    + 구불구불한 함수이므로 경사타고 내려가기 알고리즘(gradient descent algorithm)을 바로 적용하면 시작점에 따라 최저점이 달라질 수 있음
    + 시작점에서의 최저점 = local minimum, 전체에서의 최저점 = global minimum
    + model이 나쁘게 prediction 하게 됨
    + 따라서 이 상태로 알고리즘을 사용할 수 없음
9. New cost function for logistic
  - cost(W) = 1/mΣc(H(x),y)
  - c(H(x),y) = -log(H(x)) :y=1, -log(1-H(x)) :y=0
10. Understanding cost function
  - exponential(e)과 상이한 것이 로그함수이므로 로그함수 사용
  - g(z) = -log(z)라고 할 때, z가 1이면 값이 0, z가 0에 가까워지면 값이 한없이 커짐
  - cost function의 의미
    + 실제의 값과 예측한 값이 거의 같으면 cost 작아짐, 틀리면 cost 커지게 해서 model에 벌을 줌
  - cost = -log(z)
    + y=1 H(x)=1 -> cost=0 (이게 우리가 원하는 결과)
    + y=1 H(x)=0 -> cost=∞ (시스템에 벌을 주는 것)
  - cost = -log(1-z)
    + y=0 H(x)=0 -> cost=0
    + y=0 H(x)=1 -> cost=∞
  - 이 식을 가지고 coding을 하다보면 복잡해짐
    + 프로그래밍할 때 편하게 할 수 있도록 수식을 조금 수정
    + **c(H(x),y) = -ylog(H(x)) - (1-y)log(1-H(x))**
    + y=1: c = -log(H(x)), y=0: c = -log(1-H(x)) 이므로 앞선 식과 같음
11. Minimize cost - gradient descent algorithm
  - cost(W) = -1/mΣ(ylog(H(x))+(1-y)log(1-H(x)))
  ```python
  # cost function
  cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))
  # minimize
  # 이 부분은 이미 만들어져있는 GradientDescentOptimizer 라는게 있기 때문에 이걸 사용해서 그대로 하면 됨
  a = tf.Variable(0.1)  # learning rate, alpha
  optimizer = tf.train.GradientDescentOptimizer(a)
  train = optimizer.minimize(cost)
  ```

###### Laboratory
1. Logistic regression
  - H(X) = 1/(1+e^-WX)
  - cost(W) = -1/mΣ(ylog(H(x))+(1-y)log(1-H(x))) (이 식의 그래프도 밥그릇 모양)
  - W := W - α*(cost(W) 미분한 것)
2. training data
  ```python
  x_data = [[1,2],[2,3],[3,1].[4,3],[5,3],[6,2]]  # [x1,x2]
  y_data = [[0],[0],[0],[1],[1],[1]]  # 0: fail, 1: pass
  # placeholders for a tensor that will be always fed
  # shape에 유의
  X = tf.placeholder(tf.float32, shape=[None,2])  # shape=[총 n개, x가 2개]
  Y = tf.placeholder(tf.float32, shape=[None,1])  # shape=[총 n개, y가 1개]
  ```
  - matrix multiplication할 때 W의 shape에 유의
  - H(X) = 1/(1+e^-WX) -> hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
  - W := W - α*(cost(W) 미분한 것) -> train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
  - hypothesis > 0.5이면 pass, hypothesis < 0.5이면 fail
  - cast: true or false로 나타내줌
3. train the model
  - 이전과 동일
  - *[전체 코드](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-01-logistic_regression.py)*
4. Classifying diabetes
  ```python
  # 0: diabetes x, 1: diabetes o
  xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
  x_data = xy[:, 0:-1]
  y_data = xy[:, [-1]]
  # 뒷부분은 거의 공통적으로 사용됨
  # 주의할 점은 x가 총 8개이므로 placeholder, W에서의 shape
  # step이 진행될수록 cost가 점점 내려감
  ```
5. Exercise
  - CSV reading using tf.decode_csv
  - Try other classification data from [Kaggle](https://www.kaggle.com)
