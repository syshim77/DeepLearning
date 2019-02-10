#### Lecture08
## Deep Neural nets for Everyone

###### Theory
1. Ultimate dream: thinking machine
  - [schematic of a biological neuron](http://sebastianraschka.com/)
  - 사람들이 뇌가 생각보다 복잡하게 연결되어있고, 뉴런이 생각보다 단순하게 움직인다(이렇게 단순한데 어떻게 생각을 할 수 있는 것일까 싶을 정도로 단순)는 사실을 알고 놀람
  - xw+b > specific value: 활성화, xw+b < specific value: 비활성화(activation functions)
  - logistic regression units
2. Hardware Implementations
  - Frank Rosenblatt, ~1957: Perceptron
  - Widrow and Hoff, ~1960: Adaline/Madaline
3. False promises
  - *[The Article from NewYork Times](http://query.mytimes.com/gst/abstract.html?res=9D01E4D8173DE53BBC4053DFB1668383649EDE)*
4. (Simple)AND/OR problem: linearly separable?
  - Yep
5. (Simple)XOR problem: linearly separable?
  - Nope
  - 어떻게 선을 그어도 100%가 안됨(50% 밖에 안됨)
6. Perceptrons(1969) by Marvin Minsky, founder of the MIT AI Lab
  - We need to use MLP, multilayer perceptrons(multilayer neural nets)
  - **No one on earth had found a viable way to train** MLPs good enough to learn such simple functions
  - The first depression on neural nets
  - *[for more information](http://cs231n.github.io/convolutional-networks/)*
7. Backpropagation(1974, 1982 by Paul Werbos, 1986 by **Hinton**)
  - Hinton's rediscovery로 neural nets 재조명 됨
  - *[for more information](https://devblogs.nvidia.com/parallelforall/inference-next-step-gpu-accelerated-deep-learning/)*
8. Convolutional neural networks
  - 고양이를 고정시켜놓고 그림을 보게 한 다음에 시신경의 뉴런이 어떻게 동작하는지를 봄
  - 그림의 형태에 따라 일부의 뉴런이 반응, 다른 그림이면 다른 일부의 뉴런이 반응
  - 따라서, 신경망 세포 전체가 반응하는 것이 아니라 일부가 각자 담당하는 것들이 있고 이 일부들이 모여 전체를 조직한다고 생각(by LeCun 교수)
  - At some point in the late 1990s, one of these systems was reading 10 to 20% of all the checks in the US [LeNet-5, LeCun 1980]
  - 현재까지도 굉장히 잘 동작(90% 정도는 이 기계를 통해 읽혀짐)
  - ex) Terminator2(1991)
9. A Big Problem
  - Backpropagation just did not work well for normal neural nets with many layers
  - Other rising machine learning algorithms: SVM, RandomForest, etc
  - 1995 "Comparison of learning algorithms for handwritten digit recognition" by LeCun et al. found that this new approach worked better
  - The second depression on neural nets
  - 몇 개의 layer는 학습 가능, but 여러 개의 layer는 학습 불가
  - *[for more information](http://neuralnetworksanddeeplearning.com/chap6.html)*
10. CIFAR
  - Canadian institute for advanced research(CIFAR)
  - CIFAR encourages basic research without direct application, was what motivated Hinton to move to Canada in 1987, and funded his work afterward
  - in 2006, Hinton, Simon Osindero, and Yee-Whye Teh published, "A fast learning algorithm for deep belief nets"
  - Yoshua Bengio et al. in 2007 with "Greedy Layer-Wise Training of Deep Networks"
11. Breakthrough in 2006 and 2007 by Hinton and Bengio
  - Neural networks with many layers really could be trained well, if the weights are initialized in a clever way rather than randomly
  - Deep machine learning methods are more efficient for difficult problems than shallow methods
  - Rebranding to Deep Nets, Deep Learning
12. IMAGENET Large Scale Visual Recognition Challenge
  - 사람들이 관심을 갖게 된 결정적인 계기
  - Neural networks that can explain photos
13. **Deep API Learning**
  - API 자동 예측
  - Copy a file and save it to your destination path -> 시스템이 자동적으로 어떤 API를 써야하는지 알려주어서 변수만 채워넣으면 되도록 만들어줌
14. Speech recognition errors
  - noise가 많은 사람의 말을 90% 정도 알아들을 수 있음
15. Geoffrey Hinton's Summary of findings up to Today
  - Our labeled datasets were thousands of times too small
  - Our computers were millions of times too slow
  - We initialized the weights in a stupid way
  - We used the wrong type of non-linearity
16. Why should I care?
  - I am not a researcher, not a computer scientist!
  - Do you have data?, Do yo sell something?, Are doing any business?
    + 이 물음들 중 하나라도 해당한다면, 유용하게 사용 가능
  - ex1) Youtube 자막 기능: 사람이 입력하는 것이 아니라 듣고 나오는건데 굉장히 정확
  - ex2) Facebook -> 관심갈만한 피드들만 올라옴
  - ex3) Google -> 내가 클릭할만한 문서들을 학습을 통해 예측하여 보여줌
  - ex4) 넷플릭스 추천 시스템, 아마존 추천 시스템
  - ex5) 가게도 딥러닝을 통해 학습시키면 앞에 어떤 것을 진열해야할지를 알아내어 매출을 올릴 수 있음
17. Why now?
  - Students/Researchers
    + Not too late to be a world expert
    + Not too complicated(mathematically)
  - Practitioner
    + Accurate enough to be used in practice
    + Many ready-to-use tools such as TensorFlow
    + Many easy/simple programming languages such as Python
  - After all, it is fun!

###### Laboratory
1. Simple 1D Array and Slicing
  ```python
  t = np.array([0., 1., 2., 3., 4., 5., 6.])
  pp.pprint(t)
  print(t.ndim) # rank
  print(t.shape)  # shape
  print(t[0], t[1], t[-1])
  print(t[2:5], t[4:-1]) # slicing
  print(t[:2], t[3:])
  ```
  ```
  result:
  array([0., 1., 2., 3., 4., 5., 6.])
  1
  (7,)
  0.0 1.0 6.0
  [2. 3. 4.] [4. 5.]
  [0. 1.] [3. 4. 5. 6.]
  ```
2. 2D Array
  ```python
  t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
  pp.pprint(t)
  print(t.ndim) # rank
  print(t.shape)  # shape
  ```
  ```
  result:
  array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
  2
  (4,3)
  ```
3. Shape, Rank, **Axis**
  ```python
  t = tf.constant([1,2,3,4])  # rank=1
  tf.shape(t).eval()
  ```
  ```
  result:
  array([4], dtype=int32)
  ```
  ```python
  t = tf.constant([[1,2], [3,4]])  # rank=2(angle bracket(=[])의 개수)
  tf.shape(t).eval()
  ```
  ```
  result:
  array([2,2], dtype=int32)
  ```
  ```python
  t = tf.constant([[[[1,2,3,4], [5,6,7,8], [9,10,11,12]], [[13,14,15,16], [17,18,19,20], [21,22,23,24]]]])  # rank=4 axis 0부터 안쪽으로 갈수록 큰 값(마지막은 -1이라고도 함)
  tf.shape(t).eval()
  ```
  ```
  result:
  array([1,2,3,4], dtype=int32)
  ```
4. Matmul vs Multiply
  ```python
  matrix1 = tf.constant([[1., 2.], [3., 4.]]) # rank=2, shape=[2,2]
  matrix2 = tf.constant([[1.], [2.]]) # rank=2, shape=[2,1]
  print("Matrix 1 shape ", matrix1.shape)
  print("Matrix 2 shape ", matrix2.shape)

  tf.matmul(matrix1, matrix2).eval() # ≠ (matrix1*matrix2).eval()
  ```
5. Broadcasting
  - shape이 다르더라도 연산을 할 수 있게 해주는 것
  - But 되도록이면 shape 맞춰서 연산하는 것이 좋음
  ```python
  matrix1 = tf.constant([[1., 2.]])
  matrix2 = tf.constant(3.)
  (matrix1+matrix2).eval()
  ```
  ```
  result:
  array([[4., 5.]], dtype=float32)
  ```
  ```python
  matrix1 = tf.constant([[1., 2.]])
  matrix2 = tf.constant([3., 4.])
  (matrix1+matrix2).eval()
  ```
  ```
  result:
  array([[4., 6.]], dtype=float32)
  ```
  ```python
  matrix1 = tf.constant([[1., 2.]])
  matrix2 = tf.constant([[3.], [4.]])
  (matrix1+matrix2).eval()
  ```
  ```
  result:
  array([[4., 5.], [5., 6.]], dtype=float32)
  ```
6. Reduce mean
  ```python
  tf.reduce_mean([1,2], axis=0).eval()  # int인지 float인지 주의
  ```
  ```
  result:
  1
  ```
  ```python
  x = [[1., 2.],[3., 4.]]  # 보통 float
  tf.reduce_mean(x).eval()
  ```
  ```
  result:
  2.5
  ```
  ```python
  tf.reduce_mean(x, axis=0).eval()
  ```
  ```
  result:
  array([[2., 3.], dtype=float32)
  ```
  ```python
  tf.reduce_mean(x, axis=1).eval)
  ```
  ```
  result:
  array([1.5, 3.5], dtype=float32)
  ```
  ```python
  tf.reduce_mean(x, axis=-1).eval()
  ```
  ```
  result:
  array([1.5, 3.5], dtype=float32)
  ```
7. Reduce sum
  ```python
  x = [[1., 2.], [3., 4.]]
  tf.reduce_sum(x).eval()
  ```
  ```
  result:
  10.0
  ```
  ```python
  tf.reduce_sum(x, axis=0).eval()
  ```
  ```
  result:
  array([4., 6.], dtype=float32)
  ```
  ```python
  tf.reduce_sum(x, axis=-1).eval()
  ```
  ```
  result:
  array([3., 7.], dtype=float32)
  ```
  ```python
  tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()
  ```
  ```
  result:
  5.0
  ```
8. Argmax
  - max 값이 있는 위치를 구하는 것
  ```python
  x = [[0,1,2], [2,1,0]]
  tf.argmax(x, axis=0).eval()
  ```
  ```
  result:
  array([1,0,0])
  ```
  ```python
  tf.argmax(x, axis=1).eval()
  ```
  ```
  result:
  array([2,0])
  ```
  ```python
  tf.argmax(x, axis=-1).eval()
  ```
  ```
  result:
  array([2,0])
  ```
9. **Reshape**
  ```python
  t = np.array([[[0,1,2], [3,4,5]], [[6,7,8], [9,10,11]]])
  t.shape
  ```
  ```
  result:
  (2,2,3)
  ```
  ```python
  tf.reshape(t, shape=[-1,3]).eval() # 안쪽 값은 거의 같게 두고 바깥쪽만 변경하여 큰 형태만 변하게 됨
  ```
  ```
  result:
  array([[0,1,2], [3,4,5], [6,7,8], [9,10,11]])
  ```
  ```python
  tf.reshape(t, shape=[-1,1,3]).eval()
  ```
  ```
  result:
  array([[[0,1,2]], [[3,4,5]], [[6,7,8]], [[9,10,11]]])
  ```
  - squeeze
  ```python
  tf.squeeze([[0], [1], [2]]).eval()
  ```
  ```
  result:
  array([0,1,2], dtype=int32)
  ```
  - expand
  ```python
  tf.expand_dims([0,1,2], 1).eval()
  ```
  ```
  result:
  array([[0], [1], [2]], dtype=int32)
  ```
10. One hot
  ```python
  tf.one_hot([[0], [1], [2], [0]], depth=3).eval()  # rank 1개 자동적으로 expand
  ```
  ```
  result:
  array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]], [[1., 0., 0.]]], dtype=float32)
  ```
  ```python
  t = tf.one_hot([[0], [1], [2], [0]], depth=3)
  tf.reshape(t,shape=[-1,3]).eval() #rank를 expand 시키고 싶지 않으면 reshape 하면 됨
  ```
  ```
  result:
  array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=float32)
  ```
11. Casting
  ```python
  tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
  ```
  ```
  result:
  array([1,2,3,4], dtype=int32)
  ```
  ```python
  tf.cast([True, False, 1==1, 0==1], tf.int32).eval()
  ```
  ```
  result:
  array([1,0,1,0], dtype=int32)
  ```
12. Stack
  ```python
  x = [1,4]
  y = [2,5]
  z = [3,6]

  # pack along first dim.
  tf.stack([x,y,z]).eval()
  ```
  ```
  result:
  array([[1,4], [2,5], [3,6]], dtype=int32)
  ```
  ```python
  tf.stack([x,y,z], axis=1).eval()  # axis, stack relation 유의(axis=0,-1로도 바꿔보면서 어떤 연관성이 있는지 확인)
  ```
  ```
  result:
  array([[1,2,3], [4,5,6]], dtype=int32)
  ```
13. Ones and Zeros like
  ```python
  x = [[0,1,2], [2,1,0]]
  tf.ones_like(x).eval()
  ```
  ```
  result:
  array([[1,1,1], [1,1,1]], dtype=int32)
  ```
  ```python
  tf.zeros_like(x).eval()
  ```
  ```
  result:
  array([[0,0,0], [0,0,0]], dtype=int32)
  ```
14. Zip
  ```python
  for x,y in zip([1,2,3], [4,5,6]):
    print(x,y)
  ```
  ```
  result:
  1 4
  2 5
  3 6
  ```
  ```python
  for x,y,z in zip([1,2,3], [4,5,6], [7,8,9]):
    print(x,y,z)
  ```
  ```
  result:
  1 4 7
  2 5 8
  3 6 9
  ```
