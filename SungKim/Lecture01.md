#### Lecture01
## Machine Learning Basics

###### Theory
1. Basic Concepts
  - what is ML?
  - what is learning?
    + supervised
    + unsupervised
  - what is regression?
  - what is classification?
2. Machine Learning(ML)
  - limitations of explicit programming(spam filter, automatic driving)
  - 일종의 소프트웨어
  - explicit 하게 프로그램 할 수 없는 경우들
  - 일일이 프로그래밍 하지 말고 어떤 자료나 현상들에서 배워서 스스로하게 하는 것(학습해서 배우는 능력을 갖게 하는 것)
3. Supervised/Unsupervised Learning
  - supervised learning: learning with labeled examples - training set
    + 정해져있는 데이터를 가지고 학습하는 것
    + 이미지를 보고 그것이 고양이인지 강아지인지 알아내는 프로그램 -> cat이라는 label이 달린 고양이 사진을 주고 '이게 고양이야' 라고 알려주어서 학습하게 함
  - unsupervised learning: un-labeled data(google news grouping, word clustering)
    + 알아서 보고 유사한 것끼리 모아주는 것
    + 데이터를 보고 스스로 학습한다
4. Supervised Learning
  - most common problem type in ML(image labeling, email spam filter, predicting exam score)
  - training data set
    + 특정 데이터를 가지고 학습하게 할 때 주어지는 데이터 셋(label을 가진 데이터)
    + x, y 데이터를 준다고 할 때, y가 regression이나 classification이 됨(result)
  - AlphaGo
    + 기존에 사람들이 바둑판에서 바둑을 둔 것들을 학습
    + 학습한 것을 가지고 상대방이 바둑알을 두었을 때, '그럼 난 여기에 두는 것이 좋겠다' 라고 판단하게 됨
  - types of supervised learning
    + predicting final exam score based on time spent -> **regression**
    + pass/non-pass based on time spent -> binary **classification**(둘 중 하나를 고르는 것이므로)
    + letter grade(A,B,C,E and F) based on time spent -> multi-label **classification**

###### Laboratory
1. Tensorflow Basics
  - Google에서 만든 오픈소스 라이브러리
  - 조사 결과 accumulated GitHub metrics, growth over past three months 등의 분야에서 다 1등
  - 온라인 상에 많은 자료가 있음
  - open source software library for numerical computation using data flow graphs
  - python을 가지고 프로그래밍을 할 수 있음
2. Data Flow Graph
  - nodes in the graph represent mathematical operations
  - edges represent the multidimensional data arrays(tensors) communicated between them
3. Check Installation and Version
  ```python
  python3
  import tensorflow as tf
  tf.__version__
  ```
4. Tensorflow - Hello World!
  ```python
  # This op is added as a node to the default graph
  hello = tf.constant("Hello, TensorFlow!")
  # seart a TF session
  sess = tf.Session()
  # run the op and get result
  print(sess.run(hello))
  ```
  ```
  result:
  b'Hello, TensorFlow!' (b'String'에서 b는 bytes literals을 의미)
  ```
5. Computational Graph  
  - 1) build graph(tensors)
  ```python
  node1 = tf.constant(3.0, tf.float32)
  node2 = tf.constant(4.0)  # also tf.float32 implicitly
  node3 = tf.add(node1, node2) # node3 = node1 + node2
  print("node1: ", node1, "node2: ", node2)
  print("node3: ", node3)
  ```
  ```
  result:
  node1: Tensor 어쩌구 node2: Tensor 어쩌구 node3: Tensor 어쩌구 (원하는 결과값이 나오지 않고 '해당 노드는 Tensor이다' 라는 내용으로 결과값이 나옴)
  ```
  - 2) feed data and run graph(operation)
  ```python
  sess = tf.Session()
  print("sess.run(node1, node2): ", sess.run([node1, node2]))
  print("sess.run(node3): ", sess.run(node3))
  ```
  - 3) update variables
  ```
  result:
  sess.run(node1, node2): [3.0, 4.0]
  sess.run(node3): 7.0
  ```
6. Tensorflow Mechanics
  ```
  1) build graph using Tensorflow operations
    node를 지정할 때 placeholder로 만들 수 있다
  2) feed data and run graph(operation) -> sess.run(op)
    sess.run(op, feed_dict = {x:x_data})  # placeholder로 넘겨줌
  3) update variables in the graph(and return values)
  ```
7. Placeholder
  - 실행시키는 단계에서 값들을 던져주고싶다
  - node를 placeholder라는 특별한 노드로 만들어줌  
  ```python
  a = tf.placeholder(tf.float32)
  b = tf.placeholder(tf.float32)
  adder_node = a+b # + provides a shortcut for tf.add(a,b)
  print(sess.run(adder_node, feed_dict = {a:3, b:4.5})) # 값을 가지고 그래프를 실행시켜라
  print(sess.run(adder_node, feed_dict = {a:[1,3], b:[2,4]})) # n개의 값을 넘겨줄 수 있음
  ```
  ```
  result:
  7.5
  [3. 7.] (a+b니까 1+2=3, 3+4=7이 됨)
  ```
8. Everything is **Tensor**
  - ranks, shapes, and types
  - ranks
    + 0: scalar(s = 483)
    + 1: vector(v = [1.1, 2.2, 3.3])
    + 2: matrix(m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  - shapes
    + rank0: []
    + rank1: [D0]
    + rank2: [D0, D1]
  - types
    + 보통 tf.float32, tf.int32를 제일 많이 사용
  - ex) t = [[1,2,3], [4,5,6], [7,8,9]] -> shape은 [3, 3]
