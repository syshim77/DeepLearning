#### Lecture03
## How to minimize cost

###### Theory
1. Hypothesis and Cost
  - 가지고 있는 데이터를 통해서 W, b 찾자
2. Simplified Hypothesis
  - H(x) = Wx
  - cost(W) = 1/mΣ(Wx(i)-y(i))2
3. What cost(W) looks like?
  - W=1, cost(W)=? -> (1*1-1)2+(1*2-2)2+(1*3-3)2 = 0
  - W=0, cost(W)=? -> 1/3{(0*1-1)2+(0*2-2)2+(0*3-3)2} = 약 4.67
  - W=2, cost(W)=? -> 약 4.67
  - 값을 구해서 함수를 그려보면 2차 방정식 형태가 됨
4. How to minimize cost?
  - gradient descent algorithm(경사를 따라 내려가는 알고리즘)
    + minimize cost function
    + gradient descent is used many minimization problems
    + for a given cost function, cost(W,b), it will find W,b to minimize cost
    + it can be applied to more general function: cost(w1,w2,...)
  - How it works? How would you find the lowest point?
    + 산 위에서 내려갈 때 경사가 있는 쪽으로 발을 딛어서 내려가게 됨 = gradient descent algorithm
    + 해당 위치에서 경사도를 보고 경사도를 따라서 한발짝 움직임
    + 경사도가 0인 위치에 도달하면 그 자리에 머물게 됨
5. How it works?
  - start with initial guesses
    + start at 0,0(or any other value)
    + keeping changing W and b a little bit to try and reduce cost(W,b)
  - each time you change the parameters, you select the gradient which reduces cost(W,b) the most possible
  - repeat
  - do so until you converge to a local minimum
  - has an interesting property
    + where you start can determine which minimum you end up
    + 가끔 예외가 있지만 어떤 점에서 시작을 하던간에 항상 최저점에 도달할 수 있다
  - 경사도는 미분을 이용
6. Formal Definition
  - cost(W) 식에서 1/m -> 1/2m으로 바꾸어서 미분하는 것이 계산하기 편함
  - W := W - α*(cost(W) 미분한 것)(α: learning 예시)
  - **W := W - α*1/mΣ(Wx(i)-y(i))**
  - 위 수식을 기계적으로 적용만 시키면 바로 cost function을 최소화하는 W를 찾게 되고, 학습을 통해 모델을 만든다고 할 수 있음
7. Convex Function
  - 밥그릇 형태의 그래프
  - cost, W, b로 만든 3차원 그래프가 이 형태로 나타나면 항상 답을 찾는다는 것이 보장됨
  - 따라서, 반드시 cost function의 모양이 convex function이 되는지 확인해야함
  - 그러면 무조건 linear algorithm을 이용하여, 즉 gradient descent algorithm을 사용하여 답을 구할 수 있음

###### Laboratory
1. Simplified Hypothesis
  - b 생략  
  **cost = tf.reduce_mean(tf.float32)**  
2. Gradient Descent
  - 미분 = 한 점에서의 기울기
  - 기울기를 보고 +면 W가 -값이 나와서 더 작은 값(-방향)으로 이동, -면 W가 +값이 나와서 더 큰 값(+방향)으로 이동  
  // 수식을 수기로 Tensorflow에 적은 코드  
  // minimize: Gradient Descent using derivative:  
  W -= learning_rate * derivative  
  learning_rate = 0.1  
  gradient = tf.reduce_mean((Wx - y) * x)  
  descent = W - (learning_rate * gradient)  
  // update를 실행시키게 되면 일련의 과정들이 행해지게 됨  
  **update = W.assign(descent)**  // W := 하는 과정, W 값이 바뀌게 됨  
  // optimizer를 설정해두면 미분하지 않아도 알아서 함  
  // minimize: gradient descent magic  
  **optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)**  
  train = optimizer.minimize(cost)  
3. Optional: compute_gradient and apply_gradient
  - 어떻게 하는건지 좀 더 이해하고 싶거나 graident를 좀 수정해서 임의로 저장하고 싶을 때(값을 더하던지 빼던지) 사용
  - 아래 코드에서 손으로 계산한거랑 컴퓨터가 계산한 값이 거의 같음  
  // 손으로 계산한 gradient랑 컴퓨터 gradient 결과가 같은지 확인  
  gradient = tf.reduce_mean((W*x-y)* x)* 2  // 수기로 계산한 gradient  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  
  // get gradients  
  gvs = optimizer.comppute_gradients(cost)  
  // apply gradients  
  apply_gradients = optimizer.apply_gradients(gvs)  // 이 예제에서는 minimize랑 똑같은 결과, 필요하면 여기에서 수정할 수 있음  
