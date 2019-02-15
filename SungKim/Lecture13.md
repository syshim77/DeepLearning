#### Lecture13
## TensorFlow GPU @AWS

1. Deep Network
  - takes a long time for training
    + many forward/backward propagation and weight updates
    + many metrics multiplications
  - very quick for testing and use in practice
    + one simple forward propagation
2. GPU
  - graphics processing unit(= visual processing unit(VPU))
3. amazon web services
  - 사용해서 TensorFlow를 사용하는 법을 알아볼 예정
4. EC2 Console: Oregon
  - launch instance -> ubuntu, GPU, 12G or more -> key to access the server -> EC2: Create an instance -> it's ready to ssh! -> requires CUDA and CuDNN(+ add path) -> reuse ami-9e39dcf3(N.Virginia), ami-38f60658(oregon) -> creating TensorFlow device(/gpu:0)
    + add path
      * export PATH = /usr/local/cuda/bin:$PATH
      * export LD_LIBRARY_PATH = /usr/local/cuda/lib64:$LD_LIBRARY_PATH
  - GPU 가지고 있는 것과 아닌 것의 차이는 25배 정도 남
5. spot instances
  - EC2 Console: spot instances -> pricing history -> price bidding -> bill, bill, bill! -> check, stop, and terminate -> cloud watch -> shutdown after training
    + cloud watch
      * stop when CPU utilization ≤ 0.3
    + shutdown after training
    ```
    $ screen
    $ sudo -i
    # python train.py; shutdown -h now
    ```
6. For more Information of [this Lecture](http://hunkim.github.io/ml/)
