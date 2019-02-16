#### Lecture14
## TensorFlow GPU @AWS Spot Instances without losing data

1. Spot Instances
  - 가격 저렴
  - instance 항상 보장되지 않아서 사용하는 도중에 데이터가 사라질 수 있음
2. GPU
  - specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation if images in a frame buffer intended for output to a display
3. AWS G2 and P2 Instances
  - 4G -> 12G
  - P2가 최대 3배까지 속도가 빨라짐
  - but too expensive
4. On-Demand vs Spot Instances
  - On-Demand: 처음부터 주어지고 항상 사용할 수 있음
  - Spot Instances: AWS가 필요하면 가져갈 수 있음
    + added when bid > market
5. Request spot instances and save $
  - spot request -> GPU compute -> request spot instances ready(takes 2 ~ 3 min) -> work with screen
    + work with screen
      * ssh
      * screen: open a new screen
      * python train.py; ..; echo "Done" | mail -s "Finished" oooooo@gmail.com
      * ctrl-a d (to exit screen)
      * screen -r: attach the screen
6. Solution: Spot Instance + EBS Volume
  - do not delete on termination
7. workflow
  - create a spot instance(from a AMI)
    + do not delete the main disk on termination
  - run TF tasks; sudo shutdown now
    + run(in screen) and save results on the volume
    + terminate the instance(save $)
  - create a snapshot and AMI from the leftover volume
  - create a new spot using the AMI
    + delete old AMIs
8. Process
  - create AMI - TF/Cuda -> request spot instances using AMI(check price -> don't delete the volume!) -> run TF(with screen) and enjoy! -> spot instance termination + EBS volume -> create snapshot from the volume -> create AMI from the snapshot -> create new spot instance using the AMI
