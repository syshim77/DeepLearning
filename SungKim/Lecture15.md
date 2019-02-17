#### Lecture15
## Google Cloud ML Examples

1. Local TensorFlow tasks
  - TensorFlow task
  - local disk
2. Process
  - setup your environment -> verifying your environment -> initializing your Cloud ML project -> setting up your cloud storage bucket
    + setup environment
      * local: MAC/LINUX
      * cloud shell(google cloud console -> command -> path)
      * docker container
3. Google cloud commands
  - gcloud: command-line interface to google cloud platform
  - gsutil: command-line interface to google cloud storage
4. Example
  - [Example git repository](https://github.com/hunkim/GoogleCloudMLExamples.git)
  - simple multiplication -> run locally -> run on Cloud ML
  - machine learning console -> jobs -> jobs/task -> jobs/task7/logs
5. Input example
  - CSV file reading -> run locally
  - Cloud ML TensorFlow tasks
    + setting and file copy -> google storage -> run on Cloud ML
    + console -> jobs -> logs
6. Output example
  - TensorFlow saver -> local run -> configuration
  - create/check the output folder -> run on Cloud ML
  - job completed
7. With great power comes great responsibility
  - check your bills!
8. Next
  - Cloud ML deploy
  - hyper-parameter tuning
  - distributed training tasks
