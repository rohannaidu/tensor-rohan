
# **Code Example** 
Code Example

For the sake of brevity, you should download docker and initialize a tensorflow environment. 
```
$ curl -sSL https://get.docker.com/ | sh

$ sudo docker run -it -p 8888:8888 tensorflow/tensorflow /bin/bash
```
You should restart your server if your sudo docker commands cannot execute.
You need to install git and nano. You can ignore this step if you already have it
```
$ sudo apt-get update

$ sudo apt-get install git nano

```
You can then clone this repository here 
```
$ cd home

$ git clone git://github.com/muhdamrullah/tensor-rohan

$ cd tensor-rohan

$ python loop_thru_eval.py ./evaluation_files/

```

... When you execute this command, you are evaluating the images in the evaluation folder and appending to a tf_submission.csv file. 
Make sure you clean the CSV file if you are starting fresh.
