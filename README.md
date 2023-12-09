# ML-Spark-Wine-Quality-Prediction
Link to DockerHub Repo: https://hub.docker.com/repository/docker/risheek811/ml-spark-wine-quality-prediction/general

## 1. Setting up AWS Environment and Instances
a. From the AWS Management Console, navigate to EC2 under services and create 5
instances - 4 for parallel mode training (cores), 1 for prediction (master)
b. Choose Amazon/Ubuntu Machine Image AMI
c. Choose instance type as t2.large, or any other larger instance type with multiple CPUs
d. Configure other settings, store the private key, add storage volume if needed, and
launch instances
e. SSH into each of those 4 instances for training using the private key, and configure
AWS credentials.

## 2. Installing Necessary Software on EC2 Instances
a. Update system packages, install python, Java OpenJDK (required for Apache Spark),
and install apache spark.
b. To install apache spark:
    i. wget https://downloads.apache.org/spark/spark-3.3.3/spark-3.3.3-binhadoop3.tgz
    ii. tar -zxvf spark-3.3.3-bin-hadoop3.tgz
    iii. sudo mv spark-3.3.3-bin-hadoop3 /home/ec2-user/spark
    iv. echo 'export SPARK_HOME=/home/ec2-user/spark' >> ~/.bashrc
    v. echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc
    vi. source ~/.bashrc
c. Install Docker:
    i. sudo amazon-linux-extras install docker -y
    ii. sudo service docker start
    iii. sudo usermod -a -G docker ec2-user
d. Log out of the instances and log in to reflect changes.

## 3. Dataset Preparation
a. Upload TrainingDataset.csv and ValidationDataset.csv onto the EC2 instances for
training.
b. To place any file from your local onto EC2 instance:
    i. scp -i Ec2_instance_key.pem path/to/your/file ec2-
    user@IPv4address:/home/ec2-user/path.
c. To place any file from EC2 instance onto your local, open terminal on your local and
type:
    i. scp -i Ec2_instance_key.pem ec2-
    user@IPv4address:path/to/your/file/on/ec2 destination/folder/in/local
d. You can also upload documents to your s3 storage on AWS and fetch them if you have
permission.

## 4. Parallel Training of Models using Apache Spark
a. Choose 1 of your 4 training instances as the master node, and the other 3 as worker
nodes.
b. Start Spark Cluster on the Master Node
    i. spark/sbin/start-master.sh
    ii. Once you run the above command, the master node is initialized. Obtain the
    Spark Master URL from the logs using cat command.
    iii. In the logs note for the line with the following url -> spark://<master-ip>:7077
c. Start Spark Worker on each worker instance (3 executors)
    i. spark/sbin/start-worker.sh <spark-master-node-url>
d. Create a script for Spark application with Spark session for training your models and
upload it onto any 1 of your worker instances – train.py
    i. The models are validated with the validation dataset, and saved in the models
    folder.
e. Run the Spark Application
    i. spark/bin/spark-submit --master spark://<master-ip>:7077 --numexecutors 3 --executor-cores 2 train.py
    Here, we have set 3 executors with 2 vCPUs each.
    ii. After the model trains, stop the Spark cluster.
1. On master Node – spark/sbin/stop-master.sh
2. On each worker node – spark/sbin/stop-worker.sh
f. Download the saved models to local and upload it to the prediction instance.

## 5. Configuring Prediction EC2 Instance
a. Repeat the same installation process to setup the Apache Spark in this instance.

## 6. Prediction Application
a. Upload TestDatset.csv to the prediction EC2 instance using scp.
b. Create a spark application on the EC2 instance that loads the saved model and use
TestDataset.csv for testing predictions – predict.py
c. Evaluate with F1 scores and write the results to a text file.

## 7. Running the Prediction Application on single EC2 Instance
a. Without Docker
    i. Initiate master node using start-master.sh
    ii. Run the application -> spark-submit predict.py
b. With Docker
    i. Create a Dockerfile specifying the installations, configurations and
    dependencies needed for executing predict.py
    ii. Create requirements.txt file for the dependencies
    iii. Build the docker image using the Dockerfile
1. docker build -t <docker-image-name>
2. docker run -d <docker-image-name>
3. To view the contents inside the docker
a. docker exec -it <first three letters of process id> /bin/bash
4. To view the container log
a. docker logs -f <first three letters of process id>
5. Get final output.

## pushing docker image to docker hub
![image](https://github.com/RisheekK/ML-Spark-Wine-Quality-Prediction/assets/86208506/c8a2d629-3976-4ae3-bad5-f0a839a1d745)

## docker file 
![image](https://github.com/RisheekK/ML-Spark-Wine-Quality-Prediction/assets/86208506/8ac5949d-a772-42b2-9eb7-2a4280a3e82e)

## Output of f1 scores of the classifiers on the Test Dataset
![image](https://github.com/RisheekK/ML-Spark-Wine-Quality-Prediction/assets/86208506/285c4413-472d-4a43-a77f-2ee2bfb73b22)
