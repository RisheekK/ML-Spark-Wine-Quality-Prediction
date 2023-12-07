# ML-Spark-Wine-Quality-Prediction
Link to DockerHub Repo: https://hub.docker.com/repository/docker/risheek811/ml-spark-wine-quality-prediction/general

# spark-MLlib-ec2-docker
This project will demonstrate how to run Apache Spark on multiple EC2 instances, connect to a Jupyter notebook, use Spark's MLlib to train machine learning models on cloud, create a docker container which can load the ML model and predict wine classifications.
## Creating a Spark Cluster on EC2
Assuming that you already have your AWS credentials and your AWS key-pair.
### Setting up the cluster
To create a spark cluster on EC2, we will use [flintrock](https://github.com/nchammas/flintrock). To install Flintrock:
```
pip3 install flintrock
```
Next, we will have to configure the Spark and Hadoop version to use in our cluster:
```
flintrock configure
```
We should also specify AWS key-pair name, and location in the configure file. We can also use the configure file to set the number of slaves required for our spark cluster.

Launch a Spark cluster with:
```
flintrock launch <cluster-name>
```
Once the cluster is successfully launched, we can login into the cluster:
```
flintrock login <cluster-name>
```
Now you will be logged into Spark MAster's EC2 instance.

### Adding dependancies for the project
We will install Jupyter notebook and its dependancines by installing Anaconda python distribution.
```
[SparkMaster] wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
[SparkMaster] sh Anaconda2-4.2.0-Linux-x86_64.sh
```
The installation will ask if you would like to set the path variables, if not, add the path variable to the ~/.bashrc
```
[SparkMaster] export PATH=/home/user/path/to/anaconda2/bin:$PATH
```

## Training on Spark 
We will have to add the dataset and the ```Train.ipynb``` file to the master and slave nodes. This can be done with:
```
scp -i /path/to/key/pair /path/to/file/on/local username@ec2.public.dns:/path/in/remote
```
 We can now run the Train.ipynb file. The ```Train.ipynb``` file saves the ML model as directories. We can then download the model files and reuse it.
 
 ## Contaninerization of ML model
We use the base image provided by Data Mechanics which already comes with Spark installed. We copy all the required files for prediction into the docker container. Then we run a ```pip3 install``` on our reqiurements.
The saved Docker image can be pulled from my repo:
```
docker pull wanderlust011999/spark-mllib-ec2-docker:latest
```
Once you have pulled the docker image, you can run as follows:
```
docker run -v /absolute/path/to/test.csv/file:/opt/spark/work-dir/ValidationDataset.csv --platform linux/amd64 --rm -i -t projppp56789 python3 Cloud_prediction.py
