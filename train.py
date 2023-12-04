import os
import warnings

import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pyspark.sql.functions as F
import seaborn as sns  # good visualizing
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.types import FloatType

warnings.filterwarnings('ignore')

# Initialize spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

path = "datasets/TrainingDataset.csv"
validation_path = "datasets/ValidationDataset.csv"
data = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(path)
validation_data = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(validation_path)

data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()
print(validation_data.head())

feature_list = ['fixed acidity',
 'volatile acidity',
 'citric acid',
 'residual sugar',
 'chlorides',
 'free sulfur dioxide',
 'total sulfur dioxide',
 'density',
 'pH',
 'sulphates',
 'alcohol']

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=feature_list, outputCol=vector_col)
df_vector = assembler.transform(data).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
corr_matrix = matrix.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = feature_list, index=feature_list)

train_assembler = VectorAssembler(inputCols=feature_list,outputCol="features")
df_train = assembler.transform(data)

# Logistic Regression model
lr = LogisticRegression(featuresCol="corr_features", labelCol='quality')
lrModel = lr.fit(df_train)

# Save model
lrModel.save("/home/ec2-user/models/logisticregression/")

validation_assembler = VectorAssembler(inputCols=feature_list,outputCol="features")
df_validation = assembler.transform(validation_data)

lr_predictions = lrModel.transform(df_validation)
lr_predictions.select('quality', 'rawPrediction', 'prediction', 'probability').show()

lr_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
lr_f1 = lr_evaluator.evaluate(lr_predictions)
print("Logistic Regression F1 score = %s" % (lr_f1))

# Random Forest model
rf = RandomForestClassifier(featuresCol = "corr_features", labelCol = 'quality')
rfModel = rf.fit(df_train)

# Save model
rfModel.save("/home/ec2-user/models/randomforest/")

rf_predictions = rfModel.transform(df_validation)
rf_predictions.select('quality', 'rawPrediction', 'prediction', 'probability').show()

rf_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
rf_f1 = rf_evaluator.evaluate(rf_predictions)
print("RandomForest F1 score = %s" % (rf_f1))


f1_score_path = "f1_score.txt"
with open(f1_score_path, 'w') as f:
    f.write("----------Results----------")
    f.write("\nLogistic Regression F1 Score: {}".format(lr_f1))
    f.write("\nRandomForest F1 Score: {}\n".format(rf_f1))
