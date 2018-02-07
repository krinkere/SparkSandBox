from pyspark import SparkContext, SQLContext
import findspark
# https://www.analyticsvidhya.com/blog/2016/09/comprehensive-introduction-to-apache-spark-rdds-dataframes-using-pyspark/
# https://www.nodalpoint.com/spark-dataframes-from-csv-files/

findspark.init(spark_home='C:/spark/spark-2.2.0-bin-hadoop2.7/')
# First thing that a Spark program does is create a SparkContext object, which tells Spark how to access a cluster.
sc = SparkContext()

# data = range(1, 1000)
# rdd = sc.parallelize(data)
#
# print(rdd.collect())
#
# # It will print first 2 elements of rdd
# print(rdd.take(2))
#
# data = ['Hello', 'World', 'This', 'is', 'Test']
# Rdd = sc.parallelize(data)
#
# Rdd1 = Rdd.map(lambda x: (x, 1))
# print(Rdd1.collect())

sqlContext = SQLContext(sc)

train = sqlContext.read.load(format="com.databricks.spark.csv", path='train.csv', header=True, inferSchema=True)
test = sqlContext.read.load(format="com.databricks.spark.csv", path='test.csv', header=True, inferSchema=True)

print("Schema")
print(train.printSchema())
print("First 10")
print(train.head(10))
print("Count")
print(train.count())

# We can check number of not null observations in train and test by calling drop() method. By default, drop() method
# will drop a row if it contains any null value. We can also pass ‘all” to drop a row only if all its values are null.
train.na.drop().count()
test.na.drop('any').count()

# Here, I am imputing null values in train and test file with -1.
train = train.fillna(-1)
test = test.fillna(-1)

print(train.describe().show())
print(train.select('User_ID').show())

print(train.select('Product_ID').distinct().count(), test.select('Product_ID').distinct().count())

diff_cat_in_train_test=test.select('Product_ID').subtract(train.select('Product_ID'))
# For distinct count
print(diff_cat_in_train_test.distinct().count())

from pyspark.ml.feature import StringIndexer
plan_indexer = StringIndexer(inputCol='Product_ID', outputCol='product_ID_hotencode')
labeller = plan_indexer.fit(train)
Train1 = labeller.transform(train)
Test1 = labeller.transform(test)
print(Train1.show())

from pyspark.ml.feature import RFormula
formula = RFormula(formula="Purchase ~ Age+ Occupation +City_Category+Stay_In_Current_City_Years+Product_Category_1+Product_Category_2+ Gender",featuresCol="features",labelCol="label")

t1 = formula.fit(Train1)
train1 = t1.transform(Train1)
test1 = t1.transform(Test1)
print(train1.show())

print(train1.select('features').show())
print(train1.select('label').show())

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor()

# After creating a model rf we need to divide our train1 data to train_cv and test_cv for cross validation.
# Here we are dividing train1 Dataframe in 70% for train_cv and 30% test_cv.
(train_cv, test_cv) = train1.randomSplit([0.7, 0.3])

# Now build the model on train_cv and predict on test_cv. The results will save in  predictions.
model1 = rf.fit(train_cv)
predictions = model1.transform(test_cv)
print(predictions.show())

# Lets evaluate our predictions on test_cv and see what is the mean square error.
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
import numpy as np
print(np.sqrt(mse), mse)

# Now, we will implement the same process on full train1 dataset.
model = rf.fit(train1)
predictions1 = model.transform(test1)

df = predictions1.selectExpr("User_ID as User_ID", "Product_ID as Product_ID", 'prediction as Purchase')

# Now we need to write the df in csv format for submission.
df.toPandas().to_csv('submission.csv')
