from pyspark import SparkContext
import findspark
# https://www.analyticsvidhya.com/blog/2016/10/using-pyspark-to-perform-transformations-and-actions-on-rdd/

findspark.init(spark_home='C:/spark/spark-2.2.0-bin-hadoop2.7/')
sc = SparkContext()

rdd = sc.textFile("blogtexts")

print(rdd.take(5))


# Convert all words in a rdd to lowercase and split the lines of a document using space.
def Func(lines):
    lines = lines.lower()
    lines = lines.split()
    return lines

# output is not flat (it’s a nested list). So for getting the flat output, we need to apply a transformation which will
#       flatten the output
rdd1 = rdd.flatMap(Func)
print(rdd1.take(5))

# The transformation “flatMap” will help here: The “flatMap” transformation will return a new RDD by first applying a
#       function to all elements of this RDD, and then flattening the results. This is the main difference between the
#       “flatMap” and map transformations.
rdd1 = rdd.flatMap(Func)
print(rdd1.take(5))

# Next, I want to remove the words, which are not necessary to analyze this text. We call these words as “stop words”;
# Stop words do not add much value in a text. For example, “is”, “am”, “are” and “the” are few examples of stop words.
stopwords = ['is', 'am', 'are', 'the', 'for', 'a']
rdd2 = rdd1.filter(lambda x: x not in stopwords)
print(rdd2.take(10))

# After getting the results, we want to group the words based on which letters they start with. For example, suppose
# I want to group each word based on first 3 characters.
rdd3 = rdd2.groupBy(lambda w: w[0:3])
print([(k, list(v)) for (k, v) in rdd3.take(1)])

# What if we want to calculate how many times each word is coming in corpus
# map stage. Assign each element value of 1
rdd3_mapped = rdd2.map(lambda x: (x, 1))
# Manually group all elements and them sum them up
rdd3_grouped = rdd3_mapped.groupByKey()
print(list((j[0], list(j[1])) for j in rdd3_grouped.take(5)))
rdd3_freq_of_words = rdd3_grouped.mapValues(sum).map(lambda x: (x[1], x[0])).sortByKey(False)
print(rdd3_freq_of_words.take(10))
# or use reduce
print(rdd3_mapped.reduceByKey(lambda x, y: x+y).map(lambda x: (x[1], x[0])).sortByKey(False).take(10))

# What if I want to work with samples instead of full data
# “sample” transformation helps us in taking samples instead of working on full data. The sample method will return a
# new RDD, containing a statistical sample of the original RDD. We can pass the arguments insights as the sample
# operation:
#
# “withReplacement = True” or False (to choose the sample with or without replacement)
# “fraction = x” ( x= .4 means we want to choose 40% of data in “rdd” ) and “seed” for reproduce the results.
rdd3_sampled = rdd2.sample(False, 0.4, 42)
print(len(rdd2.collect()), len(rdd3_sampled.collect()))

# How to do a union of RDDs
# Please note that duplicate items will not be removed in the new RDD
sample1 = rdd2.sample(False, 0.2, 42)
sample2 = rdd2.sample(False, 0.2, 42)
union_of_sample1_sample2 = sample1.union(sample2)
print(len(sample1.collect()), len(sample2.collect()), len(union_of_sample1_sample2.collect()))

# How to calculate distinct elements in a RDD
rdd3_distinct = rdd2.distinct()
print(len(rdd3_distinct.collect()))

# Count the number of elements in RDD
print(rdd2.count())

# max, min, sum, variance and stdev
num_rdd = sc.parallelize(range(1 , 1000))
print(num_rdd.max(), num_rdd.min(), num_rdd.sum(), num_rdd.variance(), num_rdd.stdev())

# Dealing with nulls - so if you have datasource and it contains nulls, if you were to do take(n) on it, it would
#       break with some cryptic error. Mind that collect() would work still. In order to filter those out:
# data = sc.textFile(save_dir + 'palm_status_paths_with_art_class/').map(lambda x: x)
# data_filtered = data.filter(lambda x: x)
# print data_filtered.take(20)




