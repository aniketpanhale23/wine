from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark import SparkContext, SparkConf
def func(line):
    v = [float(z) for z in line.split(';')]
    return LabeledPoint(v[11], v[0:10])
c = SparkConf().setAppName("winequality")
s = SparkContext("local",conf=c)
t = s.textFile("s3://winequalitynew/TrainingDataset.csv")
h = t.first()
r = t.filter(lambda z: z != h)
tnew = r.map(func)
dnew = s.textFile("s3://winequalitynew/ValidationDataset.csv")
h = dnew.first()
r = dnew.filter(lambda z: z != h)
dnew2 = r.map(func)
m = RandomForest.trainClassifier(tnew, numClasses=11, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
p = m.predict(dnew2.map(lambda z: z.features))
lpre = dnew2.map(lambda lp: lp.label).zip(p)
me = MulticlassMetrics(lpre)
scr = me.fMeasure()
scr
m.save(s, "s3://winequalitynew/winequality")