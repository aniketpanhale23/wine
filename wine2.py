from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
def func(line):
    v = [float(z) for z in line.split(';')]
    return LabeledPoint(v[11], v[0:10])
import findspark
findspark.init()
c = SparkConf().setAppName("winequality")
s = SparkContext(conf=c)
m = RandomForestModel.load(s, "/home/ubuntu/winequality")
t = s.textFile(sys.argv[1])
h = t.first()
r = t.filter(lambda z: z != h)
tnew = r.map(func)
p = m.predict(tnew.map(lambda x: x.features))
lpre = tnew.map(lambda lp: lp.label).zip(p)
me = MulticlassMetrics(lpre)
scr = me.fMeasure()
print("#############")
print("F1 score = ")
print(scr)
print("#############")