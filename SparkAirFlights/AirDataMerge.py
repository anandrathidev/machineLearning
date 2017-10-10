from pyspark.sql.types import *
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import *
from pyspark.sql.functions import col


schema = StructType([StructField("Year",IntegerType(),True),
StructField("Month",IntegerType(),True),
StructField("DayofMonth",IntegerType(),True),
StructField("DayOfWeek",IntegerType(),True),
StructField("DepTime",StringType(),True),
StructField("CRSDepTime",IntegerType(),True),
StructField("ArrTime",StringType(),True),
StructField("CRSArrTime",IntegerType(),True), 
StructField("UniqueCarrier",StringType(),True), 
StructField("FlightNum",IntegerType(),True), 
StructField("TailNum",StringType(),True), 
StructField("ActualElapsedTime",StringType(),True), 
StructField("CRSElapsedTime",StringType(),True),
StructField("AirTime",StringType(),True),
StructField("ArrDelay",StringType(),True),
StructField("DepDelay",StringType(),True),
StructField("Origin",StringType(),True),
StructField("Dest",StringType(),True),
StructField("Distance",IntegerType(),True),
StructField("TaxiIn",IntegerType(),True),
StructField("TaxiOut",IntegerType(),True),
StructField("Cancelled",IntegerType(),True),
StructField("CancellationCode",StringType(),True),
StructField("Diverted",IntegerType(),True),
StructField("CarrierDelay",StringType(),True),
StructField("WeatherDelay",StringType(),True),
StructField("NASDelay",StringType(),True),
StructField("SecurityDelay",IntegerType(),True),
StructField("LateAircraftDelay",IntegerType(),True)])


df2007 = spark.read.csv("/home/stp/ML/AirData/data/2007.csv", header = True, schema = schema)
df2008 = spark.read.csv("/home/stp/ML/AirData/data/2008.csv", header = True, schema = schema)

dfall = df2007.union(df2008)
dfall.cache()
dfall.count()

## Inspect data & convert NA 

def chkNA( col):
 if 'NA' in col or 'na' in col:
  return True 
 else: 
  return False

like_f = UserDefinedFunction(lambda x: chkNA(x), BooleanType())

def countNA(df, group_colname ):
  nacount = df.filter(like_f(group_colname)).count()
  return group_colname  + " NA count = "  + str(nacount)
  #shcnt.show(shcnt.count(), False)

countNA(dfall, "ArrDelay")
countNA(dfall, "DepDelay")
countNA(dfall, "CarrierDelay")
countNA(dfall, "WeatherDelay")
countNA(dfall, "NASDelay")

dfall.select("ArrDelay").distinct().show()


# The function withColumn is called to add (or replace, if the name exists) a column to the data frame.
udf = UserDefinedFunction(lambda x: x.replace("NA","0"), StringType())
udf2 = UserDefinedFunction(lambda x: x, StringType())
def replaceNA(df, group_colname ):
  # Before 
  nacount = df.where( col(group_colname).like("%NA%") ).count()
  print( group_colname  + " NA count = "  + str(nacount))
  df = df.withColumn(group_colname, udf(col(group_colname)).cast(StringType()))
  # After
  nacount = df.where(col(group_colname).like("%NA%") ).count()
  print( group_colname  + " NA count = "  + str(nacount) )
  df =  df.withColumn(group_colname, udf2(col(group_colname)).cast(IntegerType()))
  return df


df = replaceNA(dfall, group_colname="ArrDelay")
df = replaceNA(dfall, group_colname="DepDelay")
df = replaceNA(dfall, group_colname="CarrierDelay")
df = replaceNA(dfall, group_colname="WeatherDelay")
df = replaceNA(dfall, group_colname="NASDelay")

df.select(col("ArrDelay")).describe().show()
df.select(col("ArrDelay")).describe().show()
dfall = df.rdd.toDF()
dfall.cache()
dfall.printSchema()
dfall.count()
dfall.describe().show()
dfall.show(2,truncate= True)

#ArrDelay
#DepDelay
#CarrierDelay
#NASDelay

from pyspark.sql.functions import  *
gall = dfall.groupBy("UniqueCarrier")
gall.agg( mean("ArrDelay").alias('MeanArrDelay') , max("ArrDelay").alias('MaxArrDelay'),
mean("DepDelay").alias('MeanDepDelay') , max("DepDelay").alias('MaxDepDelay')    ).orderBy(desc('MeanArrDelay'), desc('MeanDepDelay')).show()

gall.agg( mean("ArrDelay").alias('MeanArrDelay') , max("ArrDelay").alias('MaxArrDelay'),  mean("DepDelay").alias('MeanDepDelay') , max("DepDelay").alias('MaxDepDelay')    ).orderBy(asc('MeanArrDelay'), asc('MeanDepDelay')).show()

gall.agg( mean("CarrierDelay").alias('MeanCarrierDelay') , max("CarrierDelay").alias('MaxCarrierDelay') ).orderBy(asc('MeanCarrierDelay')).show()

gdow = dfall.groupBy("DayOfWeek")
gdow.agg( mean("ArrDelay").alias('MeanArrDelay') ,  mean("DepDelay").alias('MeanDepDelay')  ).orderBy(asc('MeanArrDelay'), asc('MeanDepDelay')).show()

  
#arrudf = UserDefinedFunction(lambda time_str: (int(time_str[:2]) * 3600 + int(time_str[-2:]) * 60)/ (24*60)  , StringType())
arrudf = UserDefinedFunction(lambda time_str: ( time_str.split(time_str[-2:])[0]  )  , StringType())
garrtime = dfall.where((col("ArrTime") != "NA") ).groupBy( arrudf("ArrTime").alias("ArrTime") )
garrtime.agg( mean("ArrDelay").alias('MeanArrDelay') ).orderBy(asc('MeanArrDelay'), asc('MeanArrDelay')).show()

garrMonth = dfall.groupBy( col("Month").alias("Month") )
garrMonth.agg( mean("ArrDelay").alias('MeanArrDelay') ).orderBy(asc('MeanArrDelay'), asc('MeanArrDelay')).show()

from graphframes import *
tripVertices = dfall.withColumnRenamed("FlightNum", "id").distinct()
tripEdges = dfall.select(col("FlightNum").alias("FlightNum"), col("ArrDelay").cast("string").alias("ArrDelay"),
col("Origin").alias("src"),col("Dest").alias("dst") )

nacount = tripEdges.where( col("ArrDelay").like("%NA%") ).count()

tripGraph = GraphFrame(tripVertices, tripEdges)
tripGraph.vertices.count()
tripGraph.edges.count()

tripEdges.where(col('ArrDelay')).like('%NA%').count()
tripEdges.where(col('FlightNum')).like('%NA%').count()
tripEdges.where(col('src')).like('%NA%').count()
tripEdges.where(col('dst')).like('%NA%').count()
tripEdges.where(col('FlightNum')).like('%NA%').count()
tripEdges.where(col('FlightNum')).like('%NA%').count()
