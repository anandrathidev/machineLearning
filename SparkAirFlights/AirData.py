
## Load data 

from pyspark.sql.types import *
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

## Inspect data & convert NA 
df2007.where((col("ArrDelay") == "NA") ).count()
df2008.where((col("ArrDelay") == "NA") ).count()
df2007.where((col("DepDelay") == "NA") ).count()
df2008.where((col("DepDelay") == "NA") ).count()
df2007.where((col("CarrierDelay") == "NA") ).count()
df2008.where((col("CarrierDelay") == "NA") ).count()
df2007.where((col("WeatherDelay") == "NA") ).count()
df2008.where((col("WeatherDelay") == "NA") ).count()


df2007.select("ArrDelay").distinct().show()


# The function withColumn is called to add (or replace, if the name exists) a column to the data frame.
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import *
udf = UserDefinedFunction(lambda x: x.replace("NA","0"), StringType())
df2007 = df2007.withColumn("ArrDelay", udf(col("ArrDelay")).cast(StringType()))
df2008 = df2008.withColumn("ArrDelay", udf(col("ArrDelay")).cast(StringType()))
df2007 = df2007.withColumn("DepDelay", udf(col("DepDelay")).cast(StringType()))
df2008 = df2008.withColumn("DepDelay", udf(col("DepDelay")).cast(StringType()))

df2007 = df2007.withColumn("CarrierDelay", udf(col("CarrierDelay")).cast(StringType()))
df2008 = df2008.withColumn("CarrierDelay", udf(col("CarrierDelay")).cast(StringType()))

df2007 = df2007.withColumn("WeatherDelay", udf(col("WeatherDelay")).cast(StringType()))
df2008 = df2008.withColumn("WeatherDelay", udf(col("WeatherDelay")).cast(StringType()))

df2007 = df2007.withColumn("NASDelay", udf(col("NASDelay")).cast(StringType()))
df2008 = df2008.withColumn("NASDelay", udf(col("NASDelay")).cast(StringType()))


df2007.where((col("ArrDelay") == "NA") ).count()
df2008.where((col("ArrDelay") == "NA") ).count()

udf2 = UserDefinedFunction(lambda x: x, StringType())
df2007 = df2007.withColumn("ArrDelay", udf2(col("ArrDelay")).cast(IntegerType()))
df2008 = df2008.withColumn("ArrDelay", udf2(col("ArrDelay")).cast(IntegerType()))
df2007 = df2007.withColumn("DepDelay", udf2(col("DepDelay")).cast(IntegerType()))
df2008 = df2008.withColumn("DepDelay", udf2(col("DepDelay")).cast(IntegerType()))
df2007 = df2007.withColumn("CarrierDelay", udf2(col("CarrierDelay")).cast(IntegerType()))
df2008 = df2008.withColumn("CarrierDelay", udf2(col("CarrierDelay")).cast(IntegerType()))
df2007 = df2007.withColumn("NASDelay", udf2(col("NASDelay")).cast(IntegerType()))
df2008 = df2008.withColumn("NASDelay", udf2(col("NASDelay")).cast(IntegerType()))
df2007 = df2007.withColumn("WeatherDelay", udf2(col("WeatherDelay")).cast(IntegerType()))
df2008 = df2008.withColumn("WeatherDelay", udf2(col("WeatherDelay")).cast(IntegerType()))


gallNA = df2007.groupBy("ArrDelay")
gallNA.count().alias('NAS').show()
gallNA = df2008.groupBy("ArrDelay")
gallNA.count().alias('NAS').show()

gallNA = df2007.groupBy("DepDelay")
gallNA.count().alias('NAS').show()
gallNA = df2008.groupBy("DepDelay")
gallNA.count().alias('NAS').show()

gallNA = df2007.groupBy("CarrierDelay")
gallNA.count().alias('NAS').show()
gallNA = df2008.groupBy("CarrierDelay")
gallNA.count().alias('NAS').show()

gallNA = df2007.groupBy("NASDelay")
gallNA.count().alias('NAS').show()
gallNA = df2008.groupBy("NASDelay")
gallNA.count().alias('NAS').show()

gallNA = df2007.groupBy("WeatherDelay")
gallNA.count().alias('NAS').show()
gallNA = df2008.groupBy("WeatherDelay")
gallNA.count().alias('NAS').show()

df2007.printSchema()
df2008.printSchema()

df2007.count()
df2008.count()

dfall = df2007.union(df2008)
dfall.count()

dfall.show(2,truncate= True)
dfall.printSchema()
dfall.describe().show()

ArrDelay
DepDelay
CarrierDelay
NASDelay

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


tripVertices = dfall.withColumnRenamed("FlightNum", "id").distinct()
tripEdges = dfall.select("FlightNum", "ArrDelay", "Origin", "Dest")
tripGraph = GraphFrame(tripVertices, tripEdges)
