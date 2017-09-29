#-- Checkpoint 1: (Data Understanding & Data Preparation) 
#-- Transform the original dataset into transaction level dataset (10%)
#-- Order ID is not unique. 
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_04_PredictiveAnalytics_II/AssociationRules/AssociationRulesAssignment")
#install.packages("arules")
#install.packages("arulesViz")

library("arules")
library("arulesViz")

#-- Checkpoint 1: (Data Understanding & Data Preparation) 
#-- Transform the original dataset into transaction level dataset 
trdataOrig <- read.csv("Global Superstore.csv", header = T, sep=",", stringsAsFactors = F)
trdataOrig$count <- 1

# Bascket size per order 
tragg <-aggregate(count~Order.ID, data=trdataOrig, sum, na.rm=TRUE)
nrow(tragg[which(tragg$count<=2),])
nrow(tragg[which(tragg$count>1),])

traggcat <-aggregate(count~Sub.Category, data=trdataOrig, sum, na.rm=TRUE)
nrow(traggcat[which(traggcat$count<=2),])
nrow(traggcat[which(traggcat$count>1),])


#-- Most relevant attribute to analyse would be the "Sub-Category" of the products.
trdata  <- as.data.frame(cbind(OrderID = trdataOrig$Order.ID, product=trdataOrig$Sub.Category))
str(trdata)
trdata$OrderID <- as.factor(trdata$OrderID)
trdata$product <- as.factor(trdata$product)
#trdata$product <- as.character(trdata$product)

str(trdata)
levels(trdata$product)
#-- Convert the transaction level data into transaction format using "arules" package
transObj <- as(split(trdata$product, trdata$OrderID), "transactions")

itemFrequencyPlot(transObj,topN=20,type="absolute")

#-- Checkpoint 2: (Association Rule Mining) (30%)
#-- Mine association rules from the data (15%)

#-- All possible Rules set of 95730 rules
rulesAll <- apriori(transObj, parameter = list(minlen=2, support = 0.00001, confidence = 0.0001))
plot(rulesAll)

itemsets <- generatingItemsets(rulesAll)
itemsets.df <- as(itemsets, "data.frame")

#-- thats 75773 rules

#-- Checkpoint 2: (Association Rule Mining) (30%)
#-- Optimise the minimum support, confidence, lift threshold level or the minimum floor on the number of items required in a transaction to qualify for consideration (15%)
# Start with minimum support = 5%, confidence =30% and min 2 items 
rules_05_30 <- apriori(transObj, parameter=list(minlen=2, support=0.05,confidence=0.3))
rules_05_30

# 0 Rules
# support = 5%, confidence =25% and min 2 items 
rules_05_25 <- apriori(transObj, parameter=list(minlen=2, support=0.05,confidence=0.25))
rules_05_25
# 0 Rules
#  support = 4%, confidence =25% and min 2 items 
rules_04_25 <- apriori(transObj, parameter=list(minlen=2, support=0.04,confidence=0.25))
rules_04_25
# 0 Rules
#  support = 3%, confidence =25% and min 2 items 
rules_03_25 <- apriori(transObj, parameter=list(minlen=2, support=0.03,confidence=0.25))
rules_03_25
# 0 Rules

for (support in seq(10/100,0.5/100,-(0.1/100) )) {
  for (confidence in seq(40/100,25/100,-(1/100) )) {
    sink("NUL")   
  rulesd <- apriori(transObj, parameter=list(minlen=2,maxlen=4, support=support,confidence=confidence))
  sink()
  ruleCount=nrow(rulesd@quality)
  if(ruleCount>0) {
    print(paste(paste0(as.character(support*100),"%",support*51290),  as.character(confidence*100), as.character(nrow(rulesd@quality)), sep = "-"))
  }
  }
}

#-- "0.900000000000001%461.61-28-1"
#--  "0.900000000000001%461.61-27-1"
#--  "0.900000000000001%461.61-26-2"
#--  "0.900000000000001%461.61-25-2"
#--  "0.800000000000001%410.32-28-1"
#--  "0.800000000000001%410.32-27-1"
#--  "0.800000000000001%410.32-26-2"
#--  "0.800000000000001%410.32-25-2"
#--  "0.700000000000001%359.03-28-1"
#--  "0.700000000000001%359.03-27-1"
#--  "0.700000000000001%359.03-26-2"
#--  "0.700000000000001%359.03-25-2"
#--  "0.600000000000001%307.74-32-1"
#--  "0.600000000000001%307.74-31-2"
#--  "0.600000000000001%307.74-30-3"
#--  "0.600000000000001%307.74-29-3"
#--  "0.600000000000001%307.74-28-4"
#--  "0.600000000000001%307.74-27-5"
#--  "0.600000000000001%307.74-26-7"
#--  "0.600000000000001%307.74-25-8"
#--  "0.5%256.45-32-2"
#--  "0.5%256.45-31-3"
#--  "0.5%256.45-30-5"
#--  "0.5%256.45-29-7" <<<<=== SELECT THIS 
#--  "0.5%256.45-28-8"
#--  "0.5%256.45-27-14"
#--  "0.5%256.45-26-19"
#--  "0.5%256.45-25-22"



# support = 0.05%, confidence =51% and min 2 items to max 4 items
rules <- apriori(transObj, parameter=list(minlen=2,maxlen=4, support=0.5/100,confidence=0.29))
rules
# 12 rules 


#-- Checkpoint 3: (Rule Relevance/Evaluation) (40%) 
#-- The numerical value of the support, confidence and the lift level for the itemsets/rule (10%)

rules <- subset(rules, subset = lift > 1.2)
inspect(rules)
itemsets <- generatingItemsets(rules)
itemsets.df <- as(itemsets, "data.frame")
itemsets.df[with(itemsets.df, order(-support)), c("items")]

#-- How logical/relevant/viable are the rules from the store's point of view (10%)

#-- So support & confidence are very low for AR rule  But we find Binders as most basket with 
#-- few combination , Lift is also very good means that , this is popular combos

#-- lhs                           rhs       support     confidence lift    
#-- [1] {Labels,Storage}      => {Binders} 0.005032954 0.2985782  1.386295
#-- [2] {Accessories,Art}     => {Binders} 0.005472339 0.2914894  1.353382
#-- [3] {Accessories,Storage} => {Binders} 0.006231276 0.3170732  1.472167
#-- [4] {Art,Furnishings}     => {Binders} 0.005991612 0.3211991  1.491324
#-- [5] {Furnishings,Storage} => {Binders} 0.005831835 0.3054393  1.418152
#-- [6] {Art,Phones}          => {Binders} 0.006430997 0.3096154  1.437541
#-- [7] {Phones,Storage}      => {Binders} 0.006870381 0.3208955  1.489915

#-- Explain and analyse the business implications of the rule (20%)
#  placements  can be used in deciding the location and promotion of goods inside a store.
plot(rules)
