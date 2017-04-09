#loading the data set

baseball = read.csv("baseball.csv")

#Data summary

summary(baseball)

# Outlier detection

quantile(baseball$RS, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(baseball$RA, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(baseball$BA, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(baseball$SLG, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(baseball$OBP, c(0.95,0.96,0.97,0.98,0.99,1))

# split

install.packages("caTools")
library(caTools)
set.seed(1000)
split_baseball = sample.split(baseball$Playoffs, SplitRatio = 0.7)
table(split_baseball)
baseball_train = baseball[split_baseball,]
baseball_test = baseball[!(split_baseball),]

# Model with all variables
initial_model = glm(Playoffs ~ ., data = baseball_train[,-c(1:3)], family = "binomial")
summary(initial_model)

# Stepwise selection of variables
best_model = step(model,direction = "both")
