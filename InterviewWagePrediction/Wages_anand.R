# for IAN , syntagium
#  by anandrathi.dev@gmail.com
#install.packages("xlsx")
require(xlsx)
require(ggplot2)
require(car)
require(MASS)

set.seed(123)

# Load the data 
wages_data <-read.xlsx("C:/Users/rb117/Documents/personal/IAN/TimeseriesData.xlsx", sheetName = "Sheet2")

# Understand data
str(wages_data)

# Good data . no na's 
summary(wages_data)

# Explore the data .. some plotting
ggplot(data=wages_data, aes(x=TIME, y=Wages, group=LOCATION, colour =LOCATION)) +   geom_line()+   geom_point()
#p + geom_text(data = wages_data[wages_data$LOCATION == "AUS",  ], aes(label = LOCATION), hjust = 0.7, vjust = 1)

# create dummy for LOCATION
dummyFy <- function(inDF) {
  xtrain_dummy <- as.data.frame( model.matrix( ~. , inDF ))
  xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  return(xtrain_dummy)
}

# Split data in train & test 
SamplTrainTest <- function(x, perc = 0.7) {
  smp_size <- floor(perc * nrow(x))
  ## set the seed to make your partition reproductible
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  
  train <- x[train_ind, ]
  test <- x[-train_ind, ]
  return (list(train,test))
}

#Split Train , Test 
wages_data_dummy <- dummyFy(wages_data)
wages_train_test  <- SamplTrainTest(wages_data_dummy, perc=0.7)
wages_train <- wages_train_test[[1]]
wages_test <- wages_train_test[[2]]

y_train <- wages_train$Wages
wages_train <- subset(wages_train, select = -c(Wages))
str(wages_train)

corrplot::corrplot(cor(wages_train))

# First Model 
wagesModel  <- lm(y_train ~ LOCATIONAUT + LOCATIONCAN + LOCATIONDEU + LOCATIONJPN + 
                    LOCATIONKOR + LOCATIONMEX + LOCATIONNZL + LOCATIONUSA + 
                    TIME + CPI + GDP + TT + Unemp +  WorkingPop + 
                    sin(TIME) , wages_train)
summary(wagesModel)
# Residual standard error: 1113 on 129 degrees of freedom
# Multiple R-squared:  0.9895,	Adjusted R-squared:  0.9883 
# F-statistic: 866.5 on 14 and 129 DF,  p-value: < 2.2e-16

# As a general rule, VIF < 5 is acceptable (VIF = 1 means there is no multicollinearity), 
# and VIF > 5 is not acceptable and we need to check our model.
vif(wagesModel)
# LOCATIONAUT LOCATIONCAN LOCATIONDEU LOCATIONJPN LOCATIONKOR LOCATIONMEX LOCATIONNZL LOCATIONUSA        TIME         CPI 
# 2.522098    2.523673    2.980543    5.050061   13.485658   24.556203    4.833103    3.335243   22.488311    3.172094 
# GDP          TT       Unemp          WorkingPop 
# 42.377230    3.588690    3.675410    6.560669 

# TIME        -2.516e+02  6.834e+01  -3.681 0.000340 ***
#CPI         -3.700e+01  3.926e+01  -0.942 0.347770    
#GDP          6.090e-01  6.085e-02  10.010  < 2e-16 ***
#TT           1.433e+01  7.939e+00   1.806 0.073298 .  
#Unemp       -7.446e+01  8.144e+01  -0.914 0.362271     
#WorkingPop   4.831e+02  8.536e+01   5.660 9.36e-08 ***


wagesModel_2  <- stepAIC(wagesModel)
vif(wagesModel_2)
summary(wagesModel_2)
y_test <- wages_test$Wages
wages_test <- subset(wages_test, select = -c(Wages))
str(wages_test)

result <-  predict(wagesModel_2, wages_test)
print(result)

## Result Adjusted R-squared:  0.9901 
y_Result <-  data.frame(year = wages_test$TIME,  y = y_test, result=result)

ggplot(y_Result, aes(year)) + 
  geom_line(aes(y = y, colour = "Value")) + 
  geom_line(aes(y = result, colour = "Predicted"))

R2 <- 1 - (sum((y_Result$y- y_Result$result )^2)/sum((y_Result$y-mean(y_Result$y))^2))
#  0.9903858
