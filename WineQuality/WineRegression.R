##
##  The task is to build a model to predict the wine-quality using 
##  the independent attributes and identify the attributes which 
##  impact quality the most.
##

## check & Install R packages 
list.of.packages <- c("ggplot2","corrplot", "car")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(ggplot2)
library(corrplot)
library(caret)
library(car)

# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************

#dpath =  "C:/Users/rb117/Documents/personal/WineQuality/"
dpath = "D:/Users/anandrathi/Documents/Work/InterView/Wine/"

wine_red_init <- read.csv( paste0(dpath , "winequality-red.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_red_init)
wine_white_init <- read.csv( paste0(dpath , "winequality-white.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_white_init)

#wine_red_init$quality <- as.factor(wine_red_init$quality)
#wine_white_init$quality <- as.factor(wine_white_init$quality)

summary(wine_red_init$quality)
summary(wine_white_init$quality)

str(wine_red_init$quality)
str(wine_white_init$quality)

wine_red_init$red <- 1
wine_white_init$red <- 0

wine_all_init <- rbind(wine_red_init, wine_white_init)
str(wine_all_init)
library(corrplot)
wcor <- cor(subset(wine_all_init, select = -quality))
corrplot(wcor)


#strong correllation 
#density/alchohol  -0.686745422 

wine_all_init$
  
wine_all_init$citric.acid <- wine_all_init$citric.acid + 0.001
str(df)
train.index <- createDataPartition(df$quality, p = .9, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

all.mod1 = lm(quality ~ ., data = train)
summary(all.mod1)
vif(all.mod1)

"
Residual standard error: 0.7346 on 5836 degrees of freedom
Multiple R-squared:  0.294,	Adjusted R-squared:  0.2926 
F-statistic: 202.5 on 12 and 5836 DF,  p-value: < 2.2e-16
"

"
Lets do some exploration, since Rsqr is very low while error is so high 
to understand the relation between field & quality
"
ggplot(data=aggregate(fixed.acidity~quality, wine_all_init, FUN = mean), aes(x=fixed.acidity, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(volatile.acidity~quality, wine_all_init, FUN = mean), aes(x=volatile.acidity, y=quality))  +   geom_line() +   geom_point()
ggplot(data=aggregate(citric.acid~quality, wine_all_init, FUN = mean), aes(x=citric.acid, y=quality)) +   geom_line() +   geom_point() 
ggplot(data=aggregate(residual.sugar~quality, wine_all_init, FUN = mean), aes(x=residual.sugar, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(chlorides~quality, wine_all_init, FUN = mean), aes(x=chlorides, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(free.sulfur.dioxide~quality, wine_all_init, FUN = mean), aes(x=free.sulfur.dioxide, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(total.sulfur.dioxide~quality, wine_all_init, FUN = mean), aes(x=total.sulfur.dioxide, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(density~quality, wine_all_init, FUN = mean), aes(x=density, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(pH~quality, wine_all_init, FUN = mean), aes(x=pH, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(sulphates~quality, wine_all_init, FUN = mean), aes(x=sulphates, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(alcohol~quality, wine_all_init, FUN = mean), aes(x=alcohol, y=quality)) +   geom_line() +   geom_point()
ggplot(data=aggregate(red~quality, wine_all_init, FUN = max), aes(x=red, y=quality)) +   geom_line() +   geom_point()

"
Based on graphs curves, lets create polynomial models  
to see how does it effect on rsqr & std error 
"

fit <- lm(quality ~  I(-exp(sulphates)) + poly(sulphates,3) + sin(sulphates) + cos(sulphates) +  
            I(-exp(pH)) + poly(pH,3) + sin(pH) + cos(pH) +
            I(-exp(total.sulfur.dioxide))  + poly(total.sulfur.dioxide,3) + sin(total.sulfur.dioxide) + cos(total.sulfur.dioxide) +
            I(-exp(free.sulfur.dioxide))  + poly(free.sulfur.dioxide,3) +  sin(free.sulfur.dioxide) + cos(free.sulfur.dioxide) +              
            I(-exp(fixed.acidity)) + poly(fixed.acidity,3) +  sin(fixed.acidity) + cos(fixed.acidity) + 
            I(-exp(volatile.acidity)) + poly(volatile.acidity,3) +  sin(volatile.acidity) + cos(volatile.acidity) +
            poly(free.sulfur.dioxide,3) +  sin(free.sulfur.dioxide) + cos(free.sulfur.dioxide) +
            I(-exp(residual.sugar)) + poly(residual.sugar,3) +   sin(residual.sugar) + cos(residual.sugar) + residual.sugar
            + poly(citric.acid,2) + poly(chlorides,2) + poly(alcohol,2) + poly(density,2)  + poly(total.sulfur.dioxide,2) , data = wine_all_init)


summary(fit)
vif(fit)
"
Residual standard error: 0.7159 on 6447 degrees of freedom
Multiple R-squared:  0.333,	Adjusted R-squared:  0.3279 
F-statistic: 65.69 on 49 and 6447 DF,  p-value: < 2.2e-16
"

"
So there was an improvement , so lets give now try poly with interaction by taking log on each side

"


cnames <- paste(colnames(wine_all_init)[c(1:11)], collapse = ",")

allvp <- paste0(" log(quality)  ~ poly(log( ", paste( colnames(wine_all_init)[c(1:11)], collapse = "),3) + poly(log(" ) , "),3) + red")
allvp_form <- as.formula(allvp)

inter.mod <- lm(formula = allvp_form, data=wine_all_init)
summary(inter.mod)
vif(inter.mod)

"
Residual standard error: 0.1257 on 6462 degrees of freedom
Multiple R-squared:  0.3277,	Adjusted R-squared:  0.3242 
F-statistic: 92.64 on 34 and 6462 DF,  p-value: < 2.2e-16
"

resultdf <- data.frame(y=wine_all_init$quality , predict_y = predict(fit))

ggplot(data=resultdf, aes(x=y, y=predict_y)) +   geom_line() +   geom_point()

log(wine_all_init)

"
Conclusion : as you can see that last polynomial model 
which contains non liner polynimial formula inludes interaction among variables 
significantly reduces  Residual standard error to : 0.1257 from  Residual standard error: 0.7346

Though R^2 is not a measure here also significantly improves Adjusted R-squared from:0.2926   to:0.3242 

This Models were mosly for experimental basis  
to comapre with classification models 
Still I belive further analysis is needed for 
Autocorrelation of Residuals & heteroscedasticity
using nlstools & non paramteric regression. 

As of Now I would recommend to go with Classification

"
