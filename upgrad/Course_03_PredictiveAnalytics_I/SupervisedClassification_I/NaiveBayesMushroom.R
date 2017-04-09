library(e1071)
install.packages("caret")
library(caret)

# Setting user directory
setwd("C:\\Users\\Manuyesh-PC\\Documents\\R\\")
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\SupervisedClassification_I\\")
# Importing data
mush<-read.csv("mushroom1.csv", header = TRUE, sep = ',', stringsAsFactors= FALSE)
summary(mush$Type.of.mushroom)
#Notice that there are some question marks in a particular column stalk.root
# Replacing question marks with NA
mush$stalk.root[mush$stalk.root == '?'] <- NA

# figuring out the number of NA
which(is.na(mush))
sum(is.na(mush$stalk.root))
#counting the number of observation of each category
table(mush$stalk.root)

# Bulbous are maximum. So, that is the mode.

# Checking if the data is unbalanced.
ggplot(mush, aes(mush$Type.of.mushroom)) + geom_bar()
# It is not unbalanced

#Replacing missing values with mode

mush[is.na(mush)] <- "BULBOUS"
# There are no outliers in the data (obvious), the missing values 
# imputation has been done. The data set is balanced too.
#Now we will perform the 70/30 split. Now we will learn the Naive Bayes Algorithm on the training data
# And test it on testing data. 

#Before we proceed to do that, let's look at our data once more, this time
# from the prespective of the Type of variables. The target variable
# type of mushroom is a categorical variable but its data type is character
#along with the rest of the variables in the data set.
# Naive bayes cannot handle character type data. So, let's convert the data
# from character to factor type.

mush$Type.of.mushroom <- as.factor(mush$Type.of.mushroom)
mush$cap.shape <- as.factor(mush$cap.shape)
mush$cap.surface <- as.factor(mush$cap.surface)
mush$cap.color <- as.factor(mush$cap.color)
mush$bruises <- as.factor(mush$bruises)
mush$odor <- as.factor(mush$odor)
mush$gill.attachment <- as.factor(mush$gill.attachment)
mush$gill.spacing <- as.factor(mush$gill.spacing)
mush$gill.size <- as.factor(mush$gill.size)
mush$gill.color <- as.factor(mush$gill.color)
mush$stalk.shape <- as.factor(mush$stalk.shape)
mush$stalk.root <- as.factor(mush$stalk.root)
mush$stalk.surface.above.ring <- as.factor(mush$stalk.surface.above.ring)
mush$stalk.surface.below.ring <- as.factor(mush$stalk.surface.below.ring)
mush$stalk.color.above.ring <- as.factor(mush$stalk.color.above.ring)
mush$stalk.color.below.ring <- as.factor(mush$stalk.color.below.ring)
mush$veil.type <- as.factor(mush$veil.type)
mush$veil.color <- as.factor(mush$veil.color)
mush$ring.number <- as.factor(mush$ring.number)
mush$ring.type <- as.factor(mush$ring.type)
mush$spore.print.color <- as.factor(mush$spore.print.color)
mush$population <- as.factor(mush$population)
mush$habitat <- as.factor(mush$habitat)


#forming training, testing
set.seed(2)
s=sample(1:nrow(mush),0.7*nrow(mush))
mush_train=mush[s,]
mush_test=mush[-s,]

# Removing the label column (type of mushroom) from the test data

mush_test1<- mush_test[, -1]
# Now we will run Naive Bayes algorithm on this data: Using the e1071 package
install.packages("e1071")

model <- naiveBayes(Type.of.mushroom ~. , data = mush_train)

pred <- predict(model, mush_test1)
table(pred, mush_test$Type.of.mushroom)

confusionMatrix(pred, mush_test$Type.of.mushroom)


