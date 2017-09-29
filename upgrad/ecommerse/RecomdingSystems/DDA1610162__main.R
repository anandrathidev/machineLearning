#-- Objective
#-- Build a recommendation system (collaborative) for your store,
#-- where customers will be recommended the beer that they are most likely to buy.

#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("recommenderlab")

library(recommenderlab)
library(dplyr)
library(ggplot2)
#-- Data preparation

#setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_06_elective/ecommerce/Assingment")
#beerFile = "C:/Users/rb117/Documents/Personal/Upgrad/Course_05_Ecommerse/beer_data.csv"
beerFile = "C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_06_elective/ecommerce/Assingment/beer_data.csv"
beerRateDF1  <- na.omit(read.csv(beerFile, ",", header = T, stringsAsFactors = F))
str(beerRateDF1)
#beerRateDF1$beer_beerid <- as.factor(beerRateDF1$beer_beerid)
#beerRateDF1$review_profilename <- as.factor(beerRateDF1$review_profilename)

#-- Clean the data of typos max rating upto 5 only
beerRateDF <- beerRateDF1[which(beerRateDF1$review_overall<=5),]
#-- remove duplicates
beerRateDF <- beerRateDF[ !duplicated(beerRateDF[,c("beer_beerid","review_profilename")]),]

#-- Choose only those beers that have at least N number of reviews
#-- Figure out an appropriate value of N using EDA; this may not have one correct answer, but you shouldn't choose beers having extremely low number of ratings
#-- Count number of ratings per beerid
beerCountDF <- aggregate(review_profilename~beer_beerid, data=beerRateDF, FUN=length)

#-- Filter out beers with rating count less than 21
beerCountDFMin20 <- as.data.frame(beerCountDF[which(beerCountDF$review_profilename>20 ), ])

colnames(beerCountDFMin20) <- c("beer_beerid", "reviewCountBeer")
#-- Merge
beerCountDF_Merge <- merge(x=beerRateDF, y=beerCountDFMin20, by  = "beer_beerid", all = FALSE)

userCountDF <- aggregate(beer_beerid~review_profilename, data=beerCountDF_Merge, FUN=length)
userCountDFMin2 <- as.data.frame(userCountDF[which(userCountDF$beer_beerid>1  ), ])
colnames(userCountDFMin2) <- c("review_profilename", "reviewCountUser")

str(beerCountDF_Merge)
str(userCountDFMin2)
beerDF_Merge <- merge(x=beerCountDF_Merge, y=userCountDFMin2, by = "review_profilename", all = FALSE)
str(beerDF_Merge)

#-- Convert this data frame to a "realratingMatrix" before you build your collaborative filtering models
#-- realmatric expextc DF to be   user, item , rating
beerCountDF_Final <- data.frame(  user=beerDF_Merge$review_profilename, item=beerDF_Merge$beer_beerid, rating = beerDF_Merge$review_overall)
beerCountDF_Final$item  <- as.factor(beerCountDF_Final$item)
beerCountDF_Final$rating <- as.numeric(beerCountDF_Final$rating)
beerCountDF_Final <- beerCountDF_Final[order(beerCountDF_Final$item,beerCountDF_Final$user),]

#beerMatrixTemp <- as(beerCountDF_Final,"matrix")
beerMatrix <- as(beerCountDF_Final,"realRatingMatrix")
summary(beerMatrix)

#####################################################################################
####  Data Exploration    
#####################################################################################
#-- Data Exploration

beerMatrix_df <- as(beerMatrix, "data.frame")
str(beerMatrix_df)

#-- Determine how similar the first ten users are with each other and visualise it
similar_users_first_10 <- similarity(beerMatrix[1:10,],
                                     method = "cosine",
                                     which = "users")
#Similarity matrix
as.matrix(similar_users_first_10)
image(as.matrix(similar_users_first_10), main = "User similarity")

#--  First 10 Similar users are :
#--   1st & 4th
#--   2nd & 4th
#--   some what similar 1st & 2nd  


#-- Compute and visualise the similarity between the first 10 beers
similar_beers_first_10 <- similarity(beerMatrix[,1:10 ],
                                     method = "cosine",
                                     which = "items")

as.matrix(similar_beers_first_10)
image(as.matrix(similar_beers_first_10), main = "Beer similarity")
#--  Of First 10 Beers Similar beers are :
#--   1st & 8th,
#--   2nd & 9th & 10TH
#--   4TH & 10th
#--   5TH & 8th
#--   some what similar
#--   1st & 6TH,10th,
#--   6TH  & 7TH
#--   etc

#--------------------------Understand users and ratings----------#

#-- What are the unique values of ratings?
#-- Visualise the rating values and notice:
qplot(getRatings(beerMatrix), binwidth = 1,
      main = "Histogram of ratings", xlab = "Rating")
qplot(getRatings(normalize(beerMatrix, method = "Z-score")),
      main = "Histogram of normalized ratings", xlab = "Rating")

summary(getRatings(beerMatrix)) # Skewed to the right
summary(getRatings(normalize(beerMatrix, method = "Z-score"))) # seems better

#-- The average user ratings

#-- The average beer ratings
boxplot(beerCountDF_Final$rating)


#-- The average number of ratings given to the beers
qplot(colCounts(beerMatrix), binwidth = 10,
      main = "Beers Rated on average",
      xlab = "# of Beers",
      ylab = "# of Beers rated")

average_beer_count  <- setNames(aggregate(review_overall ~beer_beerid , data=beerRateDF, FUN=length), c("items", "count"))
boxplot(average_beer_count$count,outline=FALSE)
#-- median = 2
mean(average_beer_count$count) #-- average number of ratings given to the beers = 11
max(average_beer_count$count) #-- max ratings given to the beers = 987
min(average_beer_count$count) #-- mix ratings given to the beers = 1

summary(colCounts(beerMatrix)) ## with min ratings > 21 of beers
#--   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#-- 21.00   30.00   50.00   87.93   99.00  977.00

#-- The average number of ratings given by the users

qplot(rowCounts(beerMatrix), binwidth = 10,
      main = "Beers Rated on average",
      xlab = "# of users",
      ylab = "# of Beers rated")

average_user_count  <- setNames(aggregate(review_overall ~ review_profilename  , data=beerRateDF, FUN=length), c("user", "count"))
boxplot(average_user_count$count,outline=FALSE)
#-- median = 3 means Max mum users rate  3 beers or stick with max 3 beers
mean(average_user_count$count) #-- average number of ratings given by users  = 21
max(average_user_count$count) #-- max ratings given to the beers = 1846 #whoops some usres have rated 1846 , must be TV food show or magzine user
min(average_user_count$count) #-- mix ratings given to the beers = 1
summary(rowCounts(beerMatrix)) ## with min ratings > 21 of beers
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   1.00    1.00    3.00   17.49   11.00  821.00

#--------------------------Recommendation models ----------------#
sum(rowCounts(beerMatrix)<2)

Beerscheme <- evaluationScheme(beerMatrix, method = "split", train = .9,
                               k = 1, given = 2, goodRating = 4)
algorithms <- list(
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
                                                 method="Cosine",
                                                 nn=9, minRating=3)),
  "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score"
  ))
)


results <- evaluate(Beerscheme, algorithms, n=c(1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100,200,300,500,700))
class(results)

# Draw ROC curve
plot(results, annotate = 1:4, legend="topleft")

# ANSWER : UBCF looks better  


#-- Give the names of the top 5 beers that you would recommend to the users "cokes", "genog" & "giblet"
Rec_model=Recommender(beerMatrix,method="UBCF", 
                      param=list(normalize = "Z-score",method="Cosine",nn=9, minRating=1))

recommended.items.cokes <- predict(Rec_model, beerMatrix["cokes",], n=5)
recommended.items.cokes@items
# ANSWER : 
#-- [1] 3974 1471  826  607  708
recommended.items.genog <- predict(Rec_model, beerMatrix["genog",], n=5)
recommended.items.genog@items
# ANSWER : 
#-- [1] 1396  929  135 1758 3307
recommended.items.giblet <- predict(Rec_model, beerMatrix["giblet",], n=5)
recommended.items.giblet@items
# ANSWER : 
#-- [1]  234 2074   50  113  412
