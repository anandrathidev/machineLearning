
list.of.packages <- c( "lubridate","ggplot2","MASS","dplyr","e1071","ROSE","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest", "lift", "plotROC")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

c( "lubridate","ggplot2","MASS","dplyr","e1071","ROSE","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest")

#install.packages('missForest')

library(plotROC)
library(lubridate)
library(ggplot2)
library(MASS)
library(dplyr)
library(e1071)
library(ROSE)

library(caret)
library(caretEnsemble)
library(MLmetrics)
library(pROC)
library(ROCR)
library(reshape)
library(cluster)
library(fpc)
library(missForest)
#install.packages('mlbench')
library(mlbench)

# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************
datamart_init <- read.csv('C:/Users/rb117/Documents/work/POC_analytics/Data/DATAMARTv2.tsv',stringsAsFactors = FALSE, sep = '\t')
str(datamart_init)

campaign_init <- read.csv('C:/Users/rb117/Documents/work/POC_analytics/Data/CAMPAIGNDETAIL_Set.tsv', stringsAsFactors = FALSE, sep = '\t')

str(campaign_init)

# ***************************************************************************


# ***************************************************************************
#                   DATA CLEANING, FORMATTING & NEW FEATURE DERIVATION  ----
# ***************************************************************************

#### . . . .   Clean Datamart   ----
datamart <- datamart_init
datamart[which(is.na(datamart)),] 
sum(is.na(datamart))
datamart <- datamart[complete.cases(datamart),]
sum(is.na(datamart))


#### . . . .   add new data types   ----
datamart$FirstNameLen <- nchar(datamart$first_name,  allowNA = T)
datamart$LastNameLen <- nchar(datamart$name,  allowNA = T)
datamart$adressLen <- nchar(datamart$adress,  allowNA = T)

#### . . . .   change data types   ----
datamart$date_of_birth <- as.Date(datamart$date_of_birth,format='%d/%m/%Y')
datamart$campaign_result <- as.factor(datamart$campaign_result)
datamart$campaign_result_detail <- as.factor(datamart$campaign_result_detail)
datamart$gender <- as.factor(datamart$gender)
datamart$zipcode <- as.factor(datamart$zipcode)
datamart$city <- as.factor(datamart$city)
datamart$date_of_contrat_subscription <-  as.Date(datamart$date_of_contrat_subscription,format='%d/%m/%Y')
datamart$phone_number_used <- as.factor(datamart$phone_number_used)
datamart$other_phone_fiability <- as.factor(datamart$other_phone_fiability)
datamart$phone_type <- as.factor(datamart$phone_type)

datamart_clean <-  subset(datamart, select=-c(first_name, name, adress ))
str(datamart_clean)
summary(datamart_clean)

#     Summary   ----
# ***************************************************************************

# contract_id   campaign_result           campaign_result_detail   gender     date_of_birth       
# Min.   :   1   NON:3203        unreachable          :1566       femme:1756   Min.   :1900-01-01  
# 1st Qu.:1001   OUI: 797        faux numéro          :1356       homme:2241   1st Qu.:1952-11-02  
# Median :2000                   Fiabilisé sans pièces: 692       non  :   3   Median :1960-05-14  
# Mean   :2000                   Wrong Number System  : 215                    Mean   :1957-03-22  
# 3rd Qu.:3000                   Fiabilisé partiel    :  52                    3rd Qu.:1965-07-02  
# Max.   :4000                   En instance de pièces:  46                    Max.   :2008-05-17  
# (Other)              :  73                    NA's   :214         
# zipcode              city      city_population  insurance_contrat_value date_of_contrat_subscription
# 34000  :  27   toulouse   : 100   Min.   :   198   Min.   :    0           Min.   :1937-06-01          
# 31200  :  23   paris      :  97   1st Qu.:  8485   1st Qu.:   90           1st Qu.:1984-12-01          
# 68000  :  23   marseille  :  78   Median : 19573   Median :  249           Median :1990-11-01          
# 31400  :  20   bordeaux   :  41   Mean   : 55778   Mean   : 2710           Mean   :1989-08-19          
# 29200  :  18   montpellier:  41   3rd Qu.: 53890   3rd Qu.: 1191           3rd Qu.:1995-10-01          
# (Other):3880   strasbourg :  30   Max.   :458298   Max.   :96111           Max.   :2014-09-11          
# NA's   :   9   (Other)    :3613   NA's   :53                                                           
# phone_number_used other_phone_fiability    phone_type    FirstNameLen     LastNameLen    
# AXA            : 765              :  20       Cellular: 504   Min.   : 1.000   Min.   : 0.000  
# AXA et CEDRICOM: 831    a verifier: 227       Landline:3496   1st Qu.: 6.000   1st Qu.: 6.000  
# CEDRICOM       :2404    fiable    : 596                       Median : 7.000   Median : 7.000  
# non trouve: 744                       Mean   : 7.223   Mean   : 7.525  
# plausible :2413                       3rd Qu.: 8.000   3rd Qu.: 9.000  
# Max.   :20.000   Max.   :19.000  



#### . . . .   should we simply remove NA    ----
# 214 rows with NA in DOB 
sum(is.na(datamart_clean$date_of_birth))
# 4 rows with wrong DOB 
sum(datamart_clean$date_of_birth == as.Date("1900-01-01", format="%Y-%m-%d"))

#### . . . .   Remove rows with invalid DOB 
#datamart_clean <- datamart_clean[which(datamart$date_of_birth != as.Date("1900-01-01", format="%Y-%m-%d") ) ,]
#datamart_clean <- datamart_clean[which(!is.na(datamart_clean$date_of_birth)),]

#### . . . .   Remove NA zip code 
# 7 invalid zip codes .. ignore them 
sum(is.na(datamart_clean$zipcode))
##datamart_clean_zipcode <- datamart_clean[which(is.na(datamart_clean$zipcode)),]
##datamart_clean  <- datamart_clean[which(!is.na(datamart_clean$zipcode)),]

# 3775 records left
nrow(datamart_clean)

#### . . . .   Impute city population
# 39
#sum(is.na(datamart_clean$city_population))
#datamart_clean  <- datamart_clean[which(!is.na(datamart_clean$city_population)),]
hist(datamart_clean$city_population)
datamart_clean_zpopulation <- datamart_clean[which(is.na(datamart_clean$city_population)),]
zipcodes <- aggregate(INDICE~zipcode , data=datamart_clean_zpopulation, FUN=length )
datamart_clean_zpopulation2 <- datamart_clean[datamart_clean$zipcode %in% zipcodes$zipcode ,  ]
datamart_clean_zpopulation2[is.na(datamart_clean_zpopulation2$city_population),]$city_population  <- 0

# --- hmm intresting these zipcodes have single values in database
aggregate(city_population~zipcode, datamart_clean_zpopulation2 , FUN=sum)
#  may be we should remove them 
# datamart_clean <- datamart_clean[ !(datamart_clean$zipcode %in% zipcodes$zipcode) ,  ]

#  4   "0" contract value  ---- remove ??
sum(datamart_clean$insurance_contrat_value==0)
#datamart_clean <- datamart_clean[datamart_clean$insurance_contrat_value!=0 ,] 

# .. 3732 left 
nrow(datamart_clean)

#impute other_phone_fiability 
summary(datamart_clean$other_phone_fiability)
datamart_clean$other_phone_fiability[datamart_clean$other_phone_fiability==""] <- 'plausible'

summary(datamart_clean$phone_type)

#### . . . .  dob in weeks    ----
datamart_clean$age_in_weeks <-  as.numeric(difftime(as.Date(Sys.Date()) , datamart_clean$date_of_birth , units = c("weeks")))
datamart_clean$doc_weeks  <- as.numeric(difftime( as.Date(Sys.Date()), datamart_clean$date_of_contrat_subscription, units = c("weeks")))

#datamart_clean$age_when_sign <-   as.numeric(difftime( datamart_clean$date_of_contrat_subscription, datamart_clean$date_of_birth ,units = c("weeks")))
#datamart_clean$age_when_sign_ratio <-  as.numeric(difftime( datamart_clean$date_of_contrat_subscription, datamart_clean$date_of_birth ,units = c("weeks")))  / as.numeric(difftime(as.Date(Sys.Date()) , datamart_clean$date_of_birth , units = c("weeks")))

#### . . . .   Clean campaign   ----

campaign <- campaign_init

#campaign$DATE <- as.Date(strptime(campaign_init$DATE,format='%Y%m%d'))
campaign$DATE <-  as.Date(strptime(campaign_init$CallLocalTime,format="%Y-%m-%d %H:%M:%OS"))
campaign$LIB_STATUS_GLO <- as.factor(campaign_init$LIB_STATUS_GLO)
campaign$CallStatusNum <- as.factor(campaign_init$CallStatusNum)
campaign$LIB_STATUS_APPEL <- as.factor(campaign_init$LIB_STATUS_APPEL)
campaign$Call_hour <-  as.factor(lubridate::hour(strptime(campaign_init$CallLocalTime,format="%Y-%m-%d %H:%M:%OS")))
campaign$Duration <-  campaign_init$Duration
campaign$Call_Duration <-  campaign_init$Call_Duration
campaign$Accept_Duration <-  campaign_init$Accept_Duration
campaign$Wait_Duration <-  campaign_init$Wait_Duration
campaign$Wrapup_Duration  <-  campaign_init$Wrapup_Duration
campaign$Conv_Duration <- campaign$Conv_Duration

#### . . . .   drop unwanted campaign coloumns   ----
campaign_clean <-  subset(campaign, select=-c(STATUS, CallStatusNum, CallLocalTime, Call_Duration,  Accept_Duration, Wait_Duration, Wrapup_Duration ))

campaign_clean$Final_Status <- as.character(campaign_clean$LIB_STATUS_GLO)
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Answering Machine')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Assuré décédé')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Déjà traité')] <- "SUCCESS"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='En instance de pièces')] <- "SUCCESS"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='En instance de pièces ')] <- "SUCCESS"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='faux numéro')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Fiabilisé partiel')] <- "SUCCESS"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Fiabilisé sans pièces')] <- "SUCCESS"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='No Answer')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Not qualified')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='rappel personnel')] <- "FAIL"
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Souscripteur décédé')] <- "FAIL"

campaign_clean$Final_Status[which(campaign_clean$Final_Status=='unreachable')]  <- "FAIL" 
campaign_clean$Final_Status[which(campaign_clean$Final_Status=='Wrong Number System')]  <- "FAIL" 

summary(campaign_clean$Final_Status)

campaign_clean$Final_Status <- as.factor(campaign_clean$Final_Status)
campaign_clean <-  subset(campaign_clean, select=-c(LIB_STATUS_GLO ))

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

campaign_clean <- campaign_clean %>%
  group_by(CONTRACT_ID ) %>%
  dplyr::mutate(
    first_call_day = min(DATE),
    last_call_day = max(DATE)
  )

campaign_clean$diff_call_days <-  campaign_clean$first_call_day - campaign_clean$last_call_day
campaign_clean$day_of_week <- as.factor(weekdays(campaign_clean$last_call_day))


campaign_grouped <- campaign_clean %>%
  dplyr::group_by(CONTRACT_ID ) %>%
  dplyr::summarise(call_try_count=n(), 
                   first_date=first(DATE),
                   last_date=last(DATE),
                   day_of_week= Mode(day_of_week), 
                   avg_Duration=mean(Duration),
                   Conv_Duration=last(Conv_Duration),
                   last_hour=last(Call_hour),
                   most_hour=Mode(Call_hour),
                   Final_Status=last(Final_Status))

campaign_summary <- campaign_grouped %>%
  dplyr::group_by(Final_Status ) %>%
  dplyr::summarise(each_count = n(),
                   call_try_count=mean(call_try_count), 
                   avg_days=mean(last_date-first_date),
                   avg_Duration=mean(avg_Duration),
                   avg_Conv_Duration=mean(Conv_Duration),
                   last_hour=Mode(most_hour),
                   day_of_week=Mode(day_of_week)
  )

# . . . .   Outlier Treatment ----

# ***************************************************************************
#                   OUTLIER TREATMENT
# ***************************************************************************
OUTLIER=F

if (OUTLIER==T) {
  
  boxplot(datamart_clean$insurance_contrat_value, main="Contract value Outliers")
  table(datamart_clean$campaign_result)
  datamart_clean <- datamart_clean[!datamart_clean$insurance_contrat_value %in% boxplot.stats(datamart_clean$insurance_contrat_value)$out,]
  table(datamart_clean$campaign_result)
  
  
  boxplot(datamart_clean$city_population, main="City population Outliers")
  table(datamart_clean$campaign_result)
  datamart_clean <- datamart_clean[!datamart_clean$city_population %in% boxplot.stats(datamart_clean$city_population)$out,]
  table(datamart_clean$campaign_result)
  
  #boxplot(datamart_clean$age_in_weeks, main="Age in weeks")
  #table(datamart_clean$campaign_result)
  #datamart_clean <- datamart_clean[!datamart_clean$age_in_weeks %in% boxplot.stats(datamart_clean$age_in_weeks)$out,]
  #table(datamart_clean$campaign_result)
  
  
  #boxplot(datamart_clean$doc_weeks, main="Contract age in weeks")
  #table(datamart_clean$campaign_result)
  #datamart_clean <- datamart_clean[!datamart_clean$doc_weeks %in% boxplot.stats(datamart_clean$doc_weeks)$out,]
  #table(datamart_clean$campaign_result)
  
  #boxplot(datamart_clean$age_when_sign, main="sign of Contract age in weeks")
  #table(datamart_clean$campaign_result)
  #datamart_clean <- datamart_clean[!datamart_clean$age_when_sign %in% boxplot.stats(datamart_clean$age_when_sign)$out,]
  #table(datamart_clean$campaign_result)
  
}


# ***************************************************************************
#                   EXPLORATORY ANALYSIS   ----
# ***************************************************************************


#--- datamart plotS
ggplot(Train_campaign_result,aes(x=age_in_weeks))+geom_histogram()+facet_grid(~campaign_result)+theme_bw()		 
ggplot(Train_campaign_result,aes(x=doc_weeks))+geom_histogram()+facet_grid(~campaign_result)+theme_bw()		 
ggplot(Train_campaign_result,aes(x=city_population))+geom_histogram()+facet_grid(~campaign_result)+theme_bw()		 
ggplot(Train_campaign_result,aes(x=insurance_contrat_value))+geom_histogram()+facet_grid(~campaign_result)+theme_bw()		 


#--- campaign plotS

#--- Status / Number of Tries
ggplot(campaign_summary[campaign_summary$Final_Status='OUI',], aes(Final_Status, call_try_count)) + geom_bar(stat = "identity")

#--- Day Of Week  / Status
ggplot(campaign_grouped, aes(day_of_week, fill=Final_Status)) + geom_bar(stat = "count")

#--- Status / Mean call time
ggplot(campaign_summary, aes(Final_Status, avg_Conv_Duration, fill=avg_Duration)) + geom_bar(stat = "identity")


names(datamart_clean)

#-- on Age 

datamart_clean$age_in_weeks_level <- as.factor(cut(as.numeric(datamart_clean$age_in_weeks), 4, include.lowest=TRUE, labels=c( "Low", "Med", "High", "Very High")))
ggplot(datamart_clean, aes(x=datamart_clean$age_in_weeks, fill=campaign_result)) + geom_density(alpha=.3) +xlab("Age in Weeks") + ylab("Density") + ggtitle("Density Age in Weeks" )

#-- on Contract age 
datamart_clean$doc_level <- as.factor(cut(as.numeric(datamart_clean$doc_weeks), 4, include.lowest=TRUE, labels=c( "Low", "Med", "High", "Very High")))
contract_age_corr_df <- datamart_clean %>% 
  dplyr::group_by(doc_level,campaign_result) %>% 
  dplyr::summarise(count=n()) %>% 
  dplyr::mutate(perc=count/sum(count))

ggplot(contract_age_corr_df, aes(doc_level, perc, fill= campaign_result )) + geom_bar(stat = "identity") +
  xlab("Contract Length level") + ylab("Percent of Success/Failure") + ggtitle("Based on Contract Length" )


ggplot(datamart_clean, aes(x=datamart_clean$doc_weeks, fill=campaign_result)) + geom_density(alpha=.3)

#-- on Contract amount 
datamart_clean$contract_level <-  as.factor(cut(datamart_clean$insurance_contrat_value, 4, include.lowest=TRUE, labels=c( "Low", "Med", "High", "Very High")))
contract_corr_df <- datamart_clean %>% 
  dplyr::group_by(contract_level,campaign_result) %>% 
  dplyr::summarise(count=n()) %>% 
  dplyr::mutate(perc=count/sum(count))

ggplot(contract_corr_df, aes(contract_level, perc, fill= campaign_result )) + geom_bar(stat = "identity") +
  xlab("Contract Amount level") + ylab("Percent of Success/Failure") + ggtitle("Based on Contract amount" )

#-- on population 
datamart_clean$population_level <-  as.factor(cut(datamart_clean$city_population, 5, include.lowest=TRUE, labels=c("Very Low", "Low", "Med", "High", "Very High")))
population_corr_df <- datamart_clean %>% 
  dplyr::group_by(population_level,campaign_result) %>% 
  dplyr::summarise(count=n()) %>% 
  dplyr::mutate(perc=count/sum(count))


ggplot(population_corr_df, aes(population_level, perc, fill= campaign_result )) + geom_bar(stat = "identity") +
  xlab("population level") + ylab("Percent of Success/Failure") + ggtitle("Based on population" )


# --on zip codes 
#ggplot(datamart_clean, aes(datamart_clean$zipcode, campaign_result,)) + geom_bar(stat = "summary") +
#  xlab("ZIP") + ylab("ZIP WISE") + ggtitle("Based on population" , subtitle = NULL)
#zip_agg <- aggregate( contract_id ~ zipcode + campaign_result, datamart_clean, FUN=length)


# ***************************************************************************
#                   MERGE DATA FOR  MODELING  
# ***************************************************************************

mergeDF <- merge(x = datamart_clean, y= campaign_grouped, by.x = "INDICE", by.y = "CONTRACT_ID", all.y = F)

sum(is.na(mergeDF))
names(mergeDF)

DFExplore <- subset(mergeDF, select = -c( date_of_contrat_subscription, FirstNameLen, campaign_result_detail,  date_of_birth, city, population_level, contract_level, doc_level, call_try_count, first_date, last_date, day_of_week, avg_Duration, Conv_Duration, last_hour, most_hour) ) 
DFExplore_3 <- subset(mergeDF, select = c( INDICE, campaign_result, Final_Status) ) 

DFAnalize_campaign_result  <- subset(mergeDF, select = -c(Final_Status, zipcode, INDICE, date_of_contrat_subscription, FirstNameLen, campaign_result_detail,  date_of_birth, city, population_level, contract_level, doc_level, age_in_weeks_level,  call_try_count, first_date, last_date, day_of_week, avg_Duration, Conv_Duration, last_hour, most_hour) ) 
names(DFAnalize_campaign_result)
DFAnalize_Final_Status <- subset(mergeDF, select = -c(campaign_result, zipcode, INDICE, date_of_contrat_subscription, FirstNameLen, campaign_result_detail,  date_of_birth, city, population_level, contract_level, doc_level, call_try_count, first_date, last_date, day_of_week, avg_Duration, Conv_Duration, last_hour, most_hour) ) 
names(DFAnalize_Final_Status)


SamplTrainTest <- function(x, perc = 0.75) {
  smp_size <- floor(perc * nrow(x))
  ## set the seed to make your partition reproductible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  
  train <- x[train_ind, ]
  test <- x[-train_ind, ]
  return (list(train,test))
}

DFAnalize_campaign_result_cat  <- subset(mergeDF, select = -c(Final_Status, doc_weeks, age_in_weeks, 
                                                              insurance_contrat_value, zipcode, INDICE, date_of_contrat_subscription, FirstNameLen, campaign_result_detail,  date_of_birth, city, city_population,  call_try_count, first_date, last_date, day_of_week, avg_Duration, Conv_Duration, last_hour, most_hour) ) 
TrainTestCat <-  SamplTrainTest(DFAnalize_campaign_result_cat, perc=0.95)
TrainCat_campaign_result <- data.frame(TrainTestCat[1])
names(TrainCat_campaign_result)
TestCat_campaign_result <- data.frame(TrainTestCat[2])
names(TestCat_campaign_result)


DFAnalize_campaign_result  <- subset(mergeDF, select = -c(Final_Status, age_in_weeks_level, doc_level,
                                                          population_level, contract_level, zipcode,  
                                                          date_of_contrat_subscription, FirstNameLen, campaign_result_detail,  
                                                          date_of_birth, city, call_try_count, first_date, 
                                                          last_date, day_of_week, avg_Duration, Conv_Duration, last_hour, most_hour) ) 

#-- keep call try count  
#install.packages('ROSE')




#Train_campaign_result <- data.rose <- ROSE(campaign_result ~ ., data = Train_campaign_result, seed = 111)$data




# ***************************************************************************
#                   PROBABLITY MODELING ---- 
# ***************************************************************************

#--- calculate probablity for n tries for ANY customer 
tdf <- as.data.frame(table(campaign_grouped$Final_Status))
tdf <- reshape::cast(tdf, ~Var1, value = 'Freq')
pnTriesSuccessAll <- (tdf$SUCCESS)/nrow( campaign_clean  )
NTRY=4
pnTriesSuccessNtriesAll <- ( factorial(NTRY) / (factorial(NTRY-1)*1) ) * (pnTriesSuccessAll^1) * ((1-pnTriesSuccessAll)^(NTRY-1))
options("scipen"=100, "digits"=4)
print(pnTriesSuccessNtriesAll)

#--- calculate probablity for n tries for SuccessFul customer 
tdf <- as.data.frame(table(campaign_grouped$Final_Status))
tdf <- reshape::cast(tdf, ~Var1, value = 'Freq')

calcBinProbablity <- function(NTRY=4) {
  SUCESS_TRY<-1
  pnTriesSuccess <- (tdf$SUCCESS)/nrow( subset( campaign_clean, Final_Status=="SUCCESS" )  )
  pnTriesSuccessNtries <- ( factorial(NTRY) / (factorial(NTRY-1)* factorial(SUCESS_TRY)  ) ) * (pnTriesSuccess^SUCESS_TRY) * ((1-pnTriesSuccess)^(NTRY-SUCESS_TRY))
  options("scipen"=100, "digits"=4)
  print(pnTriesSuccessNtries)
  return(pnTriesSuccessNtries)
}

callProb <- as.data.frame(matrix(lapply(1:20, FUN=calcBinProbablity))) 
callProb$Num_Of_Calls <-  as.numeric(rownames(callProb)) 
callProb$Probablity_Customer_Answers_Call <- unlist(callProb$V1)
str(callProb)
ggplot(callProb, aes(x=Num_Of_Calls, y=Probablity_Customer_Answers_Call)) + geom_bar(stat="identity")

# ***************************************************************************
#                   MODELING   ---- ENSEMBLE
# ***************************************************************************



mytrans <- function(x) {
  #return(log(x))) ## 
  #return(1/(1+exp(-x))) ## Better than logx(x)
  #return( sqrt(x)) ## same as sigmoid
  return(x) ## 
}

dummyFy <- function(inDF) {
  xtrain <- subset(inDF, select = -c(campaign_result ))  
  xtrain$age_in_weeks <- mytrans(scale(as.numeric(xtrain$age_in_weeks)))
  xtrain$doc_weeks  <-  mytrans(scale(as.numeric(xtrain$doc_weeks)))
  xtrain$city_population <-  mytrans(scale(xtrain$city_population))
  xtrain$insurance_contrat_value <-  mytrans(scale(xtrain$insurance_contrat_value))
  #xtrain$age_when_sign <- log(xtrain$age_when_sign)
  #previous_na_action <- options('na.action')
  #options(na.action='na.pass')
  xtrain_dummy <- as.data.frame( model.matrix( ~. , xtrain ))
  xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  
  #clus <- kmeans(xtrain_dummy, centers=5)
  # Fig 01
  #plotcluster(xtrain_dummy, clus$cluster)
  #xtrain_dummy$k <- clus$cluster
  #xtrain_dummy$k <- as.factor(xtrain_dummy$k)
  #xtrain_dummy <- as.data.frame( model.matrix( ~. , xtrain_dummy))
  #xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  colnames(xtrain_dummy)[5] <- "phone_number_usedAXA_et_CEDRICOM"
  colnames(xtrain_dummy)[8] <- "other_phone_fiabilitynon_trouve"
  #xtrain_dummy[is.na(xtrain_dummy),]
#table(xtrain_dummy$k,Train_campaign_result$campaign_result )
  return(xtrain_dummy)
  }

#xtrain.pca <- prcomp(xtrain, center = F, scale. = F) 
#plot(xtrain_dummy.pca, type = "l")
#names(xtrain_dummy.pca)





#..... Remove ID column
sum(is.na(DFAnalize_campaign_result))
DFAnalize_campaign_result[which(is.na(DFAnalize_campaign_result)),]
DFAnalize_campaign_resultBal <- subset(DFAnalize_campaign_result, select = -c(INDICE))

#..... Impute null values 
sum(is.na(DFAnalize_campaign_resultBal))
imputationResults <- missForest(DFAnalize_campaign_resultBal)
DFAnalize_campaign_resultBal <- imputationResults$ximp
sum(is.na(DFAnalize_campaign_resultBal))
DFAnalize_campaign_resultBal[which(is.na(DFAnalize_campaign_resultBal)),]



# Subsampling Techniques 
#Train_campaign_result <- ROSE(campaign_result ~ ., data = DFAnalize_campaign_resultBal, seed = 123, N=5000)$data
Train_campaign_result <- ROSE(campaign_result ~ ., data = DFAnalize_campaign_resultBal, seed = 123)$data
TrainTest <- SamplTrainTest(DFAnalize_campaign_resultBal, perc=0.80)
Train_campaign_result <- data.frame(TrainTest[1])
Test_campaign_result <- data.frame(TrainTest[2])

#Test_campaign_result_ID <- Test_campaign_result$INDICE
#Train_campaign_result_ID <- Train_campaign_result$INDICE
#Train_campaign_result <- subset(Train_campaign_result, select = -c(INDICE))
#Test_campaign_result <- subset(Test_campaign_result, select = -c(INDICE))
#names(Train_campaign_result)


xtrain_dummy <- dummyFy(Train_campaign_result) 
sum(is.na(xtrain_dummy))
imputationResults <- missForest(xtrain_dummy)
xtrain_dummy <- imputationResults$ximp
sum(is.na(xtrain_dummy))

xtrain_dummyr <- xtrain_dummy
ytrain <- Train_campaign_result$campaign_result
xtrain_dummyr$campaign_result <- ytrain


x_dummy <- dummyFy(Test_campaign_result) 
imputationResults <- missForest(x_dummy)
x_dummy <- imputationResults$ximp
y <- Test_campaign_result$campaign_result

featurePlot(x = xtrain_dummy, 
            y = ytrain, 
            plot = "box", 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(3,3 ), 
            auto.key = list(columns = 2))


# Models

#algorithmList <- c( 'rf', 'C5.0', 'LogitBoost', 'bagFDA', 'gbm' )
RunAglorithims <- function( xmethodList, xmetric, xfitControl,  xxtrain, xytrain , xverbose = F, ximportance = T ) {
  xmodelList = sapply(X=xmethodList,
                      FUN= function(lmethodName) { caret::train(x = lxtrain, 
                                               y = lytrain,
                                               trControl = lfitControl, 
                                               method = lmethodName,
                                               metric = lmetric , 
                                               verbose = lverbose, 
                                               importance = limportance) }, 
                      lmetric=xmetric, 
                      lfitControl=xfitControl,
                      lxtrain=xxtrain, 
                      lytrain=xytrain, 
                      lverbose=xverbose, 
                      limportance=ximportance
                      )
  
return(xmodelList)
}


RRunAglorithims <- function( xmethodList, xmetric, xfitControl,  xxtrain, xytrain , ...  ) {
  xmodelList = list()
  for(i in length(xmethodList)){
    xmodelList[[xmethodList[[i]]]] =  do.call(what = caret::train, list(x = xxtrain, 
                                                   y = xytrain,
                                                   metric = xmetric,
                                                   trControl = xfitControl,
                                                   method = xmethodList[[i]],
                                                   ... =  ...)
                                                   )

  }
  return(xmodelList)
}


set.seed(123)
fitControl <- trainControl( method = "repeatedcv", number = 10, savePredictions = T, classProbs = T,  summaryFunction = twoClassSummary)
#methodList <- c( 'LogitBoost' )
methodList <- c( 'LogitBoost', 'C5.0',  'rf',  'gbm', 'knn', 'dnn' )
methodList2 <- c('gbm', 'knn', 'dnn' )
models <- RRunAglorithims( xmethodList=methodList,  xmetric="Accuracy", xfitControl=fitControl, xxtrain=xtrain_dummy, xytrain=ytrain, xverbose = F, ximportance = T) 
models2 <- RRunAglorithims( xmethodList=methodList2,  xmetric="Accuracy", xfitControl=fitControl, xxtrain=xtrain_dummy, xytrain=ytrain) 
model <- append(models, models2) 
names(models2)

#fitControl <- trainControl( method = "repeatedcv", number = 10, savePredictions = T, classProbs = T)
#models <- train(campaign_result~., data=xtrain_dummyr, trControl=fitControl, methodList=algorithmList, metric="ROC", verbose=F)

#modelfs  <- train(x = xtrain_dummy, y=ytrain, trControl=fitControl, method='lvq', preProcess = 'scale' , metric="Accuracy" , verbose = F)
models_LogitBoost <- train(x = xtrain_dummy,y=ytrain, trControl=fitControl, method='LogitBoost', metric="Accuracy" , verbose = F, importance = TRUE)
importance_LB <- varImp(models_LogitBoost)
models_c5 <- train(x = xtrain_dummy, y=ytrain, trControl=fitControl, method='C5.0', metric="Accuracy" , verbose = F, importance = TRUE)
importance_c5 <- varImp(models_c5)

#models_rf <- train(x = xtrain_dummy, y=ytrain, trControl=fitControl, method='rf', metric="Accuracy" , verbose = F, importance = TRUE)
models_rf <- train(x = subset(Train_campaign_result, select=-c(campaign_result)) , y=ytrain, trControl=fitControl, method='rf', metric="Accuracy" , verbose = F, importance = TRUE)
importance_rf <- varImp(models_rf)
models_gbm <- train(x = xtrain_dummy, y=ytrain, trControl=fitControl, method='gbm', metric="Accuracy" , verbose = F)
importance_gbm <- varImp(models_gbm )
importance_nm <- varImp(models_rf, value = "nsubsets", useModel = FALSE)


models_nn <- caret::train(x = xtrain_dummy, y=ytrain, trControl=fitControl, method='dnn', metric="Accuracy", importance = TRUE )
importance_nn <- varImp(models_nn)



Lmodels_LogitBoost <- readRDS( file="C:/Users/rb117/Documents/work/POC_analytics/models_LogitBoost.rds")
Lmodels_c5 <- readRDS( file="C:/Users/rb117/Documents/work/POC_analytics/models_c5.rds")
Lmodels_c5 <- readRDS( file="C:/Users/rb117/Documents/work/POC_analytics/models_rf.rds")
Lmodels_gbm <- readRDS( file="C:/Users/rb117/Documents/work/POC_analytics/models_gbm.rds")

identical(models_LogitBoost, Lmodels_LogitBoost)

#svm.aftertune <- tune.svm(campaign_result~., data=xtrain_dummyr, cost = 2^(2:4), 
#                          gamma = 2^c(0,2), 
#                          kernel = "polynomial") 


#models_svm  <- svm(campaign_result~., data=xtrain_dummyr, kernel="linear", 
#                                       cost=1,
#                                       gamma=0.5, 
#                                       probability=F, scale = T )

models_svm  <- svm(campaign_result~., data=xtrain_dummyr, kernel="polynomial", 
                   cost=2,
                   gamma=0.5, 
                   probability=T, scale = T )

importance_svm <- varImp(models_svm )

saveRDS(models_LogitBoost, file="C:/Users/rb117/Documents/work/POC_analytics/models_LogitBoost.rds")
saveRDS(models_c5, file="C:/Users/rb117/Documents/work/POC_analytics/models_c5.rds")
saveRDS(models_c5, file="C:/Users/rb117/Documents/work/POC_analytics/models_rf.rds")
saveRDS(models_gbm, file="C:/Users/rb117/Documents/work/POC_analytics/models_gbm.rds")
saveRDS(models_svm, file="C:/Users/rb117/Documents/work/POC_analytics/models_svm.rds")
saveRDS(models_nn, file="C:/Users/rb117/Documents/work/POC_analytics/models_nn.rds")

#importance_svm <- varImp(models_svm)


plot(importance_LB, main =" Important variables- LB")
plot(importance_c5, main =" Important variables- C5")
plot(importance_rf, main =" Important variables- RF")
plot(importance_gbm, main =" Important variables- GBM")
plot(importance_nm, main =" Important variables- Filter")
plot(importance_nn, main =" Important variables- Deep NN")

#plot(importance_svm, main =" Important variables- svm")

# ***************************************************************************
#                   EVALUATE MODELS   ---- Test data 
# ***************************************************************************

ModelListRF <- list(RF=models_rf )
ModelList <- list(SVM=models_svm, LB=models_LogitBoost, C5=models_c5, GBM=models_gbm, DNN=models_nn )
names(ModelList)
PredictModel <-  function (x,testData,y) {
  pred_ <- predict(x, testData )
  table(y, pred_)
  return(pred_) 
}

PredictModelProb <-  function (x,testData,y) {
  pred_  <- predict(x, testData )
  pred_Prob <- predict(x, testData, type = "prob" , probability = TRUE  )
  table(y, pred_)
  cf <- confusionMatrix(data = y, pred_, positive = levels(pred_)[2])
  me <- list(
    prediction = pred_,
    pred_Prob = pred_Prob,
    confusionMatrix = cf
  )

  return(me) 
}


predsProbList <-  sapply(ModelList, PredictModelProb, simplify = F, testData=x_dummy, y=y )
trainProbList <-  sapply(ModelList, PredictModelProb, simplify = F, testData=xtrain_dummy, y=ytrain )

RFpredsProbList <-  sapply(ModelListRF, PredictModelProb, simplify = F, testData=Test_campaign_result, y=y )
RFtrainProbList <-  sapply(ModelListRF, PredictModelProb, simplify = F, testData=subset(Train_campaign_result, select=-c(campaign_result)), y=ytrain )

names(predsProbList)
names(trainProbList)
train_prob_svm <- as.data.frame(attr(trainProbList$SVM$pred_Prob, "probabilities"))
train_prob_c5 <- as.data.frame(trainProbList$C5$pred_Prob)
train_prob_LB  <- as.data.frame(trainProbList$LB$pred_Prob)
train_prob_RF <- as.data.frame(RFtrainProbList$RF$pred_Prob)
train_prob_GBM <- as.data.frame(trainProbList$GBM$pred_Prob)
train_prob_DNN <- as.data.frame(trainProbList$DNN$pred_Prob)


trainProbsAll <- data.frame( svmNON=train_prob_svm$NON, svmOUI=train_prob_svm$OUI,
                             c5NON=train_prob_c5$NON, c5OUI=train_prob_c5$OUI,
                             LBNON=train_prob_LB$NON, LBOUI=train_prob_LB$OUI,
                             RFNON=train_prob_RF$NON, RFOUI=train_prob_RF$OUI,
                             DNNNON=train_prob_DNN$NON, DNNOUI=train_prob_DNN$OUI,
                             gbmNON=train_prob_GBM$NON, gbmOUI=train_prob_GBM$OUI)

diffOUINON <- function(x) {
  x$DiffSVM = x$svmOUI-x$svmNON
  x$Diffc5 = x$c5OUI-x$c5NON
  x$DiffLB = x$LBOUI-x$LBNON
  x$DiffRF = x$RFOUI-x$svmNON
  x$Diffgbm = x$gbmOUI-x$RFNON
  x$DiffDNN = x$DNNOUI-x$DNNNON
    
  x2 <- subset( x, select = -c(svmNON, svmOUI, svmOUI, c5OUI, LBNON, LBOUI,RFNON, RFOUI, gbmNON, gbmOUI, DNNNON, DNNOUI) )
   return(x2)
}

#trainProbsAll <- diffOUINON(trainProbsAll)
                             
#xpredProbsAll <- merge(trainProbsAll, xtrain_dummy, by=0, all = T)
#xpredProbsAll <- subset(trainProbsAll, select = -c(LBNON, LBOUI))
xpredProbsAll <- subset(trainProbsAll)
stack_model_gbm <- train(x = xpredProbsAll, y=ytrain, trControl=fitControl, method='gbm', metric="Accuracy" , verbose = F)
stack_models_LogitBoost <- train(x = xpredProbsAll,y=ytrain, trControl=fitControl, method='LogitBoost', metric="Accuracy" , verbose = F)
stack_models_c5 <- train(x = xpredProbsAll, y=ytrain, trControl=fitControl, method='C5.0', metric="Accuracy" , verbose = F)
stack_models_rf <- train(x = xpredProbsAll, y=ytrain, trControl=fitControl, method='rf', metric="Accuracy" , verbose = F)

stack_models_svm  <- svm(x = xpredProbsAll, y=ytrain, kernel="polynomial", 
                   cost=2,
                   gamma=0.5, 
                   probability=T, scale = T )


pred_prob_svm <- as.data.frame(attr(predsProbList$SVM$pred_Prob, "probabilities"))
pred_prob_c5 <- as.data.frame(predsProbList$C5$pred_Prob)
pred_prob_LB  <- as.data.frame(predsProbList$LB$pred_Prob)
pred_prob_RF <- as.data.frame(RFpredsProbList$RF$pred_Prob)
pred_prob_GBM <- as.data.frame(predsProbList$GBM$pred_Prob)
pred_prob_DNN <- as.data.frame(predsProbList$DNN$pred_Prob)

pred_svm <- predsProbList$SVM$prediction
pred_c5 <- predsProbList$C5$prediction
pred_LB  <-predsProbList$LB$prediction
pred_RF <- RFpredsProbList$RF$prediction
pred_GBM <- predsProbList$GBM$prediction
pred_DNN <- predsProbList$DNN$prediction

perf_svm <- prediction(predictions=as.numeric(pred_svm), labels = as.numeric(y))
perf_svm <- prediction(predictions=as.numeric(pred_svm), labels = as.numeric(y))
roc_svm <- performance(perf_svm, measure="tpr", x.measure="fpr")

#install.packages('lift')

pred_list <- list(SVM=pred_svm, C5=pred_c5, LB=pred_LB, RF=pred_RF, GBM=pred_GBM, DNN=pred_DNN)


PlotMyROC <-  function(prediction, aname, colrs, y ) {
  print(aname)
  perf_ <- prediction(predictions=as.numeric(as.factor(prediction)), labels = as.numeric(as.factor(y)))
  roc_ <- performance(perf_, measure="tpr", x.measure="fpr")
  plot( roc(as.numeric(prediction), as.numeric(y), levels = c(1, 2)) ,
            main=  paste(aname," ROC chart"), 
            xlab="1 - Specificity: False Positive Rate",
            ylab="Sensitivity Rate",
            col=colrs
  )  
}

PerfMyROC <-  function(prediction, aname, y  ) {
  perf_ <- prediction(predictions=as.numeric(as.factor(prediction)), labels = as.numeric(as.factor(y)))
  return(perf_)
}

plot( roc(as.numeric(prediction), as.numeric(y), levels = c(1, 2)) ,
      main=  paste(aname," ROC chart"), 
      xlab="1 - Specificity: False Positive Rate",
      ylab="Sensitivity Rate",
      col=colrs
)  

mapply(PlotMyROC, prediction = pred_list, aname = names(pred_list), colrs =  1:length(pred_list) ,  MoreArgs = list(y=y),  SIMPLIFY = FALSE)

PerfMyROCList <-  mapply(PerfMyROC, prediction = pred_list, aname = names(pred_list), MoreArgs = list(y=y),  SIMPLIFY = FALSE)

preds <- cbind(SVM = PerfMyROCList$SVM@predictions[[1]], 
               C5 = PerfMyROCList$C5@predictions[[1]], 
               LB = PerfMyROCList$LB@predictions[[1]], 
               RF = PerfMyROCList$RF@predictions[[1]], 
               DNN = PerfMyROCList$DNN@predictions[[1]], 
               GBM = PerfMyROCList$GBM@predictions[[1]]
)

labels <- cbind(SVM = PerfMyROCList$SVM@labels[[1]], 
               C5 = PerfMyROCList$C5@labels[[1]], 
               LB = PerfMyROCList$LB@labels[[1]], 
               RF = PerfMyROCList$RF@labels[[1]], 
               DNN = PerfMyROCList$DNN@labels[[1]], 
               GBM = PerfMyROCList$GBM@labels[[1]]
)

predsDF <- data.frame(SVM = PerfMyROCList$SVM@predictions[[1]], 
               C5 = PerfMyROCList$C5@predictions[[1]], 
               LB = PerfMyROCList$LB@predictions[[1]], 
               RF = PerfMyROCList$RF@predictions[[1]], 
               GBM = PerfMyROCList$GBM@predictions[[1]],
               DNN = PerfMyROCList$DNN@predictions[[1]], 
               LSVM = PerfMyROCList$SVM@labels[[1]], 
              LC5 = PerfMyROCList$C5@labels[[1]], 
              LLB = PerfMyROCList$LB@labels[[1]], 
              LRF = PerfMyROCList$RF@labels[[1]], 
              LDNN = PerfMyROCList$DNN@labels[[1]], 
              LGBM = PerfMyROCList$GBM@labels[[1]]
)


pred.mat <- prediction(preds, labels = matrix(labels, nrow = nrow((labels)), ncol = ncol(labels)) )

  
ggplot() + 
  geom_roc(data=predsDF, aes(d= LSVM , m = SVM, color="roc SVM")) + 
  geom_roc(data = predsDF, aes(d= LLB , m = LB, color="roc LB") ) +
  geom_roc(data = predsDF, aes(d= LC5 , m = C5, color="roc C5") ) + 
  geom_roc(data = predsDF, aes(d= LRF , m = RF, color="roc RF") ) +
  geom_roc(data = predsDF, aes(d= LGBM , m = GBM, color="roc GBM") ) +
  geom_roc(data = predsDF, aes(d= LDNN , m = DNN, color="roc DNN") ) +
  scale_color_manual(values=c("roc SVM"="red", "roc C5"="blue",  "roc LB" = "darkgoldenrod", "roc RF" = "yellow" , "roc GBM" = "greenyellow", "roc DNN" = "brown"), 
     name="color legend", guide="legend") + 
  style_roc()

#--- Plot Lift 
library(lift)
PlotMyLIFT <-  function(prediction, aname, y  ) {
  plotLift(predicted = prediction, y, main= paste(aname, " - Lift"))
}

perf.mat <- performance(pred.mat, "tpr", "fpr")
plot(perf.mat, colorize = TRUE)
plot(perf.mat, colorize = TRUE)

y2 <- ifelse(as.numeric(y)==1,0,1)
mapply(PlotMyLIFT, prediction = pred_list, aname = names(pred_list), MoreArgs =  list(y=y2), SIMPLIFY = FALSE)

predProbsAll <- data.frame(  svmNON=pred_prob_svm$NON, svmOUI=pred_prob_svm$OUI, 
                             c5NON=pred_prob_c5$NON, c5OUI=pred_prob_c5$OUI, 
                             LBNON=pred_prob_LB$NON, LBOUI=pred_prob_LB$OUI, 
                             RFNON=pred_prob_RF$NON, RFOUI=pred_prob_RF$OUI, 
                             DNNNON=pred_prob_DNN$NON, DNNOUI=pred_prob_DNN$OUI,
                             gbmNON=pred_prob_GBM$NON, gbmOUI=pred_prob_GBM$OUI
)

#predProbsAll <- diffOUINON(predProbsAll)

#predProbsAll <- subset(predProbsAll, select = -c(LBNON, LBOUI))
ensembleModelList <- list(LB=stack_models_LogitBoost, 
                          C5=stack_models_c5, 
                          RF=stack_models_rf, 
                          GBM=stack_model_gbm,
                          SVM=stack_models_svm)
ensembleModelList <- list(LB=stack_models_LogitBoost, 
                          C5=stack_models_c5, 
                          RF=stack_models_rf, 
                          GBM=stack_model_gbm
                          )

ensemblePred <- sapply(ensembleModelList, PredictModelProb, simplify = F, testData=predProbsAll, y=y )
ensemblePredTrain <- sapply(ensembleModelList, PredictModelProb, simplify = F, testData=trainProbsAll, y=ytrain )
epreds <- sapply(X = ensemblePred, FUN = function(x ) {x$prediction}, simplify = F, USE.NAMES = T  )


mapply(PlotMyROC, prediction = epreds, aname = names(epreds), colrs =  1:length(pred_list) , MoreArgs =  list(y=y),  SIMPLIFY = FALSE)
mapply(PlotMyLIFT, prediction = epreds, aname = names(epreds), MoreArgs =  list(y=y2), SIMPLIFY = FALSE)


PerfMyROCList <-  mapply(PerfMyROC, prediction = epreds, aname = names(epreds), MoreArgs = list(y=y),  SIMPLIFY = FALSE)

enpredsDF <- data.frame(C5 = as.numeric(PerfMyROCList$C5@predictions[[1]]), 
                      LB = as.numeric(PerfMyROCList$LB@predictions[[1]]), 
                      RF = as.numeric(PerfMyROCList$RF@predictions[[1]]), 
                      GBM = as.numeric(PerfMyROCList$GBM@predictions[[1]]),
                      DNN = as.numeric(PerfMyROCList$DNN@predictions[[1]]),
                      y = PerfMyROCList$RF@labels[[1]]
)


str(enpredsDF)

ggplot() + 
  geom_roc(data = enpredsDF, aes(d= y, m = LB, color="roc LB") ) +
  geom_roc(data = enpredsDF, aes(d= y, m = C5, color="roc C5") ) + 
  geom_roc(data = enpredsDF, aes(d= y, m = RF, color="roc RF") ) +
  geom_roc(data = enpredsDF, aes(d= y, m = GBM, color="roc GBM") ) +
  scale_color_manual(values=c("roc C5"="blue",  "roc LB" = "darkgoldenrod", "roc RF" = "yellow" , "roc GBM" = "greenyellow"), 
                     name="Ensemble Algo \n color legend", guide="legend") + 
  style_roc()


ggplot() + 
  geom_roc(data = enpredsDF, aes(d= y, m = LB, color="roc LB") ) +
  geom_roc(data = enpredsDF, aes(d= y, m = C5, color="roc C5") ) + 
  geom_roc(data = enpredsDF, aes(d= y, m = RF, color="roc RF") ) +
  scale_color_manual(values=c("roc C5"="blue",  "roc LB" = "darkgoldenrod", "roc RF" = "yellow" , 
                              "roc GBM" = "greenyellow"), 
                     name="Ensemble Algo \n color legend", guide="legend") + 
  style_roc()


importance_sm_LB <- varImp(stack_models_LogitBoost )
plot(importance_sm_LB, main =" Important variables- SMLB")
importance_sm_c5 <- varImp(stack_models_c5 )
plot(importance_sm_c5, main =" Important variables- SM c5")
importance_sm_RF <- varImp(stack_models_rf )
plot(importance_sm_RF, main =" Important variables- SM RF")


unlist(lapply(predsProbList, FUN = function(x) { return(list(Kappa=as.numeric(x$confusionMatrix$overall["Kappa"] )))  } ))
unlist(lapply(predsProbList, FUN = function(x) {x$confusionMatrix$byClass["Sensitivity"]}))
unlist(lapply(predsProbList, FUN = function(x) {x$confusionMatrix$byClass["Pos Pred Value"]}))
unlist(lapply(predsProbList, FUN = function(x) {x$confusionMatrix$overall["Accuracy"]}))

unlist(lapply(ensemblePred,FUN = function(x) {x$confusionMatrix$overall["Kappa"]}))
unlist(lapply(ensemblePred,FUN = function(x) {x$confusionMatrix$byClass["Sensitivity"]}))
unlist(lapply(ensemblePred,FUN = function(x) {x$confusionMatrix$byClass["Pos Pred Value"]}))
unlist(lapply(ensemblePred,FUN = function(x) {x$confusionMatrix$overall["Accuracy"]}))

predProbsAllOUI <- subset(predProbsAll, select = c(  svmOUI, c5OUI, LBOUI, RFOUI, gbmOUI)) 
predProbsAllOUI$stack_gbm_OUI <- ensemblePred$GBM$pred_Prob$OUI
predProbsAllOUI$stack_LogitBoost_OUI  <- ensemblePred$LB$pred_Prob$OUI
predProbsAllOUI$stack_c5_OUI <- ensemblePred$C5$pred_Prob$OUI
predProbsAllOUI$stack_rf_OUI <- ensemblePred$RF$pred_Prob$OUI
#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI

gm_mean = function(x, na.rm=TRUE){
  prod(x[x > 0], na.rm=na.rm) 
  exp(mean(log(x)))
}

gm_mean = function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

#-- Find the maximum of probablity 
predProbsAllOUI2 <- predProbsAllOUI
predProbsAllOUI2$Final <- apply(predProbsAllOUI, 1, max)
predProbsAllOUI2$Mean <- apply(predProbsAllOUI, 1, mean)
predProbsAllOUI2$Geom <- apply(predProbsAllOUI, 1, gm_mean)

predProbsAllOUI2$Class <- y
predProbsAllOUI2 <- as.data.frame( model.matrix( ~. , predProbsAllOUI2))
predProbsAllOUI2 <- subset(predProbsAllOUI2, select=-`(Intercept)`)
predProbsAllOUI2$ClassNON <- ifelse(predProbsAllOUI2$ClassOUI==0,1,0)
predProbsAllOUI2$Class <- y
predProbsAllOUI2 <- predProbsAllOUI2[order(predProbsAllOUI2$Final, decreasing = F),]
predProbsAllOUI2$OUISum  <- cumsum(predProbsAllOUI2$ClassOUI)
predProbsAllOUI2$NONSum  <- cumsum(predProbsAllOUI2$ClassNON)
predProbsAllOUI2$Percent <- predProbsAllOUI2$OUISum/ predProbsAllOUI2$NONSum
predProbsAllOUI2$ID <- Test_campaign_result_ID


pseq <- seq(from = 0, to = 1, by = 0.1)


aggFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Final>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100)/max(ct$NON + ct$OUI,1)
}

aggFUNMean <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Mean>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100) /max(ct$NON + ct$OUI,1)
}

aggFUNGeom <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Geom>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100) /max(ct$NON + ct$OUI,1)
}

StraggFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Final>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

StraggFUNMean <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Mean>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  #(ct$OUI*100) /max(ct$NON + ct$OUI,1)
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

StraggFUNGeom <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Geom>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  #(ct$OUI*100) /max(ct$NON + ct$OUI,1)
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

scoreratioMax <- sapply(pseq, FUN=aggFUN)
scoreratioMean <- sapply(pseq, FUN=aggFUNMean)
scoreratioGeomMean <- sapply(pseq, FUN=aggFUNGeom)

StrscoreratioMax <- sapply(pseq, FUN=StraggFUN)
StrscoreratioMean <- sapply(pseq, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq, FUN=StraggFUNGeom)

ScoreconclusionDF <- data.frame(Score=pseq, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDF$PercentMean  <- scoreratioMean
ScoreconclusionDF$RatioMean  <- StrscoreratioMean
ScoreconclusionDF$PercentGeomMean  <- scoreratioGeomMean
ScoreconclusionDF$RatioGeomMean  <- StrscoreratioGeomMean


ggplot(ScoreconclusionDF ) + 
  geom_line(aes(Score, y=Percent), stat = "identity", color="blue" ) +
  geom_line(aes(Score, y=PercentMean), stat = "identity", color="darkgoldenrod") +
  geom_line(aes(Score, y=PercentGeomMean), stat = "identity", color="black") +
  scale_color_manual(values=c("Max"="blue",  "Mean" = "darkgoldenrod", "GeomMean" = "black" ), 
                     name="Score Vs % \n color legend", guide="legend") + 
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Percentage " )

ggplot(ScoreconclusionDF )  + 
  geom_line(aes(Score, y=Percent), stat = "identity", color="blue") +
  geom_line(aes(Score, y=PercentMean), stat = "identity", color="darkgoldenrod") +
  geom_line(aes(Score, y=PercentGeomMean), stat = "identity", color="black") +
  scale_color_manual(values=c("Max"="blue",  "Mean" = "darkgoldenrod", "GeomMean" = "yellow" ), 
                     name="Score Vs % \n color legend", guide="legend") + 
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity " )





################################ above 0.75
pseq75 <- seq(from = 0, to = 1, by = 0.001)

scoreratioMax <- sapply(pseq75, FUN=aggFUN)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)
scoreratioGeomMean <- sapply(pseq, FUN=aggFUNGeom)

StrscoreratioMax <- sapply(pseq75, FUN=StraggFUN)
StrscoreratioMean <- sapply(pseq75, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq75, FUN=StraggFUNGeom)

ScoreconclusionDFMax <- data.frame(Score=pseq75, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDFMean <- data.frame(Score=pseq75, Percent=scoreratioMean, Ratio= StrscoreratioMean)
ScoreconclusionDFMean <- subset(ScoreconclusionDFMean,ScoreconclusionDFMean$Percent>0 )
#scoreratioMean <- scoreratioMean[-11] 
ScoreconclusionDFGeomMean <- data.frame(Score=pseq75, Percent=scoreratioGeomMean, Ratio= StrscoreratioGeomMean)

ggplot(ScoreconclusionDFMax, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity[Max of all] " )

ggplot(ScoreconclusionDFMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Mean of all] " )

ggplot(ScoreconclusionDFGeomMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Geometric Mean of all] " )


plotLift(predicted = predProbsAllOUI2$Final, predProbsAllOUI2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Mean, predProbsAllOUI2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Geom, predProbsAllOUI2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )


plotLift(predicted = predProbsAllOUI2$Final, predProbsAllOUI2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Mean, predProbsAllOUI2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Geom, predProbsAllOUI2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )

lift2 <- lift( Class ~ Final+Mean+Geom, data = predProbsAllOUI2, class = "OUI")
xyplot(lift2, auto.key = list(columns = 3))

# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
#                   EVALUATE MODELS   ---- OUI - NON 
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************

# ***************************************************************************
#   Train model
# ***************************************************************************


trainProbsAllOUI_NON <- subset(xpredProbsAll, select = c(  svmOUI , c5OUI, LBOUI, RFOUI, gbmOUI)) 

trainProbsAllOUI_NON$svmOUI_svmNON  <- xpredProbsAll$svmOUI - xpredProbsAll$svmNON
trainProbsAllOUI_NON$c5OUI_c5NON  <- xpredProbsAll$c5OUI - xpredProbsAll$c5NON
trainProbsAllOUI_NON$LBOUI_LBNON  <- xpredProbsAll$LBOUI - xpredProbsAll$LBNON
trainProbsAllOUI_NON$RFOUI_RFNON  <- xpredProbsAll$RFOUI - xpredProbsAll$RFNON
trainProbsAllOUI_NON$gbmOUI_gbmNON  <- xpredProbsAll$gbmNON - xpredProbsAll$gbmNON


trainProbsAllOUI_NON$stack_gbm_OUI <- ensemblePredTrain$GBM$pred_Prob$OUI
trainProbsAllOUI_NON$stack_LogitBoost_OUI  <- ensemblePredTrain$LB$pred_Prob$OUI
trainProbsAllOUI_NON$stack_c5_OUI <- ensemblePredTrain$C5$pred_Prob$OUI
trainProbsAllOUI_NON$stack_rf_OUI <- ensemblePredTrain$RF$pred_Prob$OUI
trainProbsAllOUI_NON$stack_svm_OUI <- ensemblePredTrain$SVM$pred_Prob$OUI

trainProbsAllOUI_NON$stack_gbm_OUI_NON <- ensemblePredTrain$GBM$pred_Prob$OUI - ensemblePredTrain$GBM$pred_Prob$NON
trainProbsAllOUI_NON$stack_LogitBoost_OUI_NON  <- ensemblePredTrain$LB$pred_Prob$OUI  - ensemblePredTrain$LB$pred_Prob$NON
trainProbsAllOUI_NON$stack_c5_OUI_NON <- ensemblePredTrain$C5$pred_Prob$OUI - ensemblePredTrain$C5$pred_Prob$NON
trainProbsAllOUI_NON$stack_rf_OUI_NON <- ensemblePredTrain$RF$pred_Prob$OUI - ensemblePredTrain$RF$pred_Prob$NON
trainProbsAllOUI_NON$stack_svm_OUI_NON <- ensemblePredTrain$SVM$pred_Prob$OUI - ensemblePredTrain$SVM$pred_Prob$NON


#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI


#-- Find the maximum of probablity 


stack_models_svm_3Level  <- svm(x = trainProbsAllOUI_NON2, y=ytrain, kernel="polynomial", 
                                cost=1,
                                gamma=0.5, 
                                probability=T, scale = T )




# ***************************************************************************
#   Pred model
# ***************************************************************************


predProbsAllOUI_NON <- subset(predProbsAll, select = c(  svmOUI , c5OUI, LBOUI, RFOUI, gbmOUI)) 

predProbsAllOUI_NON$svmOUI_svmNON  <- predProbsAll$svmOUI - predProbsAll$svmNON
predProbsAllOUI_NON$c5OUI_c5NON  <- predProbsAll$c5OUI - predProbsAll$c5NON
predProbsAllOUI_NON$LBOUI_LBNON  <- predProbsAll$LBOUI - predProbsAll$LBNON
predProbsAllOUI_NON$RFOUI_RFNON  <- predProbsAll$RFOUI - predProbsAll$RFNON
predProbsAllOUI_NON$gbmOUI_gbmNON  <- predProbsAll$gbmNON - predProbsAll$gbmNON


predProbsAllOUI_NON$stack_gbm_OUI <- ensemblePred$GBM$pred_Prob$OUI
predProbsAllOUI_NON$stack_LogitBoost_OUI  <- ensemblePred$LB$pred_Prob$OUI
predProbsAllOUI_NON$stack_c5_OUI <- ensemblePred$C5$pred_Prob$OUI
predProbsAllOUI_NON$stack_rf_OUI <- ensemblePred$RF$pred_Prob$OUI
predProbsAllOUI_NON$stack_svm_OUI <- ensemblePred$SVM$pred_Prob$OUI

predProbsAllOUI_NON$stack_gbm_OUI_NON <- ensemblePred$GBM$pred_Prob$OUI - ensemblePred$GBM$pred_Prob$NON
predProbsAllOUI_NON$stack_LogitBoost_OUI_NON  <- ensemblePred$LB$pred_Prob$OUI  - ensemblePred$LB$pred_Prob$NON
predProbsAllOUI_NON$stack_c5_OUI_NON <- ensemblePred$C5$pred_Prob$OUI - ensemblePred$C5$pred_Prob$NON
predProbsAllOUI_NON$stack_rf_OUI_NON <- ensemblePred$RF$pred_Prob$OUI - ensemblePred$RF$pred_Prob$NON
predProbsAllOUI_NON$stack_svm_OUI_NON <- ensemblePred$SVM$pred_Prob$OUI - ensemblePred$SVM$pred_Prob$NON



#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI



pred_3dLevel  <- predict(stack_models_svm_3Level, predProbsAllOUI_NON2 )
pred_Prob3dLevel <- predict(stack_models_svm_3Level, predProbsAllOUI_NON2, type = "prob" , probability = TRUE  )
table(y, pred_Prob3dLevel)
cf <- confusionMatrix(data = y, pred_Prob3dLevel, positive = levels(pred_Prob3dLevel)[2])
cf



pseq75 <- seq(from = 0, to = 1, by = 0.001)

scoreratioMax <- sapply(pseq75, FUN=aggFUN)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)

StrscoreratioMax <- sapply(pseq75, FUN=StraggFUN)
StrscoreratioMean <- sapply(pseq75, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq75, FUN=StraggFUNGeom)

ScoreconclusionDFMax <- data.frame(Score=pseq75, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDFMean <- data.frame(Score=pseq75, Percent=scoreratioMean, Ratio= StrscoreratioMean)
ScoreconclusionDFMean <- subset(ScoreconclusionDFMean,ScoreconclusionDFMean$Percent>0 )
#scoreratioMean <- scoreratioMean[-11] 
ScoreconclusionDFGeomMean <- data.frame(Score=pseq75, Percent=scoreratioGeomMean, Ratio= StrscoreratioGeomMean)

ggplot(ScoreconclusionDFMax, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity[Max of all] " )

ggplot(ScoreconclusionDFMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Mean of all] " )

ggplot(ScoreconclusionDFGeomMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Geometric Mean of all] " )

if(F) {
  
plotLift(predicted = predProbsAllOUI_NON2$Final, predProbsAllOUI_NON2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Mean, predProbsAllOUI_NON2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Geom, predProbsAllOUI_NON2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )

plotLift(predicted = predProbsAllOUI_NON2$Final, predProbsAllOUI_NON2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Mean, predProbsAllOUI_NON2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Geom, predProbsAllOUI_NON2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )
}

lift2 <- lift( Class ~ Final+Mean+Geom, data = predProbsAllOUI_NON2, class = "OUI")
xyplot(lift2, auto.key = list(columns = 3))

