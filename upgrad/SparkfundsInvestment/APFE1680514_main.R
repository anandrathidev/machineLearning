#------------------------------------------------------------------------------
# PGDDA Module 6 Assignment: Group Submission
# Spark Funds Investment Scenario Case Study
#--
# OBjective:
# 1. invest between 5 to 15 million USD per round of investment.
# 2. invest only in English-speaking countries.
# 3. invest where most other investors are investing.
#------------------------------------------------------------------------------

#############   CHECKPOINT 1 ####################
#
#--Begin setting up environment------------------------------------------------
#-Set working directory
#setwd("C:\\Users\\Vishesh Sakshi\\Documents\\Assignment\\Investment Case study")
#
#-Install and load packages as per requirement
#install.packages("ggplot2")
library(ggplot2)
#

#
#--Begin Data load-------------------------------------------------------------
#
#-Load the round2 data into a dataset.This will be the fact table
rounds2<-read.csv("rounds2.csv", strip.white = TRUE )

#-Load the companies data into a dataset with seperator as tab. This will be our dimension table
companies_master<-read.csv("companies.txt", sep = "\t", strip.white = TRUE )


#
#--Understand the data---------------------------------------------------------
#
#-Always good to observe the stats of the data frame
str(companies_master)
str(rounds2)

#
#--Initial Data Cleanup--------------------------------------------------------
#
#-Extract the companies name from permalink and convert to uppercase 
companies_master$permalink <- as.factor(sub('/ORGANIZATION/', "", toupper(as.character(companies_master$permalink))))
companies_master$name <- as.factor(toupper(as.character(companies_master$name)))
#--Overwrite the column permalink with the extracted data. This will ensure we only have companies name
#--in the dataframe which is required.
rounds2$company_permalink <- as.factor(sub('/ORGANIZATION/', "", toupper(as.character(rounds2$company_permalink))))

#
#-Begin Initial Analysis of Data-----------------------------------------------
#
#
#-How many unique companies are present in companies name column?[66038]
CompmanyName_Count <- length(unique(tolower(trimws(companies_master$name, which = "both"))))

#-How many unique companies are present in companies permalink column?[66368]
CompmanyPermalink_Count <- length(unique(tolower(trimws(companies_master$permalink, which = "both"))))

#-There is a difference between the names column and permalink column
#-Lets analyse further
dup_Companynames<-companies_master$name[duplicated(companies_master$name)]
dup_companies_dataframe<-subset(companies_master,is.element(name ,dup_Companynames))

#-number of duplicate companies are [625]
#-These include all the duplicated rows
nrow(dup_companies_dataframe)

#------------------------------------------------------------------------------
#--Why are they duplicates ?
#-1. Closed and reopened with diff permalink and same name
#-2. Additional Category /sector are duplicated with same name and diffrent permalink 

#
#-We have a category where the companies status is described
#-We are not intrested in the closed companies as we cannot invest in them
companies <- companies_master[which(tolower(companies_master$status)!="closed"),]

#
#--How many unique companies are present in rounds2? [66368]
uniqueCompmaniesInRound2 <- sort(unique(rounds2$company_permalink))
cnt_uniqueCompmaniesInRound2 <- length(uniqueCompmaniesInRound2)

#--How many unique companies are present in companies? [60130]
uniquePermaLinkIncompanies <- sort(unique(companies_master$permalink))
cnt_uniquePermaLinkIncompanies <- length(uniquePermaLinkIncompanies)

#-We have equal number of Unique companies in both dataframes. this indicates that we have a good join
#-option between the master (companies) data and the child (rounds2) data.

#-In the companies data frame, which column can be used as the  unique key for each company? Write the name of the column.
#-[name] comment: the name column is the unique identifier for a company
keycolCompanies <-  "companies$name"

#-[permalink] comment: the permalink column is the unique identifier for each row in the companies df
keycolCompanies <-  "companies$permalink"

#-Are there any companies in the rounds2 file which are not  present in companies ? [No]
isCompaniesInrounds2NotIncompanies <- all(rounds2$company_permalink %in% companies_master$permalink, na.rm = T)

#-Merge the two data frames so that all variables (columns) in the companies frame
#-are added to the rounds2 data frame.
#-Key is Permalink. The merged frame is master_frame. 
master_frame<-merge(x=rounds2,y=companies_master, by.x = "company_permalink", by.y="permalink", all.x = TRUE )

#-After Merge Check if any companies are present in (child) merge and not in (master) Companies dataframe.[TRUE]
isCompaniesInrounds2NotIncompanies <- all(master_frame$company_permalink %in% companies_master$permalink, na.rm = T)

#-How many observations are present in master_frame ?[114949]
Observations_master_frame <- nrow(master_frame)

#?
#--Lets Cleanup the Data further-----------------------------------------------
#-As we know that there are many closed companies and we are not intrested in these companies
#-let us remove these comanies fromt he dataframe
master_frame <- master_frame[master_frame$status != "closed",]
#--let us now check the number of rows [106224]
Observations_master_frame <- nrow(master_frame)


#############   CHECKPOINT 2 ##################################################
#
#-How many NA values are present in the column raised_amount_usd ?[18321]
Number_of_NA_values <- sum(is.na(master_frame$raised_amount_usd))

#-How many 0 values are present in the column raised_amount_usd ?[407]
Number_of_0_values <- length(master_frame$raised_amount_usd[which(master_frame$raised_amount_usd==0)])

#-Lets calculate the percentage of invalid values [18728]
Total_invalid_rows = Number_of_NA_values +  Number_of_0_values
#-LEts calculate the percentage of inavlid entries in the dataset [~17.63%]
Percent_invalid_rows = 100*(Number_of_NA_values+Number_of_0_values)/ nrow(master_frame)

# 
#-What do you replace NA values of raised_amount_usd  with? Enter a numeric value.
#--1] since "NA" + "" + 0 = less than 20% , we will be eliminating the missing from our analysis 
#     to form the final dataset.
#--2] For Investment purpose the outliers are not to be eliminated as funding can be of any amount 
#     and from anyone. Thus an investment already made as funding cannot be eliminated.
master_frame_orig <- master_frame
master_frame <- subset(master_frame_orig, !is.na(raised_amount_usd))
master_frame <- subset(master_frame, raised_amount_usd > 0)

#
#--Lets have a look at our data now. We can begin working now with this data
#--As we ahve eliminated any data which is not required or in error
View(master_frame)

#############   CHECKPOINT 3  #################################################
#
#--Calculating the total funding done based on funding type using aggregate function
Funding_type_analysis_Orig <- aggregate(master_frame[,6], list(master_frame$funding_round_type), mean)

#--------------------------------------------------------------------------#
#-Average funding amount of venture type	= 10,634,054 ~ 11.9 Million
#-Average funding amount of angel type	=  764,564.3 ~  0.7 million
#-Average funding amount of seed type	 556,606.7 ~0.5 Million
#-Average funding amount of private equity type	62,111,788 ~ 62 Million
#--------------------------------------------------------------------------#
#
#-Saving the different funding values
amountOfVentureType <- as.numeric(Funding_type_analysis_Orig[which(Funding_type_analysis_Orig$Group.1=="venture"),"x"])
amountOfAngelType <- as.numeric(Funding_type_analysis_Orig[which(Funding_type_analysis_Orig$Group.1=="angel"),"x"])
amountOfSeedType <- as.numeric(Funding_type_analysis_Orig[which(Funding_type_analysis_Orig$Group.1=="seed"),"x"])
amountOfPrivateEquityType <- as.numeric(Funding_type_analysis_Orig[which(Funding_type_analysis_Orig$Group.1=="private_equity"),"x"])

#
#-Formating the different amounts as currency
paste('$',formatC(amountOfVentureType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('$',formatC(amountOfAngelType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('$',formatC(amountOfSeedType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('$',formatC(amountOfPrivateEquityType, big.mark=',', digits = 2, format = 'f'), sep = "")

#-Considering that Spark Funds wants to invest between 5 to 15 million USD per  investment round, 
#-which investment type is the most suitable for them?
#-[venture type is 10 Million it falls within rang of 5 to 15 million]
#--[venture]
prefered_venture_type <- as.character(Funding_type_analysis_Orig[which(Funding_type_analysis_Orig$x >= 5000000 & Funding_type_analysis_Orig$x <= 15000000 & Funding_type_analysis_Orig$Group.1 != 'undisclosed'),1])
  
#############   CHECKPOINT 4   ################################################
#
#--Load meta info 
#--not sure why but got a Unicode UTF-8 BOM at the start of the file
#--hence added the fileencoding type
countryInfo<-read.csv("countries.csv", strip.white = TRUE )
Englishcountry<-read.csv("EnglishSpeaking.csv", strip.white = TRUE, header = T, fileEncoding="UTF-8-BOM")
Englishcountry$language <- "English"

#--Ignore NA countries  
master_frame2<-master_frame[which(master_frame$country_code!="" ),]

#--Let us analyse and see if there is any difference after removing no countries entries
Funding_type_analysis <- aggregate(master_frame2[,6], list(master_frame2$funding_round_type), mean)

#-Saving the different funding values
amountOfVentureType <- as.numeric(Funding_type_analysis[which(Funding_type_analysis_Orig$Group.1=="venture"),"x"])
amountOfAngelType <- as.numeric(Funding_type_analysis[which(Funding_type_analysis_Orig$Group.1=="angel"),"x"])
amountOfSeedType <- as.numeric(Funding_type_analysis[which(Funding_type_analysis_Orig$Group.1=="seed"),"x"])
amountOfPrivateEquityType <- as.numeric(Funding_type_analysis[which(Funding_type_analysis_Orig$Group.1=="private_equity"),"x"])

#
#-Formating the different amounts as currency
paste('Venture: $',formatC(amountOfVentureType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('Angel: $',formatC(amountOfAngelType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('Seed: $',formatC(amountOfSeedType, big.mark=',', digits = 2, format = 'f'), sep = "")
paste('Private Equity: $',formatC(amountOfPrivateEquityType, big.mark=',', digits = 2, format = 'f'), sep = "")

#-We are interested ONLY for VENTURE type as it falls within 5-15 million range
master_frame_VENTURE<-master_frame2[which(master_frame2$funding_round_type=="venture" ),]

#-Aggregating based on country
FundsByCountryCode <- aggregate(raised_amount_usd~country_code,master_frame_VENTURE, FUN=sum)

#
#--Merging the Language Data
#-merging the fact and dimension data for country to get country name.
FundsByCountry<-merge(x=FundsByCountryCode,y=countryInfo, by.x = "country_code", by.y="country_code", all.x = TRUE )

#-merging the fact and dimensions data for countries that speak english
FundsByCountryName<-merge(x=FundsByCountry,y=Englishcountry, by.x = "country_name", by.y="country_name", all.x = TRUE )

#
#--Spark Funds wants to see the top 9 countries which have received the highest total funding 
#-(across ALL sectors for the chosen investment type).
#-For the chosen investment type, make a data frame named top9 
#-with top9 countries (based on the total investment amount each country has received).
topFundsByCountry <- FundsByCountryName[order(FundsByCountryName$raised_amount_usd, decreasing = TRUE),]
top9 <- head(topFundsByCountry,9)

#-we filter by only chosing the English speaking countries and then choose teh 3 top contenders
tempFundsByEnglisgCountry <- top9[which(top9$language=="English"),]
top3 <- tempFundsByEnglisgCountry[order(tempFundsByEnglisgCountry$raised_amount_usd,decreasing = TRUE),]


#############   CHECKPOINT 5  #################################################
#
#--Getting the Category list from the master dataframe and converting to character
#--to make use during extraction
category_list <- as.character(master_frame$category_list)
#
#--Extracting the first string before the vertical bar which will be considered the primary sector
master_frame$primary_category <- trimws(sapply(strsplit(category_list, '[|]'), function(x) x[1]),which = "both")

#
#--Make NA as blanks -- this will help making main_sector field as "Blank" when merging 
master_frame$primary_category[which(is.na(master_frame$primary_category))] <- ""

#--Lets count the number of blanks that were created [613]
length(which(master_frame$primary_category == ""))

#--Loading mapping file for categories
mapping = read.csv("mapping_file.csv", strip.white = TRUE)

#--Merging the data to map the primary sector to the main sector type
#--Since the sector is important for us we will use inner join this will eliminate any
#--data which does not have a sector
master_frame3 <- merge(x = master_frame, y = mapping, by.x = "primary_category", by.y="category_list")
#write.table(master_frame3, file="C:/Sandy Files +++/Upgrad files/Group Assignment 1/master_frame3.txt",sep="\t")

#--Santosh -> map the unassigned categories
#
#--Reordering the columns to have a more cleaner and easier veiw to compare
master_frame3 <- master_frame3[c(2:10,1,17,11:16)]

#write.table(master_frame3, file="C:/Sandy Files +++/Upgrad files/Group Assignment 1/master_frame_3_final.txt", sep="\t")

#Validating if we have any NA values
#sum(is.na(master_frame3$main_sector))
#length(which(master_frame3$main_sector=="Blanks"))
#unique(master_frame3$main_sector)

#validate if the blanks have been successfully mapped [188191]
sum(which(master_frame3$main_sector == "Blanks"))
rm(test2)
test2 <- master_frame3[which(master_frame3$primary_category == "Natural Gas Uses"),]


#
#--LEts view our final dataset
View(master_frame3)


#############   CHECKPOINT - 6 ####################################################
#
#--Create a dataframe by filtering based on top investment type and Top country No 1
#--Note you can create the same stats for top country no 2 by replacing top3[index] index = order you want
#
#
#--We will begin with analysis using only the preferred invedtment type [venture]
top_invest_type <- prefered_venture_type

D1 <- subset(master_frame3, funding_round_type == top_invest_type & country_code == top3$country_code[1])
D2 <- subset(master_frame3, funding_round_type == top_invest_type & country_code == top3$country_code[2])
D3 <- subset(master_frame3, funding_round_type == top_invest_type & country_code == top3$country_code[3])

#--since main_Sector is a factor we can get the counts by using the feature
#--we convert to data frame for ease of access of the values
main_sector_counts1 <- as.data.frame(table(D1$main_sector))
main_sector_counts2 <- as.data.frame(table(D2$main_sector))
main_sector_counts3 <- as.data.frame(table(D3$main_sector))

#--merge the dataframes to create a new column that holds the main_Sector and count for each main_Sector 
D1 <- merge(x=D1, y=main_sector_counts1, by.x = "main_sector", by.y = "Var1", all.x = T)
D2 <- merge(x=D2, y=main_sector_counts2, by.x = "main_sector", by.y = "Var1", all.x = T)
D3 <- merge(x=D3, y=main_sector_counts3, by.x = "main_sector", by.y = "Var1", all.x = T)

#--Get the sum of investments for each main sector using aggregate function
main_Sector_invested1 <- aggregate(D1$raised_amount_usd, list(D1$main_sector), FUN = sum)
main_Sector_invested2 <- aggregate(D2$raised_amount_usd, list(D2$main_sector), FUN = sum)
main_Sector_invested3 <- aggregate(D3$raised_amount_usd, list(D3$main_sector), FUN = sum)

#--merge the datframes to get a new column that shows the main_Sector and the sum invested in it
D1 <- merge(x=D1, y=main_Sector_invested1, by.x = "main_sector", by.y = "Group.1", all.x = T)
D2 <- merge(x=D2, y=main_Sector_invested2, by.x = "main_sector", by.y = "Group.1", all.x = T)
D3 <- merge(x=D3, y=main_Sector_invested3, by.x = "main_sector", by.y = "Group.1", all.x = T)

#--rename column so it makes more sense
names(D1)[names(D1) == "Freq"] <- "Sectorwise_invest_count"
names(D1)[names(D1) == "x"] <- "Sectorwise_invest_sum"

names(D2)[names(D2) == "Freq"] <- "Sectorwise_invest_count"
names(D2)[names(D2) == "x"] <- "Sectorwise_invest_sum"

names(D3)[names(D3) == "Freq"] <- "Sectorwise_invest_count"
names(D3)[names(D3) == "x"] <- "Sectorwise_invest_sum"

#--reorder columns to make it easier to compare and read
D1 = D1[c(2:11,1,18,19,13,14,15,16,12,17)]
D2 = D2[c(2:11,1,18,19,13,14,15,16,12,17)]
D3 = D3[c(2:11,1,18,19,13,14,15,16,12,17)]

#--Begin Analysis--------------------------------------------------------------

#-Point 1
#-Total number of investments (count)
Total_num_invest_count_1 <- nrow(D1) 
Total_num_invest_count_2 <- nrow(D2) 
Total_num_invest_count_3 <- nrow(D3) 

# Point 2
#-Total amount of investment (USD)
Total_amnt_invest_1 <- sum(D1$raised_amount_usd) 
Total_amnt_invest_2 <- sum(D2$raised_amount_usd) 
Total_amnt_invest_3 <- sum(D3$raised_amount_usd) 

#--We Begin Top Sector analysis for Country Rank 1-----------------------------
#--Get the top sector name in terms of number of counts of investment
#
#--Point 8
#-Top sector name (no. of  investment-wise)
S1 <- aggregate(Sectorwise_invest_count~main_sector,D1,FUN=mean)
S1 <- S1[order(S1$Sectorwise_invest_count,decreasing = TRUE),]
S2 <- aggregate(Sectorwise_invest_count~main_sector,D2,FUN=mean)
S2 <- S2[order(S2$Sectorwise_invest_count,decreasing = TRUE),]
S3 <- aggregate(Sectorwise_invest_count~main_sector,D3,FUN=mean)
S3 <- S3[order(S3$Sectorwise_invest_count,decreasing = TRUE),]

#-- 3. Top sector name (no. of  investment-wise
Top1stSectorName_for_Country1 = S1$main_sector[1]
Top1stSectorName_for_Country2 = S2$main_sector[1]
Top1stSectorName_for_Country3 = S3$main_sector[1]

#--4. Second sector name (no. of  investment-wise)
Top2ndSectorName_for_Country1 = S1$main_sector[2]
Top2ndSectorName_for_Country2 = S2$main_sector[2]
Top2ndSectorName_for_Country3 = S3$main_sector[2]

#--5. Second sector name (no. of  investment-wise)
Top3rdSectorName_for_Country1 = S1$main_sector[3]
Top3rdSectorName_for_Country2 = S2$main_sector[3]
Top3rdSectorName_for_Country3 = S3$main_sector[3]


#-- 6. Number of investments in top  sector (3)
# investments  in top sectors
TOP1_sector_Country1_df <- D1[which(D1$main_sector == Top1stSectorName_for_Country1 ),]
TOP1_sector_Country2_df <- D2[which(D2$main_sector == Top1stSectorName_for_Country2 ),]
TOP1_sector_Country3_df <- D3[which(D3$main_sector == Top1stSectorName_for_Country3 ),]


TOP2_sector_Country1_df <- D1[which(D1$main_sector == Top2ndSectorName_for_Country1 ),]
TOP2_sector_Country2_df <- D2[which(D2$main_sector == Top2ndSectorName_for_Country2 ),]
TOP2_sector_Country3_df <- D3[which(D3$main_sector == Top2ndSectorName_for_Country3 ),]


TOP3_sector_Country1_df <- D1[which(D1$main_sector == Top3rdSectorName_for_Country1 ),]
TOP3_sector_Country2_df <- D2[which(D2$main_sector == Top3rdSectorName_for_Country2 ),]
TOP3_sector_Country3_df <- D3[which(D3$main_sector == Top3rdSectorName_for_Country3 ),]

Investments_TOP_1_sector_Country1 <- nrow(TOP1_sector_Country1_df)
Investments_TOP_1_sector_Country2 <- nrow(TOP1_sector_Country2_df)
Investments_TOP_1_sector_Country3 <- nrow(TOP1_sector_Country3_df)

Investments_TOP_2_sector_Country1 <- nrow(TOP2_sector_Country1_df)
Investments_TOP_2_sector_Country2 <- nrow(TOP2_sector_Country2_df)
Investments_TOP_2_sector_Country3 <- nrow(TOP2_sector_Country3_df)

Investments_TOP_3_sector_Country1 <- nrow(TOP3_sector_Country1_df)
Investments_TOP_3_sector_Country2 <- nrow(TOP3_sector_Country2_df)
Investments_TOP_3_sector_Country3 <- nrow(TOP3_sector_Country3_df)


#-- Point 9
#-- 9. For point 3 (top sector count-wise),  which company received the  highest investment?

#--aggregate amount based on company within top 1st main sector within country 
COMPANIES_TOP_sector_1_country_1 <- aggregate(raised_amount_usd~company_permalink, TOP1_sector_Country1_df, sum)
COMPANIES_TOP_sector_1_country_2 <- aggregate(raised_amount_usd~company_permalink, TOP1_sector_Country2_df, sum)
COMPANIES_TOP_sector_1_country_3 <- aggregate(raised_amount_usd~company_permalink, TOP1_sector_Country3_df, sum)

#--decending order the aggregate on amount  within sector within country  
COMPANIES_TOP_sector_1_country_1 <- COMPANIES_TOP_sector_1_country_1[order(COMPANIES_TOP_sector_1_country_1$raised_amount_usd, decreasing = TRUE),]
COMPANIES_TOP_sector_1_country_2 <- COMPANIES_TOP_sector_1_country_2[order(COMPANIES_TOP_sector_1_country_2$raised_amount_usd, decreasing = TRUE),]
COMPANIES_TOP_sector_1_country_3 <- COMPANIES_TOP_sector_1_country_3[order(COMPANIES_TOP_sector_1_country_3$raised_amount_usd, decreasing = TRUE),]


#--aggregate amount based on company within top 2ND  main sector within country 
COMPANIES_TOP_sector_2_country_1 <- aggregate(raised_amount_usd~company_permalink, TOP2_sector_Country1_df, sum)
COMPANIES_TOP_sector_2_country_2 <- aggregate(raised_amount_usd~company_permalink, TOP2_sector_Country2_df, sum)
COMPANIES_TOP_sector_2_country_3 <- aggregate(raised_amount_usd~company_permalink, TOP2_sector_Country3_df, sum)

#--decending order the aggregate on amount  within 2ND sector within country  
COMPANIES_TOP_sector_2_country_1 <- COMPANIES_TOP_sector_2_country_1[order(COMPANIES_TOP_sector_2_country_1$raised_amount_usd, decreasing = TRUE),]
COMPANIES_TOP_sector_2_country_2 <- COMPANIES_TOP_sector_2_country_2[order(COMPANIES_TOP_sector_2_country_2$raised_amount_usd, decreasing = TRUE),]
COMPANIES_TOP_sector_2_country_3 <- COMPANIES_TOP_sector_2_country_3[order(COMPANIES_TOP_sector_2_country_3$raised_amount_usd, decreasing = TRUE),]

#-- 9. For point 3 (top sector count-wise),  which company received the  highest investment?
Country_1_Top1_Cmpny <- as.character(COMPANIES_TOP_sector_1_country_1$company_permalink[1])
Country_2_Top1_Cmpny <- as.character(COMPANIES_TOP_sector_1_country_2$company_permalink[1])
Country_3_Top1_Cmpny <- as.character(COMPANIES_TOP_sector_1_country_3$company_permalink[1])

#-- Point 10
#-- 10. For point 4 (second best sector  count-wise), which company  received the highest investment?
Country_1_Top2_Cmpny <- as.character(COMPANIES_TOP_sector_2_country_1$company_permalink[1])
Country_2_Top2_Cmpny <- as.character(COMPANIES_TOP_sector_2_country_2$company_permalink[1])
Country_3_Top2_Cmpny <- as.character(COMPANIES_TOP_sector_2_country_3$company_permalink[1])


#############   CHECKPOINT 7   - Santosh Francis   ####################
#
#--Create a dataframe by filtering based on top investment type and Top country No 1
#--Note you can create the same stats for top country no 2 by replacing top3[index] index = order you want
#
#
#--Extract the data from funding analysis df
pietable <- Funding_type_analysis[Funding_type_analysis$Group.1 %in% c("venture", "seed", "private_equity"),]

#--Calculating the total count of investments in each category
mf <- master_frame3[master_frame3$funding_round_type %in% c("private_equity", "seed", "venture"),]
t_count <- nrow(mf)
v_mf <- master_frame3[master_frame3$funding_round_type %in% c( "venture"),]
v_count <- nrow(v_mf)
s_mf <- master_frame3[master_frame3$funding_round_type %in% c( "seed"),]
s_count <- nrow(s_mf)
e_mf <- master_frame3[master_frame3$funding_round_type %in% c( "private_equity"),]
e_count <- nrow(e_mf)

v_val <- as.numeric(pietable$x[pietable$Group.1 == "venture"])/v_count
s_val <- pietable$x[pietable$Group.1 == "seed"]
e_val <- pietable$x[pietable$Group.1 == "private_equity"]

#--Calculating the cost per unit investment for each category
pietable$cui <- c(v_count/t_count, s_count/t_count, e_count/t_count )*100


#--Rename columns
names(pietable)[names(pietable) == "Group.1"] <- "Funding_Type"
names(pietable)[names(pietable) == "x"] <- "Funding_Average"
#-Dividing by 10000 (million) to mae the values more readable
pietable$Funding_Average <- (pietable$Funding_Average)/(1000000)

pielabels <- paste(formatC(pietable$cui, digits = 2, format = 'f'), "%\n","($",formatC(pietable$Funding_Average, big.mark=',', digits = 2, format = 'f')," million)" ,sep = "")


pie(pietable$cui, main="Fraction of total investments globally \n in Top 3 categories \n (average investment in each type)",col=cm.colors(length(pietable$cui)), labels = pielabels, cex = 0.8)
legend("topright", title = "Funding Types", legend = pietable$Funding_Type, fill = cm.colors(length(pietable$cui)),cex = 0.8)

#------------------------------------------------------------------------------
options("scipen"=100, "digits"=4)
bartable <- as.data.frame(topFundsByCountry[1:9,])
bartable$raised_amount_usd <- (bartable$raised_amount_usd)/(1000000)
bartable$language[is.na(bartable$language)] <- "Others"

barlable <- bartable$language[bartable$language == "English" & !is.na(bartable$language)]
plot2 <- ggplot(data = bartable, aes(x = bartable$country_code, y = bartable$raised_amount_usd, fill = bartable$language)) + scale_x_discrete(limits = bartable$country_code)
plot2 <- plot2 + geom_bar(stat = "identity") 
plot2 <- plot2 + xlab("Country Codes") + ylab("Sum of Investments in million USD (x1000000)") 
plot2 <- plot2 + scale_fill_discrete(name="Language")
plot2 <- plot2 + ggtitle("Country [Top 9] Analysis \n (Based on Sum of investments)")
plot2

#------------------------------------------------------------------------------

main_sector_counts1 <- main_sector_counts1[order(main_sector_counts1$Freq, decreasing = T),]
TOP3_sector_Top1_Cntry <- head(main_sector_counts1, 3)
TOP3_sector_Top1_Cntry$country_name <- top3$country_name[1]
TOP3_sector_Top1_Cntry$country_code <- top3$country_code[1]

main_sector_counts2 <- main_sector_counts2[order(main_sector_counts2$Freq, decreasing = T),]
TOP3_sector_Top2_Cntry <- head(main_sector_counts2, 3)
TOP3_sector_Top2_Cntry$country_name <- top3$country_name[2]
TOP3_sector_Top2_Cntry$country_code <- top3$country_code[2]

main_sector_counts3 <- main_sector_counts3[order(main_sector_counts3$Freq, decreasing = T),]
TOP3_sector_Top3_Cntry <- head(main_sector_counts3, 3)
TOP3_sector_Top3_Cntry$country_name <- top3$country_name[3]
TOP3_sector_Top3_Cntry$country_code <- top3$country_code[3]

maptable <- rbind(TOP3_sector_Top1_Cntry, TOP3_sector_Top2_Cntry, TOP3_sector_Top3_Cntry)

plot3 <- ggplot(data = maptable, aes(x = reorder(Var1, -Freq), y = Freq, fill = factor(Var1))) 
plot3 <- plot3 + geom_bar(position = "stack", stat = "identity") + facet_wrap(~ country_name, ncol = 1, scales="free_y") 
plot3 <- plot3 +  theme(axis.text.x = element_blank()) 
plot3 <- plot3 + ylab("Number of Investments") + xlab("Investment Main Sectors") 
plot3 <- plot3 + scale_fill_discrete(name="Sector types") 
plot3 <- plot3 + geom_text(data = maptable, aes(x = Var1, y= Freq, label = Freq))
plot3 <- plot3 + ggtitle("Sector vs Country [Top 3] Analysis \n (Based on Count of investments)")
plot3



