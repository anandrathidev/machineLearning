# DATA PREPARATION:
#   Make a grouped bar chart depicting the hour-wise trip request made at city and airport 
#   respectively. You can aggregate the data for all 5 days on the same axis of 24 hours. 
#   Each bar should correspond to an hour and pick-up point (city / airport) should be displayed in two colours?(Hint: you can either use the as.Date & as.Time format to convert the Request_time to the required format or can extract the first 2 digits of the string using substring function)
# In the bar chart (question 1), you'll be able to see 5 major time blocks based on the frequency of requests made at the city and airport. You have to now divide the request-time into 5 time-slots described below. Make an additional column "Time_Slot" which takes these 5 categorical values depending on the request time: (Hint: you can either use "elseif" function or a simple conditional for loop to achieve this.) 
# Pre_Morning
# Morning_Rush
# Day_Time
# Evening_Rush
# Late_Night
# Note: The division of time-slots may not have one right answer.
# 
# PROBLEM IDENTIFICATION:
# 1:Make a stacked bar chart where each bar represents a time slot 
#   and the y-axis shows the frequency of requests. 
#   Different proportions of bars should represent the completed, 
#   cancelled and no cars available out of the total customer requests. (Hint: ggplot)
# 2:Visually identify the 2 most pressing problems for Uber, 
#   out of the 15 possible scenarios (5 slots * 3 trip status).
# 
# Problem 1:
# 1:For the time slot when problem 1 exists, plot a stacked bar chart to find out if the problem is more severe for pick-up requests made at the airport or the city. As a next step, you have to determine the number of times this issue exists in that time slot. Also find the percentage breakup for the total number of issues in this time slot based on the pick-up point?
# 2:Now let's find out the gap between supply and demand. For this case, the demand is the number of trip requests made at the city, whereas the supply is the number of trips completed from city to the airport?
# 3:What do you think is the reason for this issue for the supply-demand gap? (Write the answer in less than 100 words).?
# 4:What is your recommendation to Uber (Not more than 50 words)?
# 
# Problem 2:
# 1:For the time slot when problem 2 exists, plot the stacked bar chart to find out if the issue is for pick-up request made at the airport or the city. Just like problem 1, find the percentage breakup for issue based on the pick-up point for the time slot in which problem 2 exists.
# 2:Now let's find out the gap between supply and demand. For this case, the demand is the number of trip requests made at the airport, whereas the supply is the number of trips completed from airport to the city.
# 3:What do you think is the reason for this issue for this supply-demand gap. (Not more than 100 words)?
# 4:What is your recommendation to Uber (Not more than 50 words)?

setwd("C://Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/CaseStudyUberSupplyDemandGap/")

library(lubridate)
library(ggplot2)
#-- Load data
request_data<-read.csv("Uber request data.csv", sep = ",", strip.white = TRUE )

#--Clean & reformat data as per need
request_data$count <- 1
request_data$reqhours <- hour(hms(request_data$Request.time))
hrs<-request_data$reqhours
request_data$reqhours <-  as.factor(request_data$reqhours)
request_data$Time_Slot <- as.factor(ifelse( (hrs <= 5), yes="Pre_Morning", no=ifelse( (5 < hrs) & (hrs<=10), yes="Morning_Rush", no=ifelse((10<hrs) & (hrs<=15), yes="Day_Time", no=ifelse((15 < hrs) & (hrs<=20), yes="Evening_Rush", no="Late_Night"  ) ) ) ))
request_data$reqhours <- hrs



#-- 1. Make a grouped bar chart depicting the hour-wise trip request made at city and airport respectively. 
hour_wise_city_airport_bgraph <- ggplot(data=request_data, aes(x=reqhours,fill=factor(Pickup.point))) + geom_bar( position="dodge")
hour_wise_city_airport_bgraph  + ggtitle("Hour-wise trip request made at city & airport")


#--Filter completed trips from requests
trips_data <- subset(request_data, Status=="Trip Completed")
trips_data_hrs <- aggregate(count~Time_Slot, trips_data, sum)

#-- 
#--2. Plot a bar chart for number of trips made during different time-slots in R
hour_wise_trips_bgraph <- ggplot(data=trips_data_hrs, aes(x=Time_Slot,y=count)) + geom_bar( stat= "identity") + ylab('Trips') + xlab('Time Slot')
hour_wise_trips_bgraph <- hour_wise_trips_bgraph  + geom_text(stat='identity',aes(y=count, label=count), position=position_stack( ), hjust=1.0, vjust=-0.3) 
hour_wise_trips_bgraph + ggtitle("Trips made during different time-slots")

#--3. Make a stacked bar chart where each bar represents a time slot and y axis
#--   shows the frequency of requests. Different proportions of bars should 
#--   represent the completed, cancelled and no cars available out of the total customer requests.

completed_cancelled <- ggplot(data=request_data, aes(x=as.character(Time_Slot), fill=Status, order = reqhours))
completed_cancelled <- completed_cancelled + geom_bar(position="stack" ) + geom_text(stat='count',aes(y=..count.., label=..count..), position=position_stack( ), hjust=1.0, vjust=-0.3)  + ylab('Requests') + xlab('Time Slot')
#completed_cancelled <- completed_cancelled + facet_wrap(~Pickup.point)
completed_cancelled + ggtitle("Completed, Cancelled and no cars available out of the total customer requests")


#--4. For the time slot when problem 1 exists, plot a stacked bar chart to find out if the problem 
#-- is more severe for pick-up requests made at the airport or the city. As a next step, you have 
#-- to determine the number of times this issue exists in that time slot. 

cancelled_data <- subset(request_data, Status=="Cancelled")

cancelled_city_airport <- ggplot(data=cancelled_data, aes(x=as.character(Time_Slot), fill=Pickup.point, order = reqhours))
cancelled_city_airport <- cancelled_city_airport + geom_bar(position="stack" ) + geom_text(stat='count',aes(y=..count.., label=..count..), position=position_stack( ), hjust=1.0, vjust=-0.3)  + ylab('Cancellations') + xlab('Time Slot')
cancelled_city_airport + ggtitle("Cancelled City Vs Airport")


#-- 5. For the time slot when problem 1 exists, plot a stacked bar chart to find out if the problem is more severe 
#-- for pick-up requests made at the airport or the city. 
#-- As a next step, you have to determine the number of times this issue exists in that time slot.
#-- 
Morning_Rush_all <-  subset(request_data, Time_Slot=="Morning_Rush" )

Morning_Rush <- subset(request_data, Time_Slot=="Morning_Rush" & Status=="Cancelled" )
Morning_Rush_cancelled_city_airport <- ggplot(data=Morning_Rush, aes(x=as.character(Time_Slot), fill=Pickup.point, order = reqhours))
Morning_Rush_cancelled_city_airport <- Morning_Rush_cancelled_city_airport + geom_bar(stat='count', position="stack" ) 
Morning_Rush_cancelled_city_airport <- Morning_Rush_cancelled_city_airport + geom_text(stat='count', check_overlap = T, position="stack", aes( label= paste(round( (..count..)*100/sum(..count..) ,1), "% [",(..count..), "]"), hjust=0, vjust=1) )  + ylab('Cancellations') + xlab('Time Slot')
Morning_Rush_cancelled_city_airport + ggtitle("Morning Cancelled City Vs Airport")

#-- Now let's find out the gap between supply and demand. 
#-- For this case, the demand is the number of trip requests made at the city, 
#-- whereas the supply is the number of trips completed from city to the airport.
supply_demand <- aggregate(count~Status+Pickup.point, Morning_Rush_all, sum)

Morning_Rush <- subset(request_data, Time_Slot=="Morning_Rush" & Status=="Cancelled" )




#1457
Demand_city_morning_rush <- sum(Morning_Rush_Morning_Rush$count[which(Morning_Rush_Morning_Rush$Pickup.point=="City")])
#434
Supply_city_morning_rush <- sum(Morning_Rush_Morning_Rush$count[which(Morning_Rush_Morning_Rush$Pickup.point=="City" & Morning_Rush_Morning_Rush$Status=="Trip Completed")])

#--Problem 2: Evening Rush No Cars
#--stacked bar chart to find out if the issue is for pick-up request made at the airport or the city.
#--Find the percentage breakup for issue based on the pick-up point for the time slot in which problem 2 exists.

Evening_Rush_all <-  subset(request_data, Time_Slot=="Evening_Rush" )
Evening_Rush_pointwise <-  aggregate(count~ Status+Pickup.point, Evening_Rush_all,sum)
#1570
Demand_city_Evening_rush <- sum(Evening_Rush_pointwise$count[which(Evening_Rush_pointwise$Pickup.point=="Airport")])

#361
Supply_city_Evening_rush <- sum(Evening_Rush_pointwise$count[which(Evening_Rush_pointwise$Pickup.point=="Airport" & Evening_Rush_pointwise$Status=="Trip Completed")])


#ind the percentage breakup for issue based on the pick-up point for the time slot in which problem 2 exists.
Evening_Rush <-  subset(request_data, Time_Slot=="Evening_Rush" & Status=="No Cars Available")

Evening_Rush_cancelled_city_airport <- ggplot(data=Evening_Rush, aes(x=as.character(Time_Slot), fill=Pickup.point, order = reqhours))
Evening_Rush_cancelled_city_airport <- Evening_Rush_cancelled_city_airport + geom_bar(stat='count', position="stack" ) 
Evening_Rush_cancelled_city_airport <- Evening_Rush_cancelled_city_airport + geom_text(stat='count', check_overlap = T, position="stack", aes( label= paste(round( (..count..)*100/sum(..count..) ,1), "% [",(..count..), "]"), hjust=0, vjust=1) )  + ylab('Cancellations') + xlab('Time Slot')
Evening_Rush_cancelled_city_airport + ggtitle("Evening No Cars City Vs Airport")
