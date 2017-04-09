#-------------------------------------------------------------------------------------------
#------------------------------Multiple Linear regression-----------------------------------

# Now calculate the VIF of this model 

install.packages("car")

library(car)

vif(model_2)
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------


# Check the summary of model_2

summary(model_2)

#-------------------------------------------------------------------------------------------

# Remove the variables from the model whose VIF is more than 2

# But check the maximum VIF and then the significance value of that variable, and then take the call of removing this variable
# Remove the "carbodysedan" variable 

model_3 <-lm(formula = price ~ aspiration + enginelocation + wheelbase + 
               carlength + carwidth + carheight + curbweight + enginesize + 
               boreratio + peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

#-------------------------------------------------------------------------------------------

# let's check the summary first
summary(model_3)
# Again calculate VIF of model_3

vif(model_3)

#-------------------------------------------------------------------------------------------

# Now the variable "wheelbase " is more collinear So we will  
# remove this variable now and develop the new model_4

model_4 <-lm(formula = price ~ aspiration + enginelocation +  
               carlength + carwidth + carheight + curbweight + enginesize + 
               boreratio + peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

summary(model_4)

vif(model_4)

#-------------------------------------------------------------------------------------------