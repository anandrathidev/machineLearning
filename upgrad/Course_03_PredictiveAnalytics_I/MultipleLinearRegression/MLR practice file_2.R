#------------------------------Multiple Linear regression-----------------------------------
#-------------------------------------------------------------------------------------------

# Develop the first model 

model_1 <-lm(price~.,data=train[,-1])
summary(model_1)


#-------------------------------------------------------------------------------------------

# Apply the stepwise approach

step <- stepAIC(model_1, direction="both")

#-------------------------------------------------------------------------------------------


# Run the step object

step

#-------------------------------------------------------------------------------------------

# create a new model_2 after stepwise method

model_2 <-lm(formula = price ~ aspiration + enginelocation + wheelbase + 
               carlength + carwidth + carheight + curbweight + enginesize + 
               boreratio + peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodysedan + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

#-------------------------------------------------------------------------------------------

# summary of model_2 

summary(model_2)

#-------------------------------------------------------------------------------------------
