#-------------------------------------------------------------------------------------------
#------------------------------Multiple Linear regression-----------------------------------

# Now, "curbweight" 

model_5 <-lm(formula = price ~ aspiration + enginelocation +  
               carlength + carwidth + carheight + enginesize + 
               boreratio + peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

summary(model_5)

vif(model_5)

#-------------------------------------------------------------------------------------------

# Next is the car length, remove it from the model 

model_6 <-lm(formula = price ~ aspiration + enginelocation +  
               carwidth + carheight + enginesize + 
               boreratio + peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

summary(model_6)

vif(model_6)

#-------------------------------------------------------------------------------------------

# Check the summary of model_6

summary(model_6)

# Again check the vif of model_6

vif(model_6)

#-------------------------------------------------------------------------------------------
# Remove the boreratio variable

model_7 <-lm(formula = price ~ aspiration + enginelocation +  
               carwidth + carheight + enginesize + 
               peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemmpfi + fuelsystemspdi, data = train[, -1])

# check the summary of model_7

summary(model_7)
# Again check the VIF of model_7

vif(model_7)

#-------------------------------------------------------------------------------------------

# Remove fuelsystemmfi

model_8 <-lm(formula = price ~ aspiration + enginelocation +  
               carwidth + carheight + enginesize + 
               peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemspdi, data = train[, -1])
# lets see the summary

summary(model_8)

# Vif of model_8

vif(model_8)

#-------------------------------------------------------------------------------------------





# Remove car height variable

model_9 <-lm(formula = price ~ aspiration + enginelocation +  
               carwidth +enginesize + 
               peakrpm + carCompanybmw + carCompanychevrolet + 
               carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
               carCompanymercury + carCompanymitsubishi + carCompanynissan + 
               carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
               carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
               carbodyhardtop + carbodyhatchback + carbodywagon + 
               cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
               fuelsystemspdi, data = train[, -1])

# summary of model_9

summary(model_9)


# vif of model_9


vif(model_9)

#-------------------------------------------------------------------------------------------

# Correlation between engine size and carwidth variable

cor(train$enginesize,train$carwidth)

#-------------------------------------------------------------------------------------------

# Remove carwidth variable 

model_10 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + carbodywagon + 
                cylindernumberfive + cylindernumbersix + fuelsystemmfi + 
                fuelsystemspdi, data = train[, -1])

# check the summary of model_10

summary(model_10)

# Vif of model_10

vif(model_10)

#-------------------------------------------------------------------------------------------

# Remove cylindernumbersix variable

model_11 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + carbodywagon + 
                cylindernumberfive + fuelsystemmfi + 
                fuelsystemspdi, data = train[, -1])

# Summay of model_11

summary(model_11)

# vif of model_11

vif(model_11)

#-------------------------------------------------------------------------------------------

# Remove fuelsystemspdi variable 

model_12 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanytoyota + carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + carbodywagon + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summay of model_12

summary(model_12)

# vif of model_12

vif(model_12)

#-------------------------------------------------------------------------------------------


# remove carcompanytoyota 

model_13 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + carCompanymazda + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + carbodywagon + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_13

summary(model_13)

# vif of model_13

vif(model_13)

#-------------------------------------------------------------------------------------------
