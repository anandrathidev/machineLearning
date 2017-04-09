#-------------------------------------------------------------------------------------------
#------------------------------Multiple Linear regression-----------------------------------
# Remove carcompanymazda

model_14 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + carbodywagon + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_14
summary(model_14)

#-------------------------------------------------------------------------------------------

# Remove carbodywagon

model_15 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanypeugot + carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_15

summary(model_15)

#-------------------------------------------------------------------------------------------

# Remove carCompanypeugot 

model_16 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + carCompanyisuzu + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model 16
summary(model_16)

#-------------------------------------------------------------------------------------------

# Remove carCompanyisuzu 
model_17 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + 
                carCompanymercury + carCompanymitsubishi + carCompanynissan + 
                carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_17

summary(model_17)

#-------------------------------------------------------------------------------------------

# Remove carCompanymercury 
model_18 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + 
                carCompanymitsubishi + carCompanynissan + 
                carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carCompanyvolvo + 
                carbodyhardtop + carbodyhatchback + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

summary(model_18)

#-------------------------------------------------------------------------------------------

# Remove carcompanyvolvo

model_19 <-lm(formula = price ~ aspiration + enginelocation +  
                enginesize + peakrpm + carCompanybmw + carCompanychevrolet + 
                carCompanydodge + carCompanyhonda + 
                carCompanymitsubishi + carCompanynissan + 
                carCompanyplymouth + carCompanysubaru + 
                carCompanyvolkswagen + carbodyhardtop + carbodyhatchback + 
                cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_19
summary(model_19)

#-------------------------------------------------------------------------------------------

# Remove carCompanychevrolet
model_20<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanydodge + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + carCompanysubaru + 
               carCompanyvolkswagen + carbodyhardtop + carbodyhatchback + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_20
summary(model_20)

#-------------------------------------------------------------------------------------------

#Remove carbodyhardtop

model_21<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanydodge + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + carCompanysubaru + 
               carCompanyvolkswagen + carbodyhatchback + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_21
summary(model_21)

#-------------------------------------------------------------------------------------------

# Remove carbodyhatchback
model_22<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanydodge + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + carCompanysubaru + 
               carCompanyvolkswagen + cylindernumberfive + fuelsystemmfi , data = train[, -1])

# summary of model_22
summary(model_22)

#-------------------------------------------------------------------------------------------

# Remove carCompanyvolkswagen

model_23<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanydodge + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + carCompanysubaru + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_23
summary(model_23)

#-------------------------------------------------------------------------------------------

# Remove carCompanysubaru

model_24<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanydodge + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1])

# summary of model_24
summary(model_24)

#-------------------------------------------------------------------------------------------

# Remove carCompanydodge

model_25<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + carCompanyplymouth + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# # summary of model_25
summary(model_25)

#-------------------------------------------------------------------------------------------

# Remove carCompanyplymouth 
model_26<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + carCompanyhonda + 
               carCompanymitsubishi + carCompanynissan + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_26
summary(model_26)

#-------------------------------------------------------------------------------------------

# Remove carCompanyhonda

model_27<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + 
               carCompanymitsubishi + carCompanynissan + 
               cylindernumberfive + fuelsystemmfi , data = train[, -1]) 

# summary of model_27
summary(model_27)

#-------------------------------------------------------------------------------------------

# Remove cylindernumberfive

model_28<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + 
               carCompanymitsubishi + carCompanynissan + 
               fuelsystemmfi , data = train[, -1]) 

# summary of model_28
summary(model_28)

#-------------------------------------------------------------------------------------------

# Remove fuelsystemmfi

model_29<-lm(formula = price ~ aspiration + enginelocation +  
               enginesize + peakrpm + carCompanybmw + 
               carCompanymitsubishi + carCompanynissan, data = train[, -1]) 

# summary of model_29
summary(model_29)

#-------------------------------------------------------------------------------------------
# Test the model on test dataset

Predict_1 <- predict(model_29,test[,-c(1,20)])

#-------------------------------------------------------------------------------------------

# Add a new column "test_predict" into the test dataset
test$test_price <- Predict_1
#-------------------------------------------------------------------------------------------

# calculate the test R2 

cor(test$price,test$test_price)
cor(test$price,test$test_price)^2
#-------------------------------------------------------------------------------------the End
