#VIF to remove multi-collinearity
library(car)
vif(best_model) # elimniate BA

#model with BA excluded
model_1 =   glm(Playoffs ~. - BA, data = baseball_train[,-1], family = "binomial") 
summary(model_1)
vif(model_1) # eliminate OBP

#model with OBP excluded
model_2 = glm(Playoffs ~ RS + RA+SLG, data = baseball_train, family = "binomial")
summary(model_2)
vif(model_2)# eliminate SLG

#model with SLG excluded
model_3 = glm(Playoffs ~ RS + RA, data = baseball_train, family = "binomial")
summary(model_3)
vif(model_3) # no elimination

# model_3 is our final model
model_final = model_3
