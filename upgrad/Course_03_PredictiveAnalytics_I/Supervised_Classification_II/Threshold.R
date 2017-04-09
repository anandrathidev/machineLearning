# ROC curve
plot(model_perf,col = "red", lab = c(10,10,10))

#confusion matrix

confusionMatrix(as.numeric(baseball_train$predicted_prob > 0.5),baseball_train$Playoffs. positive = "1")

confusionMatrix(as.numeric(baseball_test$predicted_prob > 0.5),baseball_test$Playoffs, positive = "1")

foreign_worker_Dummy <- data.frame(model.matrix( ~foreign.worker, data = german_credit))

churn_dummy_gender  <- as.factor(churn_dummy_gender ,data=churn))
churn_dummy_SeniorCitizen  <- data.frame(model.matrix( ~SeniorCitizen ,data=churn))
churn_dummy_Partner  <- data.frame(model.matrix( ~Partner ,data=churn))
churn_dummy_Dependents  <- data.frame(model.matrix( ~Dependents ,data=churn))
churn_dummy_PhoneService  <- data.frame(model.matrix( ~PhoneService ,data=churn))
churn_dummy_Contract  <- data.frame(model.matrix( ~Contract ,data=churn))
churn_dummy_PaperlessBilling  <- data.frame(model.matrix( ~PaperlessBilling ,data=churn))
churn_dummy_PaymentMethod   <- data.frame(model.matrix( ~PaymentMethod ,data=churn))
churn_dummy_Churn  <- data.frame(model.matrix( ~Churn ,data=churn))
churn_dummy_MultipleLines  <- data.frame(model.matrix( ~MultipleLines ,data=churn))
churn_dummy_InternetService  <- data.frame(model.matrix( ~InternetService ,data=churn)) 
churn_dummy_OnlineSecurity  <- data.frame(model.matrix( ~OnlineSecurity ,data=churn))
churn_dummy_OnlineBackup  <- data.frame(model.matrix( ~OnlineBackup ,data=churn)) 
churn_dummy_DeviceProtection  <- data.frame(model.matrix( ~DeviceProtection ,data=churn))
churn_dummy_TechSupport  <- data.frame(model.matrix( ~TechSupport ,data=churn))
churn_dummy_StreamingTV  <- data.frame(model.matrix( ~StreamingTV ,data=churn))    
churn_dummy_StreamingMovies  <- data.frame(model.matrix( ~StreamingMovies ,data=churn))

churn_knn <- churn
churn_knn<- cbind(churn_knn,
churn_dummy_gender
churn_dummy_SeniorCitizen  ,
churn_dummy_Partner  ,
churn_dummy_Dependents  ,
churn_dummy_PhoneService  ,
churn_dummy_Contract  ,
churn_dummy_PaperlessBilling  ,
churn_dummy_PaymentMethod   ,
churn_dummy_Churn  ,
churn_dummy_MultipleLines  ,
churn_dummy_InternetService  ,
churn_dummy_OnlineSecurity  ,
churn_dummy_OnlineBackup  ,
churn_dummy_DeviceProtection  ,
churn_dummy_TechSupport  ,
churn_dummy_StreamingTV  ,
churn_dummy_StreamingMovies 
)


remove(churn_dummy_gender
churn_dummy_SeniorCitizen  ,
churn_dummy_Partner  ,
churn_dummy_Dependents  ,
churn_dummy_PhoneService  ,
churn_dummy_Contract  ,
churn_dummy_PaperlessBilling  ,
churn_dummy_PaymentMethod   ,
churn_dummy_Churn  ,
churn_dummy_MultipleLines  ,
churn_dummy_InternetService  ,
churn_dummy_OnlineSecurity  ,
churn_dummy_OnlineBackup  ,
churn_dummy_DeviceProtection  ,
churn_dummy_TechSupport  ,
churn_dummy_StreamingTV  ,
churn_dummy_StreamingMovies) 


(gender
SeniorCitizen  ,
Partner  ,
Dependents  ,
PhoneService  ,
Contract  ,
PaperlessBilling  ,
PaymentMethod   ,
Churn  ,
MultipleLines  ,
InternetService  ,
OnlineSecurity  ,
OnlineBackup  ,
DeviceProtection  ,
TechSupport  ,
StreamingTV  ,
StreamingMovies) 
