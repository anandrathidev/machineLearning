## C-statistic
install.packages("Hmisc")
library(Hmisc)
final_model<- model_final
baseball_train$predicted_prob = predict(final_model,  type = "response")
# 1st argument is your vector of predicted probabilities, 
# 2nd observed values of outcome variable
rcorr.cens(baseball_train$predicted_prob,baseball_train$Playoffs) 

baseball_test$predicted_prob = predict(final_model, newdata = baseball_test,type = "response")
rcorr.cens(baseball_test$predicted_prob,baseball_test$Playoffs)

#KS-statistic
#install.packages("ROCR")
library(ROCR)

model_score <- prediction(baseball_train$predicted_prob,baseball_train$Playoffs)

model_perf <- performance(model_score, "tpr", "fpr")

ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf, "x.values")[[1]])

ks = max(ks_table)

which(ks_table == ks)


model_score_test <- prediction(baseball_test$predicted_prob,baseball_test$Playoffs)

model_perf_test <- performance(model_score_test, "tpr", "fpr")

ks_table_test <- attr(model_perf_test, "y.values")[[1]] - (attr(model_perf_test, "x.values")[[1]])

which(ks_table_test == ks_test)
