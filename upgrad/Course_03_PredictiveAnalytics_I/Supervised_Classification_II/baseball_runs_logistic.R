#View(baseball_runs)
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\Supervised_Classification_II/\\")
baseball_runs <- read.csv("baseball.csv", sep = ",")
summary(baseball_runs)

# check missing values 

na_vals <- sum(is.na(baseball_runs))


set.seed(2)
s=sample(1:nrow(baseball_runs),0.7*nrow(baseball_runs))
base_train = baseball_runs[s,]
base_test = baseball_runs[-s,]

model_runs <- glm(Playoffs ~ RS, family = binomial, data = base_train)
summary(model_runs)


