
setwd(dir = "/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/SupervisedClassification_I")
fish <- read.csv("a_mammals.csv")
fish$dist2 <- sqrt(  (fish$weight-624.85)^2 + (fish$length-4.01)^2  )
head(fish[order(fish$dist2,decreasing = F),],6)

