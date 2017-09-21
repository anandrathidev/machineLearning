

dbinom(1, size=4, prob=0.2) + dbinom(2, size=4, prob=0.2) + dbinom(3, size=4, prob=0.2) + dbinom(4, size=4, prob=0.2)
install.packages("ggplot2")
library(ggplot2)
library(MASS)

# Generate gamma rvs

x <- rgamma(100000, shape = 67, rate =6)

den <- density(x)

dat <- data.frame(x = den$x, y = den$y)

# Plot density as points

ggplot(data = dat, aes(x = x, y = y)) + 
  geom_point(size = 3) +   theme_classic()  + 
  geom_point(size = 3) +   theme_classic()

x <- rgamma(10000, shape = 8, rate =2)

den <- density(x)

dat <- data.frame(x = den$x, y = den$y)
ggplot(data = dat, aes(x = x, y = y)) + 
  geom_point(size = 3) +   theme_classic()  + 
  geom_point(size = 3) +   theme_classic()

upper <- qgamma(.90, shape=6*12.5, scale=1/6)
lower <- qgamma(.90, shape=6*12.5, scale=1/6, lower.tail=FALSE)

pbeta(q, shape1, shape2, ncp = 0, lower.tail = TRUE, log.p = FALSE)

pbeta(1, shape1 = 1, shape2 = 2)

pbeta(0.5, shape1=2+3, shape2=2+7, ncp = 0, lower.tail = TRUE, log.p = FALSE)


1/16 / 1/20  = b
20/16

1/20
6/93.5


?pgamma()

# Gausiian Distribution 
postdata = c(94.6, 95.4, 96.2, 94.9, 95.9)
n <-  length(postdata)
meanx =  mean(postdata)
varx =  var(postdata)
sqrt(varx)
MuPrior = 100
priorsig = 0.5
sig = 0.5
galpha =  1/(priorsig^2)
gbeta = n/(sig^2)

MuPost = (galpha* MuPrior + gbeta*meanx) / (galpha + gbeta)
SigPost = 1 / (galpha + gbeta)

SigPost^2

