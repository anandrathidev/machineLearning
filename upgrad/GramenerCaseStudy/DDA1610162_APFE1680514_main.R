###############################################################################
# PGDDA Course-2 
# Group Assignment 1: Gramener Case Study
#-----------------------------------------------------------------------------
# Objective:
# The company wants to understand the driving factors behind loan default 
# (loan_status_1).  The company can utilise this knowledge for its portfolio 
# and risk assessment. Specifically, the company wants to determine which 
# driver variables are having the most influence on the tendency of loan default.
###############################################################################
setwd("c:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_02/GramenerCaseStudy/")
#
#--Setup environment-----------------------------------------------------------
#-download relevent packages
#install.packages("psych")
#install.packages("grid")
#install.packages("gridExtra")
#install.packages("corrplot")

#-load relevent libraries 
library(psych)
library(ggplot2)
library(grid)
library(gridExtra)
library(corrplot)

#
#--Begin loading data----------------------------------------------------------
loan_master_data <- read.csv("loan.csv", strip.white = TRUE, header = T, stringsAsFactors = F )

#=================CHECK POINT -1 ==============================================
#--Dataframe analysis----------------------------------------------------------
#
#-Lets view the columns in the df
str(loan_master_data)

loan_master_data <- loan_master_data[loan_master_data$id != "" & !is.na(loan_master_data$id),]
loan_master_data <- loan_master_data[loan_master_data$member_id != "" & !is.na(loan_master_data$member_id),]

#-Clearly we have many columns that are not needed
#-Lets choose only the driving attributes
driving_attributes <- c("annual_inc","loan_amnt", "funded_amnt", "int_rate", 
                        "grade", "dti", "emp_length","purpose","home_ownership",
                        "loan_status")
loan_col_filterd_data <- loan_master_data[,colnames(loan_master_data) %in% driving_attributes]

nrow(loan_col_filterd_data)

#-Lets view the new dataset
str(loan_col_filterd_data)
summary(loan_col_filterd_data)

#-We notice that there are some attribues that have not translated correctly
#-int_rate
loan_col_filterd_data$int_rate <- as.integer(sub("%","0",loan_col_filterd_data$int_rate))

#-grade [convert to factor]
loan_col_filterd_data$grade <- as.factor(loan_col_filterd_data$grade)

#-emp_length [convert to factor]
loan_col_filterd_data$emp_length <- as.factor(loan_col_filterd_data$emp_length)

#-home_ownership
loan_col_filterd_data$home_ownership <- as.factor(loan_col_filterd_data$home_ownership)

#-purpose
loan_col_filterd_data$purpose <- as.factor(loan_col_filterd_data$purpose)

#-Lets filter out any fully paid loans
data_filter_1 <- c("Fully Paid","Does not meet the credit policy. Status:Fully Paid")
loan_row_filterd_data <- loan_col_filterd_data[!loan_col_filterd_data$loan_status %in% data_filter_1,]

#-Lets view the new dataset
str(loan_row_filterd_data)
summary(loan_row_filterd_data)
nrow(loan_row_filterd_data)

#-Lets categorise some loans based on 
#-loan_status
current_new <- c("Current","In Grace Period")
charged_off <- c("Default","Charged Off","Does not meet the credit policy. Status:Charged Off")
late <- c("Late (16-30 days)","Late (31-120 days)")

loan_row_filterd_data$loan_status_1 <- ifelse(loan_row_filterd_data$loan_status %in% current_new , "current_new",
                                              ifelse(loan_row_filterd_data$loan_status %in% charged_off , "default_new",
                                                     ifelse (loan_row_filterd_data$loan_status %in% late , "late",
                                                                          NA)))
#-int_rate
loan_row_filterd_data$int_rate_grp <- ifelse(loan_row_filterd_data$int_rate <= 10, "Low",
                                             ifelse(loan_row_filterd_data$int_rate > 10 & loan_row_filterd_data$int_rate <= 18, "Medium",
                                                    ifelse(loan_row_filterd_data$int_rate > 18, "High",
                                                           NA)))
#-emp_length
junior <- c("< 1 year","1 year","2 years","3 years","4 years")
midlevel <- c("5 years","6 years","7 years","8 years")
senior <- c("9 years","10+ years")

loan_row_filterd_data$emp_len_grp <- ifelse(loan_row_filterd_data$emp_length %in% junior, "Junior",
                                            ifelse(loan_row_filterd_data$emp_length %in% midlevel, "Mid-level",
                                                   ifelse(loan_row_filterd_data$emp_length %in% senior, "Senior",
                                                          NA)))
#-Lets factorise the recent attributes as all are categorical in nature
loan_row_filterd_data$loan_status_1 <- as.factor(loan_row_filterd_data$loan_status_1)
loan_row_filterd_data$int_rate_grp <- as.factor(loan_row_filterd_data$int_rate_grp)
loan_row_filterd_data$emp_len_grp <- as.factor(loan_row_filterd_data$emp_len_grp)

#-This is our final data set 
loan_final_data <- loan_row_filterd_data

#
#--Data CleanUp---------------------------------------------------------------
#
summary(loan_final_data)
summary_loan_data <- describe(loan_final_data, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
                              type=3,check=TRUE)

#--Lets take care of the N/A values
#-
loan_final_data$emp_len_grp[is.na(loan_final_data$emp_len_grp)] <- "Junior"

summary(loan_final_data)
summary_loan_data <- describe(loan_final_data, na.rm = TRUE, interp=FALSE,skew = TRUE, ranges = TRUE,trim=.1,
                              type=3,check=TRUE)

#--Lets look at our data-------------------------------------------------------
annual_inc_density_remOut <- ggplot(loan_final_data, aes(x=annual_inc)) + geom_density() + 
  scale_size_area() + xlab("Annual Income") + ylab("Density") +  ggtitle("Annual Income - Density  plot")
annual_inc_boxplot_remOut <- ggplot(loan_final_data, aes(x=1, y=annual_inc)) + geom_boxplot() + 
  scale_size_area() + xlab("Annual Income") + ylab("Boxplot") +  ggtitle("Annual Income plot")

loan_amnt_density_remOut <- ggplot(loan_final_data, aes(x=loan_amnt)) + geom_density() + 
  scale_size_area() + xlab("Loan Amnt") + ylab("Density") +  ggtitle("Loan Amnt - Density  plot")
loan_amnt_boxplot_remOut <- ggplot(loan_final_data, aes(x=1, y=loan_amnt)) + geom_boxplot() + 
  scale_size_area() + xlab("Loan Amnt") + ylab("Boxplot") +  ggtitle("Loan Amnt plot")

funded_amnt_density_remOut <- ggplot(loan_final_data, aes(x=funded_amnt)) + geom_density() + 
  scale_size_area() + xlab("Funded Amnt") + ylab("Density") +  ggtitle("Funded Amnt plot")
funded_amnt_boxplot_remOut <- ggplot(loan_final_data, aes(x=1, y=funded_amnt)) + geom_boxplot() + 
  scale_size_area() + xlab("Funded Amnt") + ylab("Boxplot") +  ggtitle("Funded Amnt Box plot")

dti_density_remOut <- ggplot(loan_final_data, aes(x=dti)) + geom_density() + 
  scale_size_area() + xlab("DTI") + ylab("Density") +  ggtitle("DTI - Density  plot")
dti_boxplot_remOut <- ggplot(loan_final_data, aes(x=1, y=dti)) + geom_boxplot() + 
  scale_size_area() + xlab("DTI") + ylab("Boxplot") +  ggtitle("DTI Box plot")

grid.arrange(annual_inc_boxplot_remOut, annual_inc_density_remOut,
             funded_amnt_boxplot_remOut, funded_amnt_density_remOut,
             loan_amnt_boxplot_remOut, loan_amnt_density_remOut, 
             dti_boxplot_remOut, dti_density_remOut, ncol = 2, top = "Plots Before Outlier Treatment")
                              
#----Outliers Treatment--------------------------------------------------------
#-Based on teh above graphs we clearly see a need to perform some outlier cleaning
#-especially for the Annual income data
impute_outliers <- function(x) {
  q1<-unname(quantile(x, 0.25, na.rm = TRUE))
  q2<-unname(quantile(x, 0.75, na.rm = TRUE))
  iqr <- unname(q2-q1)*1.5
  x2<-x
  x2[x < (q1 - iqr) ] <- q1 - iqr
  x2[x > (q2 + iqr) ] <- q2 + iqr
  return(x2)
}

loan_final_data$annual_inc <- impute_outliers(loan_final_data$annual_inc)
loan_final_data$funded_amnt <- impute_outliers(loan_final_data$funded_amnt)
loan_final_data$dti <- impute_outliers(loan_final_data$dti)


loan_no_outlier_data <- loan_final_data

#-Lets make the numbers a little more manageable
loan_no_outlier_data$loan_amnt <- loan_no_outlier_data$loan_amnt / 1000
loan_no_outlier_data$funded_amnt <- loan_no_outlier_data$funded_amnt / 1000
loan_no_outlier_data$annual_inc <- loan_no_outlier_data$annual_inc / 1000

#=================CHECK POINT -2 ==============================================

#
loan_analysis_data <- loan_no_outlier_data
#
#--Univariate Analysis---------------------------------------------------------
#
#-Continuous
#
#-Lets begin by analysing the anual income
annual_inc_density <- ggplot(loan_analysis_data, aes(x=annual_inc)) 
annual_inc_density <- annual_inc_density + geom_density() 
annual_inc_density <- annual_inc_density + scale_size_area() 
annual_inc_density <- annual_inc_density + xlab("Annual Income in thousands (x1000)") + ylab("Density")   
annual_inc_density <- annual_inc_density + ggtitle("Annual Income - Density  plot") + guides(fill=guide_legend(title="New Legend Title"))

annual_inc_boxplot <- ggplot(loan_analysis_data, aes(x=1, y=annual_inc)) 
annual_inc_boxplot <- annual_inc_boxplot + geom_boxplot()
annual_inc_boxplot <- annual_inc_boxplot + scale_size_area() 
annual_inc_boxplot <- annual_inc_boxplot + xlab("Annual Income in thousands (x1000)") + ylab("Boxplot") 
annual_inc_boxplot <- annual_inc_boxplot + ggtitle("Annual Income plot")

grid.arrange(annual_inc_boxplot, annual_inc_density, ncol = 2, top = "Annual Income PLOTS")

#-Lets begin by analysing the loan amount
loan_amnt_density <- ggplot(loan_analysis_data, aes(x=loan_amnt)) 
loan_amnt_density <- loan_amnt_density + geom_density() 
loan_amnt_density <- loan_amnt_density + scale_size_area() 
loan_amnt_density <- loan_amnt_density + xlab("Loan Amnt") + ylab("Density") 
loan_amnt_density <- loan_amnt_density +  ggtitle("Loan Amnt - Density  plot")

loan_amnt_boxplot <- ggplot(loan_analysis_data, aes(x=1, y=loan_amnt)) 
loan_amnt_boxplot <- loan_amnt_boxplot + geom_boxplot() 
loan_amnt_boxplot <- loan_amnt_boxplot + scale_size_area() 
loan_amnt_boxplot <- loan_amnt_boxplot + xlab("Loan Amnt") + ylab("Boxplot") 
loan_amnt_boxplot <- loan_amnt_boxplot +  ggtitle("Loan Amnt plot")

grid.arrange(loan_amnt_boxplot, loan_amnt_density, ncol = 2, top = "Loan Amnt PLOTS")

#-Lets begin by analysing the funded amount
funded_amnt_density <- ggplot(loan_analysis_data, aes(x=funded_amnt)) 
funded_amnt_density <- funded_amnt_density + geom_density() 
funded_amnt_density <- funded_amnt_density + scale_size_area() 
funded_amnt_density <- funded_amnt_density + xlab("Funded Amnt") + ylab("Density") 
funded_amnt_density <- funded_amnt_density +  ggtitle("Funded Amnt plot")

funded_amnt_boxplot <- ggplot(loan_analysis_data, aes(x=1, y=funded_amnt)) 
funded_amnt_boxplot <- funded_amnt_boxplot + geom_boxplot() 
funded_amnt_boxplot <- funded_amnt_boxplot + scale_size_area() 
funded_amnt_boxplot <- funded_amnt_boxplot + xlab("Funded Amnt") 
funded_amnt_boxplot <- funded_amnt_boxplot + ylab("Boxplot") 
funded_amnt_boxplot <- funded_amnt_boxplot +  ggtitle("Funded Amnt Box plot")
grid.arrange(funded_amnt_boxplot, funded_amnt_density, ncol = 2, top = "Funded Amnt PLOTS")

#-Lets begin by analysing the DTI
dti_density <- ggplot(loan_analysis_data, aes(x=dti)) 
dti_density <- dti_density + geom_density() 
dti_density <- dti_density + scale_size_area() + xlab("DTI") + ylab("Density") 
dti_density <- dti_density + ggtitle("DTI - Density  plot")

dti_boxplot <- ggplot(loan_analysis_data, aes(x=1, y=dti)) 
dti_boxplot <- dti_boxplot + geom_boxplot() 
dti_boxplot <- dti_boxplot + scale_size_area() + xlab("DTI") + ylab("Boxplot") 
dti_boxplot <- dti_boxplot + ggtitle("DTI Box plot")

grid.arrange(dti_boxplot, dti_density, ncol = 2, top = "DTI PLOTS")

#
#-Categorical
# 
int_rate_plot <- ggplot(loan_analysis_data, aes(x=int_rate_grp)) 
int_rate_plot <- int_rate_plot + geom_bar() 
int_rate_plot <- int_rate_plot + scale_size_area() + xlab("interest rates") + ylab("Count") 
int_rate_plot <- int_rate_plot + ggtitle("Intrest Rates Groups (Low,Med,High)")

grade_plot <- ggplot(loan_analysis_data, aes(x=grade)) 
grade_plot <- grade_plot + geom_bar() 
grade_plot <- grade_plot + scale_size_area() + xlab("LC Grade") + ylab("Count") 
grade_plot <- grade_plot + ggtitle("LC Grade")

emp_len_plot <- ggplot(loan_analysis_data, aes(x=emp_len_grp)) 
emp_len_plot <- emp_len_plot + geom_bar() 
emp_len_plot <- emp_len_plot + scale_size_area() + xlab("Employment Length (yrs)") + ylab("Count") 
emp_len_plot <- emp_len_plot + ggtitle("Employment Length")

ownership_plot <- ggplot(loan_analysis_data, aes(x=home_ownership)) 
ownership_plot <- ownership_plot + geom_bar() 
ownership_plot <- ownership_plot + scale_size_area() + xlab("Home ownership") + ylab("Count") 
ownership_plot <- ownership_plot + ggtitle("Home Ownership Status")

loan_status_1_plot <- ggplot(loan_analysis_data, aes(x=loan_status_1)) 
loan_status_1_plot <- loan_status_1_plot + geom_bar()  
loan_status_1_plot <- loan_status_1_plot + scale_size_area() + xlab("Loan Status") + ylab("Count") 
loan_status_1_plot <- loan_status_1_plot + ggtitle("Loan Status")


grid.arrange(int_rate_plot,
             grade_plot ,
             emp_len_plot,
             ownership_plot ,
             loan_status_1_plot,
             ncol = 2, top = "Categorical Data Plots")
#
#--Multiivariate Analysis---------------------------------------------------------
#-Multivariate Analysis: This analysis will show how different variables interact 
#-with each other. 
#-This includes the following:
#-Finding correlations for all different pairs of continuous variables for e.g. dti v/s annual_inc. 
#            loan_amnt  funded_amnt  int_rate  annual_inc
#loan_amnt   1.0000000   0.9783484  0.3166643  0.4851695
#funded_amnt 0.9783484   1.0000000  0.3180512  0.4776764
#int_rate    0.3166643   0.3180512  1.0000000  0.1817332
#annual_inc  0.4851695   0.4776764  0.1817332  1.0000000

#
#--Lets subset the continous variables for ease.
loan_final_continiues <- subset(x=loan_analysis_data, select=c(loan_amnt,funded_amnt,int_rate,annual_inc))
#-Determine the correlation
loan_data_corr <- cor(loan_final_continiues)
#-Lets map it in terms of a plot 
corrplot.mixed(loan_data_corr, upper="number", lower="circle" , order ="hclust")

#
#-Lets plot the distributions for various continious variables to the loan status
#-This will gie us a picture of the trends that can be found for the various loan status values
#
#-Loan Status Vs Annual Income
loan_status_annual_inc <- ggplot(loan_analysis_data, aes(x=annual_inc, fill=factor(loan_status_1) )) 
loan_status_annual_inc <- loan_status_annual_inc + geom_histogram(position="fill", bins = 10) 
loan_status_annual_inc <- loan_status_annual_inc + xlab("Annual Inc") + ylab("Density") 
loan_status_annual_inc <- loan_status_annual_inc + ggtitle("Annual Inc - plot")
loan_status_annual_inc

#2---
loan_status_annual_inc <- ggplot(loan_analysis_data, aes(x=loan_status_1, y=annual_inc)) 
loan_status_annual_inc <- loan_status_annual_inc + geom_boxplot()
loan_status_annual_inc <- loan_status_annual_inc + xlab("Loan Status") + ylab("Annual Income in thousands (x1000)") 
loan_status_annual_inc <- loan_status_annual_inc + ggtitle("Annual Inc - plot")
loan_status_annual_inc

#-Loan Amount
loan_status_loan_amnt <- ggplot(loan_analysis_data, aes(x=loan_amnt, fill=factor(loan_status_1) )) 
loan_status_loan_amnt <- loan_status_loan_amnt + geom_histogram(position="fill", bins = 10)  
loan_status_loan_amnt <- loan_status_loan_amnt + xlab("Loan Amnt") + ylab("Density") 
loan_status_loan_amnt <- loan_status_loan_amnt + ggtitle("Loan Amnt - plot")

#2--
loan_status_loan_amnt <- ggplot(loan_analysis_data, aes(x=loan_status_1, y=loan_amnt)) 
loan_status_loan_amnt <- loan_status_loan_amnt + geom_boxplot()
loan_status_loan_amnt <- loan_status_loan_amnt + xlab("Loan Status") + ylab("Loan Amount in thousands (x1000)") 
loan_status_loan_amnt <- loan_status_loan_amnt + ggtitle("Loan Amount - plot")
loan_status_loan_amnt

#-Funded Amount
loan_status_funded_amnt <- ggplot(loan_analysis_data, aes(x=funded_amnt, fill=factor(loan_status_1) )) 
loan_status_funded_amnt <- loan_status_funded_amnt + geom_histogram(position="fill", bins = 10)  
loan_status_funded_amnt <- loan_status_funded_amnt + xlab("Fund Amnt") + ylab("Density") 
loan_status_funded_amnt <- loan_status_funded_amnt + ggtitle("Fund Amnt - plot")

#2--
loan_status_funded_amnt <- ggplot(loan_analysis_data, aes(x=loan_status_1, y=loan_amnt)) 
loan_status_funded_amnt <- loan_status_funded_amnt + geom_boxplot()
loan_status_funded_amnt <- loan_status_funded_amnt + xlab("Loan Status") + ylab("Funded Amount") 
loan_status_funded_amnt <- loan_status_funded_amnt + ggtitle("Funded Amount - plot")
loan_status_funded_amnt


#-DTI
loan_status_dti <- ggplot(loan_analysis_data, aes(x=dti, fill=factor(loan_status_1) )) 
loan_status_dti <- loan_status_dti + geom_histogram(position="fill", bins = 10)  
loan_status_dti <- loan_status_dti + xlab("DTI") + ylab("Density") 
loan_status_dti <- loan_status_dti + ggtitle("DTI - plot")

#2--
loan_status_dti <- ggplot(loan_analysis_data, aes(x=loan_status_1, y=loan_amnt)) 
loan_status_dti <- loan_status_funded_amnt + geom_boxplot()
loan_status_dti <- loan_status_dti + xlab("Loan Status") + ylab("DTI") 
loan_status_dti <- loan_status_dti + ggtitle("DTI - plot")
loan_status_dti


grid.arrange(loan_status_annual_inc,
             loan_status_loan_amnt,
             loan_status_funded_amnt,
             loan_status_dti, ncol = 2, top = "Distribution of Loan Status")

#
#-Lets plot the distributions for various continious variables to the intrest rate
#-This will gie us a picture of the trends that can be found for the various intrest rate values
#
#-Annual Income
int_rate_grp_annual_inc <- ggplot(loan_analysis_data, aes(x=annual_inc, fill=factor(int_rate_grp) )) 
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + geom_histogram(position="fill", bins = 10) 
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + xlab("Annual Inc") + ylab("Density") 
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + ggtitle("Annual Inc/Interest Rate group - plot")
#--2
int_rate_grp_annual_inc <- ggplot(loan_analysis_data, aes(x=int_rate_grp, y=annual_inc)) 
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + geom_boxplot()
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + xlab("Interst Rate Group") + ylab("Annual Income in thousands (x1000)") 
int_rate_grp_annual_inc <- int_rate_grp_annual_inc + ggtitle("Annual Inc/Interest Rate group - plot")
int_rate_grp_annual_inc

#-Loan Amount
int_rate_grp_loan_amnt <- ggplot(loan_analysis_data, aes(x=loan_amnt, fill=factor(int_rate_grp) )) 
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + geom_histogram(position="fill", bins = 10) 
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + xlab("Loan Amnt") + ylab("Density") 
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + ggtitle("Loan Amount/Interest Rate group - plot")
#--2
int_rate_grp_loan_amnt <- ggplot(loan_analysis_data, aes(x=int_rate_grp, y=loan_amnt)) 
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + geom_boxplot()
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + xlab("Interst Rate Group") + ylab("Loan Amount in thousands (x1000)") 
int_rate_grp_loan_amnt <- int_rate_grp_loan_amnt + ggtitle("Loan Amount/Interest Rate group - plot")
int_rate_grp_loan_amnt

#-Funded Amount
int_rate_grp_funded_amnt <- ggplot(loan_analysis_data, aes(x=funded_amnt, fill=factor(int_rate_grp) )) 
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + geom_histogram(position="fill", bins = 10)  
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + xlab("Fund Amnt") + ylab("Density") 
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + ggtitle("Fund Amnt/Interest Rate group - plot")

#--2
int_rate_grp_funded_amnt <- ggplot(loan_analysis_data, aes(x=int_rate_grp, y=funded_amnt)) 
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + geom_boxplot()
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + xlab("Interst Rate Group") + ylab("Funded Amount in thousands (x1000)") 
int_rate_grp_funded_amnt <- int_rate_grp_funded_amnt + ggtitle("Fund Amnt/Interest Rate group - plot")
int_rate_grp_funded_amnt

#-DTI
int_rate_grp_dti <- ggplot(loan_analysis_data, aes(x=dti, fill=factor(int_rate_grp) )) 
int_rate_grp_dti <- int_rate_grp_dti + geom_histogram(position="fill", bins = 10)  
int_rate_grp_dti <- int_rate_grp_dti + xlab("DTI") + ylab("Density") 
int_rate_grp_dti <- int_rate_grp_dti + ggtitle("DTI/Interest Rate group - plot")

#--2
int_rate_grp_dti <- ggplot(loan_analysis_data, aes(x=int_rate_grp, y=dti)) 
int_rate_grp_dti <- int_rate_grp_dti + geom_boxplot()
int_rate_grp_dti <- int_rate_grp_dti + xlab("Interst Rate Group") + ylab("DTI") 
int_rate_grp_dti <- int_rate_grp_dti + ggtitle("DTI/Interest Rate group - plot")
int_rate_grp_dti

grid.arrange(int_rate_grp_annual_inc,
             int_rate_grp_loan_amnt,
             int_rate_grp_funded_amnt,
             int_rate_grp_dti, ncol = 2, top = "Distribution of Interest Rate group")

#=================CHECK POINT -3 ==============================================
#
#--Test hypotheses (95 % conf. level) for two levels of loan_status_1 - default_new and current_new
default_new <-  subset(loan_analysis_data, loan_status_1=="default_new")
current_new <-  subset(loan_analysis_data, loan_status_1=="current_new")

annual_inc_loan_status <- t.test(x = current_new$annual_inc, y = default_new$annual_inc, conf.level = 0.95, alternative = "two.sided")
loan_amnt_loan_status <- t.test(x = current_new$loan_amnt, y = default_new$loan_amnt, conf.level = 0.95, alternative = "two.sided")
funded_amnt_loan_status <- t.test(x = current_new$funded_amnt, y =default_new$funded_amnt, conf.level = 0.95, alternative = "two.sided")
dti_loan_status <- t.test(x = current_new$dti, y =default_new$dti, conf.level = 0.95, alternative = "two.sided")

#--Test hypotheses (95 % conf. level) for two levels of int_rate_grp - high and low
high_rate <- subset(loan_analysis_data, int_rate_grp=="High")
low_rate <-  subset(loan_analysis_data, int_rate_grp=="Low")

annual_inc_rate <- t.test(x = high_rate$annual_inc, y = low_rate$annual_inc, conf.level = 0.95, alternative = "two.sided")
loan_amnt_rate <- t.test(x = current_new$loan_amnt, y = default_new$loan_amnt, conf.level = 0.95, alternative = "two.sided")
funded_amnt_rate <- t.test(x = high_rate$funded_amnt, y=low_rate$funded_amnt, conf.level = 0.95, alternative = "two.sided")
dti_loan_rate <- t.test(x = high_rate$dti, y=low_rate$dti, conf.level = 0.95, alternative = "two.sided")


#--Save Final data-------------------------------------------------------------
#
write.csv2(loan_analysis_data, "loan_final_data.csv", row.names = F)

sum(is.null(loan_analysis_data$funded_amnt))

##########################END##################################################