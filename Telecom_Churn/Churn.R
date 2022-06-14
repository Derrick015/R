#Data Set

# customerID:       Customer ID
# genderCustomer:   Categorical Data  (female, male)
# SeniorCitizen:    Categorical Data  (1, 0)
# PartnerWhether:   Categorical Data  (Yes, No)
# Dependents:       Categorical Data  (Yes, No)
# tenure:           Numerical Data
# PhoneService:     Categorical Data  (Yes, No)
# MultipleLines:    Categorical Data  (Yes, No, No phone service)
# InternetService:  Categorical Data  (DSL, Fiber optic, No)
# OnlineSecurity:   Categorical Data  (Yes, No, No internet service)
# OnlineBackup:     Categorical Data  (Yes, No, No internet service)
# DeviceProtection: Categorical Data  (Yes, No, No internet service)
# TechSupport:      Categorical Data  (Yes, No, No internet service)
# StreamingTV:      Categorical Data  (Yes, No, No internet service)
# StreamingMovies:  Categorical Data  (Yes, No, No internet service)
# Contract:         Categorical Data  (Month-to-month, One year, Two year)
# PaperlessBilling: Categorical Data  (Yes, No)
# PaymentMethod:    Categorical Data  (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges:   Numerical Data
# TotalCharges:     Numerical Data
# Churn:            Categorical Data  (Yes or No)

######################################################################################
#  Loading libraries:
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

library(ggplot2)
library(dplyr)
library(dplyr)
library(gridExtra)
library(corrplot)
library(dummies)
library(party)
library(MASS)
library(pROC)
library(caret)
library(rpart)
library(rpart.plot)
library(R6)
library(mice)
library(Amelia)
library(neuralnet)


######################################################################################
#  Data Importation 
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

my_data <- read.csv("Telco-Customer-Churn-MANM354.csv")
head(my_data)
dim(my_data)
str(my_data)

######################################################################################
#  Preprocessing
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

#Handle Na values
Total_NA <- sum(is.na(my_data))
Column_sums <- colSums(is.na(my_data))
Column_sums
Column_names <- colnames(my_data)[apply(my_data,2,anyNA)]
cat(Column_names)

# Impute missing data
impute <- mice(my_data,m=3,seed = 76)
# Complete impute data
my_data <- complete(impute,1) 
# Visualize missing data- No missing data after imputation
missmap(my_data)


#converting tenure into factor
my_data <- mutate(my_data,tenure_bin=tenure)
my_data$tenure_bin[my_data$tenure_bin >= 0 & my_data$tenure_bin <= 12]   <- "0 - 1 years"
my_data$tenure_bin[my_data$tenure_bin >= 13 & my_data$tenure_bin <= 24]  <- "1 - 2 years"
my_data$tenure_bin[my_data$tenure_bin >= 25 & my_data$tenure_bin <= 36]  <- "2 - 3 years"
my_data$tenure_bin[my_data$tenure_bin >= 37 & my_data$tenure_bin <= 48]  <- "3 - 4 years"
my_data$tenure_bin[my_data$tenure_bin >= 49 & my_data$tenure_bin <= 60]  <- "4 - 5 years"
my_data$tenure_bin[my_data$tenure_bin >= 61 & my_data$tenure_bin <= 72]  <- "5 - 6 years"
my_data$tenure_bin = as.factor(my_data$tenure_bin)

# Convert some words in to No 
my_data <- data.frame(lapply(my_data, function(x) {
  gsub("No internet service", "No", x)}))
my_data <- data.frame(lapply(my_data, function(x) {
  gsub("No phone service", "No", x)}))

# Convert numerical variables
my_data$tenure <- as.numeric(my_data$tenure)
my_data$MonthlyCharges<- as.numeric(my_data$MonthlyCharges)
my_data$TotalCharges <- as.numeric(my_data$TotalCharges)

# Convert certain variables into factors
fac.data<-
  c('gender','SeniorCitizen','Partner','Dependents','PhoneService',
           'MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport',
           'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
           'PaymentMethod','Churn','tenure_bin')
my_data[fac.data] <- lapply(my_data[fac.data], factor)


######################################################################################
#  Data Analysis
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
which(my_data$TotalCharges==0)
max(my_data$TotalCharges)
# Data Analysis
# 4.1 Exploration of the tenure_bin, MonthlyCharges, TotalCharges Variables
plot_A <- ggplot(my_data,aes(x = tenure_bin,fill=tenure_bin))+ geom_bar(width=0.4)
plot_B <- ggplot(my_data,aes(x = MonthlyCharges))+ geom_bar(fill="Steelblue",width=0.4)
plot_C <- ggplot(my_data,aes(x = TotalCharges))+ geom_histogram(fill="steelblue",binwidth =500)
grid.arrange(plot_A,plot_B,plot_C)


plot1 <- ggplot(data=my_data) + geom_bar(mapping = aes(x = gender,y=..prop..,group=2),fill="Steelblue",stat='count',width=0.4)
plot2 <- ggplot(data=my_data) + geom_bar(mapping = aes(x = PaperlessBilling,y=..prop..,group=2),fill="Steelblue",width=0.4,stat='count')
plot3 <- ggplot(data=my_data) + geom_bar(mapping = aes(x = Churn,y=..prop..,group=2),fill="Steelblue",width=0.4,stat='count')
grid.arrange(plot1,plot2,plot3,ncol=3)

# TotalCharges vs MonthlyCharges
my_data %>%
  ggplot(aes(MonthlyCharges,TotalCharges,col=Churn)) +
  geom_point(alpha=0.3)+
  theme_classic() +
  geom_smooth(method = "lm",se=F)

######################################################################################
#  Data Modelling
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# Preparation
# Select factor variables
factorVars <- which(sapply(my_data, is.factor)) #index vector numeric variables
factor.dat <- my_data[, factorVars]

# Select numeric variables
numbVars <- which(sapply(my_data, is.numeric)) #index vector numeric variables
numb.dat <- my_data[, numbVars]

# Creating dummy variables 
dummy.dat<- data.frame(sapply(factor.dat,function(x) data.frame(model.matrix(~x-1,data = my_data))[,-1]))
my_new_data_set <- cbind(my_data[,c(1,6,19,20)],dummy.dat)

# Min-Max Normalization
my_new_data_set$MonthlyCharges <- (my_new_data_set$MonthlyCharges - min(my_new_data_set$MonthlyCharges))/(max(my_new_data_set$MonthlyCharges) - min(my_new_data_set$MonthlyCharges))
my_new_data_set$TotalCharges <- (my_new_data_set$TotalCharges - min(my_new_data_set$TotalCharges))/(max(my_new_data_set$TotalCharges) - min(my_new_data_set$TotalCharges))


#Removing customerID &  tenure from
my_new_data_set$customerID <- NULL
my_new_data_set$tenure <- NULL

# Feature Selection

#spliting data
num = sample(2,nrow(my_new_data_set),replace = T,prob = c(0.7,0.3))
train_data <-  my_new_data_set[num==1,]
test_data  <-  my_new_data_set[num==2,]


# Class imbalance in train set
prop.table(table(train_data$Churn))
barplot(prop.table(table(train_data$Churn)),
        col = rainbow(7),
        ylim = c(0,1),
        main = 'Train Set Class Distribution')

# downsampling to correct class imbalance
train_data <- downSample(x = train_data[, -23],
                      y = as.factor(train_data$Churn))


colnames(train_data)[28]<-"Churn"

barplot(prop.table(table(train_data$Churn)),
        col = rainbow(7),
        ylim = c(0,1),
        main = ' Balanced Train Set Class Distribution')


# 1. Artificial Neural Network
set.seed(212)
nn<- neuralnet(Churn~.,
               data=train_data)

plot(nn)

# Prediction- Remove the predictor variable-Churn
output<- compute(nn,test_data[-23])
n1 <- output$net.result
n1<-as.data.frame(n1)
colnames(n1) = levels(as.factor(test_data$Churn))

# Prediction
n1$prediction = apply(n1,1,function(x) colnames(n1)[which.max(x)])

# Confusion Matrix
nn.cm<-confusionMatrix(as.factor(n1$prediction),as.factor(test_data$Churn))
nn.cm

#Decision Tree

# Convert columns into factor
fact_dat<-train_data[3:28]
train_data[3:28][sapply(train_data[3:28], is.numeric)] <- 
  lapply(train_data[3:28][sapply(train_data[3:28], is.numeric)], 
                                       as.factor)

test_data[3:28][sapply(test_data[3:28], is.numeric)] <- 
  lapply(test_data[3:28][sapply(test_data[3:28], is.numeric)], 
         as.factor)


#Fitting decision tree model to the training set 
set.seed(212)
rpartmodel<-rpart(Churn~., data = train_data,method = 'class')

# Decision tree visualizaton
rpart.plot(rpartmodel, extra =4)


#Predicting the test set results with the decision tree model
rpart.predictions<- predict(rpartmodel,test_data, type = 'class')
rpart.cm<-confusionMatrix(rpart.predictions,test_data$Churn)

######################################################################################
#  Results and Evaluation
#     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# Confusion Matrix
nn.cm
rpart.cm


#ROC
nn.roc <- roc(response = as.numeric(test_data$Churn), predictor = as.numeric(n1$prediction))
rpart.roc <- roc(response = test_data$Churn, predictor = as.numeric(rpart.predictions))

plot(nn.roc, col = "red", lty = 2, print.auc.y =0.9, print.auc=T,
     main = "ROC analysis for Neural Network - Decision Tree Model ")
plot(rpart.roc, col = "green", print.auc.y = 0.8, print.auc=T,
     lty = 3, add = TRUE)
legend(0.6,0.45, c('Neural Network','Decison Tree'),lty=c(1,1),lwd=c(2,2),col=c('red','green'))


mean(my_data$TotalCharges)
