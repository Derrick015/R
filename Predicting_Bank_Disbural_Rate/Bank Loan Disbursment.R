##############################################################################
# Installing and loading libraries
##############################################################################
if (!require(tidyverse)) install.packages('tidyverse') # Data manipulation, exploration and visualization
library(tidyverse)

if (!require(caret)) install.packages('caret') # Classification and regression
library(caret)

if (!require(ROSE)) install.packages('ROSE')# Package for Binary Imbalance
library(ROSE)

if (!require(rpart)) install.packages('rpart') # Prediction: Decision Tree
library(rpart)

if (!require(rpart.plot)) install.packages('rpart.plot')# Plot Decision Tree
library(rpart.plot)

if (!require(randomForest)) install.packages('randomForest')# Classification algorithm
library(randomForest)


##############################################################################
# Data importation and dimension assessment and Structure
##############################################################################

Train_set <- read_csv("./Train.csv")
Test_set <- read_csv("./Test.csv")
dim(Train_set)
dim(Test_set)

# Save the IDs in the train and test set
Train_id = Train_set$ID
Test_id = Test_set$ID

# Remove the Logged-in variable as it has no bearing in developing the prediction model
Train_set<-Train_set[-25]

# Create and fill the response variable in the test set with NA
Test_set$Disbursed<-NA

# Bind the train and test set
Full_set <- rbind(Train_set,Test_set)

# Remove ID as it will not be of use in prediction
Full_set<- Full_set[-1]

# Structure of the full set
str(Full_set)

##############################################################################
# Exploratory Data Analysis
##############################################################################

# Disbursement per City
Train_set %>%
  group_by(City) %>%
  summarise(Disbursement_per_city=sum(as.numeric(Disbursed))) %>%
  arrange(desc(Disbursement_per_city)) %>% top_n(10) %>%
  ggplot(aes(x=reorder(City,+Disbursement_per_city),y=Disbursement_per_city, fill=City)) +
  geom_col() +
  theme_bw() + theme(legend.position="none") +
  labs(x = 'City', y = '',
       title = 'Disbursement per City') +
  coord_flip()

##############################################################################
# Data Preprocessing
##############################################################################
## Handling missing data ##

# Number and percentage of missing data in each variables.
colSums(is.na(Full_set))
colMeans(is.na(Full_set))

# Remove variables with more than 50% missing data
Full_set<-Full_set[, which(colMeans(!is.na(Full_set)) > 0.5)]

# Remove some chaaracter varibales that will not be used in prediction
Full_set<-Full_set %>% dplyr::select(!c(City,DOB,Lead_Creation_Date,
                                        Employer_Name,Salary_Account))

# Transform remaining character variables to factors
factor_vars <- c('Gender','Mobile_Verified','Var1','Filled_Form',
                 'Device_Type','Var2','Source','Var4','Disbursed')

Full_set[factor_vars] <- lapply(Full_set[factor_vars], function(x) as.factor(x))


# Replace missing NA in numeric variables with the median
## Replace NA values in Loan_Amount_Applied with the median
Full_set$Loan_Amount_Applied =ifelse(is.na(Full_set$Loan_Amount_Applied),
                                     median(Full_set$Loan_Amount_Applied,na.rm=T),
                                     Full_set$Loan_Amount_Applied)

## Replace NA values in Loan_Tenure_Applied with the median
Full_set$Loan_Tenure_Applied =ifelse(is.na(Full_set$Loan_Tenure_Applied),
                                     median(Full_set$Loan_Tenure_Applied,na.rm=T),
                                     Full_set$Loan_Tenure_Applied)

## Replace NA values in Existing_EMI with the median
Full_set$Existing_EMI =ifelse(is.na(Full_set$Existing_EMI),
                              median(Full_set$Existing_EMI,na.rm=T),
                              Full_set$Existing_EMI)

## Replace Loan_Amount_Submitted with 0 since there was no mobile verification so there was no submission.
Full_set$Loan_Amount_Submitted[is.na(Full_set$Loan_Amount_Submitted)] = 0

## Replace Loan_Tenure_Submitted with 0 since there was no mobile verification so there was no submission.
Full_set$Loan_Tenure_Submitted[is.na(Full_set$Loan_Tenure_Submitted)] = 0

# All NA have been replaced except in the test set of the dependent variable which was implanted there.
colMeans(is.na(Full_set))


##############################################################################
# Model building
##############################################################################

# Divide into test and train set
train <- Full_set[1:87020,]
test <- Full_set[87021:124737,]

#Data Partitioning Trainset into Train: 70%, Validation: 30%
set.seed(76)
y <- train$Disbursed
validation_index <- createDataPartition(y, times = 1, p = 0.3, list = FALSE)
validation <- train[validation_index,]
training <- train[-validation_index,]

# Class imbalance assessment
# Serious imbalance in training set
table(training$Disbursed)
barplot(prop.table(table(training$Disbursed)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Imbalanced Class Distribution')

# Over and under sampling to correct class imbalance
new_train_set<- ovun.sample(Disbursed~., data = training, 'both',# both means oversampling
                            p=0.5,# probability of the real class
                            seed =222,#set seed to 222 for repeatability
                            N=60913)$data #60913 represents the total number of observations in the trainset


# Balanced class achieved
table(new_train_set$Disbursed)
barplot(prop.table(table(new_train_set$Disbursed)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Balanced Class Distribution')



## Decision Tree ##
#Fitting decision tree model to the training set 
set.seed(46, sample.kind = "Rounding")
fit_rpart<-rpart(Disbursed~., data = new_train_set,
                 method = 'class')

# Plot decision tree 
rpart.plot(fit_rpart, extra = 4)

#Predicting the validation set results with the decision tree model
pred_rpart<- predict(fit_rpart,validation,type = 'class')

#Evaluate Model performance
confusionMatrix(pred_rpart,validation$Disbursed)
# Balanced Accuracy : 0.7578  
# Sensitivity : 0.7145         
# Specificity : 0.8010

##Random Forest
# Fitting Random Forest Classification to the Training set
set.seed(46, sample.kind = "Rounding")
fit_rf<-randomForest(Disbursed~., 
                     data= new_train_set)

#Predicting the validation set results with the random forest model
pred_rf<- predict(fit_rf,validation)

# Evaluate Model performance.
confusionMatrix(pred_rf,validation$Disbursed)
# Balanced Accuracy : 0.60189
# Sensitivity : 0.93415         
# Specificity : 0.26963


# Retrain best model (decision tree has the highest balanced accuracy) on whole train set
# Expected Imbalance in whole train set
table(train$Disbursed)
barplot(prop.table(table(train$Disbursed)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Imbalanced Class Distribution')

# Over and under sampling to correct class imbalance in whole train set
new_train_set1<- ovun.sample(Disbursed~., data = train, 'both',# over means oversampling
                             p=0.5,# probability of the real class
                             seed =222,#set seed to 222 for repeatability
                             N=87020)$data #60913 represents the total number of observations in the trainset

## Decision Tree ##
#Fitting decision tree model to the train set 
set.seed(46, sample.kind = "Rounding")
fit_rpart1<-rpart(Disbursed~., data = new_train_set1,
                  method = 'class')

#Predicting the test set results with the decision tree model
pred_rpart1<- predict(fit_rpart1,test,type = 'class')



##############################################################################
# Results
##############################################################################

# Saving decision tree prediction results for test set
df<-data.frame(Id=Test_id, Disbursed=pred_rpart1)
write.csv(df, "results_dt.csv", row.names = FALSE)