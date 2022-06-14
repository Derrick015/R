################################################################################
#Installing and Loading Libraries
################################################################################
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(e1071)
library(ROCR)
library(tidytext)
library(gridExtra) 
library(grid)
library(classifierplots)
library(knitr)
library(xgboost)

################################################################################
# Data Importation and Exploration    
################################################################################

dat.yelp <- read_csv('./yelp.csv')
str(dat.yelp)


################################################################################
# Data Preprocessing
################################################################################
dat<-dat.yelp %>%
  mutate(stars=ifelse(stars >3,0,1)) %>%
  select(text,stars,user_id)



################################################################################
# Data Modelling
################################################################################

## Data partitioning

# Split data into train (80%) and test set (20%)
set.seed(76)
trainIndex <- createDataPartition(dat$stars, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
dat_train <- dat[trainIndex, ]
dat_test <- dat[-trainIndex, ]


## Natural Language Processing
# Doc2vec 
# define preprocessing function and tokenization function
prep_fun <- tolower
tok_fun <- word_tokenizer

# Apply preprocessing to train set
it_train <- itoken(dat_train$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   ids = dat_train$user_id)

# Apply preprocessing to test set
it_test <- itoken(dat_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = dat_test$user_id)


# Creating vocabulary and Document-term matrix
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)

# define tf-idf model
tfidf <- TfIdf$new()

# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- fit_transform(dtm_test, tfidf)

## Model Calibration
### Glmnet Model
# Train the model

set.seed(46, sample.kind = "Rounding")
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = dat_train[['stars']], 
                               family = 'binomial', 
                               alpha = 1, # Lambda set to 1
                               nfolds = 10, # 10 fold cross validation
                               thresh = 2 # convergence threshold of 2
)

# Store predictions
preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
y_hat<- ifelse(preds>0.5,1,0)

# Calibration Plot
y <- (dat_test$stars)
calibration_plot(y,preds)

# ROC curve
# Use the predictions to build a ROC curve to assess the performance of our model
fitpred = prediction(preds,dat_test$stars)
fitperf = performance(fitpred,"tpr","fpr")
plot(fitperf,col="purple",lwd=2,main="ROC Curve")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Confusion Matrix
cm.gbm <- confusionMatrix(as.factor(dat_test$stars),as.factor(y_hat))
cm.gbm


## Model Improvement
### Xgboost Model

# Convert to Xgb matrix
train_matrix <- xgb.DMatrix(dtm_train, label = dat_train$stars)

# Define parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

# Selecting the best iteration with 10 fold cross validation
set.seed(46, sample.kind = "Rounding")
xgbcv <- xgb.cv( params = params, data = train_matrix, nrounds = 100, nfold = 10, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F, label = dat_test$stars)

# Train model
set.seed(46, sample.kind = "Rounding")
xgb_fit <- xgboost(data = train_matrix,params = params, nrounds = xgbcv$best_iteration)

# Store predictions
xgb_pred <- predict(xgb_fit, dtm_test)
xgb_y_hat <- ifelse(xgb_pred > 0.5, 1, 0)

# Calibration Plot
y <- (dat_test$stars)
calibration_plot(y,xgb_pred)

# ROC curve
# Use the predictions to build a ROC curve to assess the performance of our model
fitpred = prediction(xgb_pred,dat_test$stars)
fitperf = performance(fitpred,"tpr","fpr")
plot(fitperf,col="purple",lwd=2,main="ROC Curve")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# Confusion Matrix
cm.xgb <- confusionMatrix(as.factor(dat_test$stars),as.factor(xgb_y_hat))
cm.xgb

################################################################################
# Results
################################################################################

rmse_results <- data_frame(Model = "Glmnet Model",
                           Accuracy = 0.798,
                           Balanced.Accuracy= 0.8097,
                           Sensitivity= 0.7903,
                           Specificity= 0.8291)


rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Xgboost Model",  
                                     Accuracy = 0.8355,
                                     Balanced.Accuracy = 0.8362,
                                     Sensitivity = 0.8348,
                                     Specificity = 0.8377))

rmse_results %>% kable()

