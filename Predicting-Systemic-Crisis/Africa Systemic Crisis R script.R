#Load Packages and libraries
if (!require(tidyverse)) install.packages('tidyverse') #Data manipulation, exploration and visualization
library(tidyverse)
if (!require(caret)) install.packages('caret')#Classification and regression
library(caret)
if (!require(ROSE)) install.packages('ROSE')# Package for Binary Imbalance
library(ROSE)
if (!require(randomForest)) install.packages('randomForest')# Classification algorithm
library(randomForest)
if (!require(e1071)) install.packages('e1071')#SVM, Naive Bayes, Parameter Tuning
library(e1071)
if (!require(knitr)) install.packages('knitr')# Dynamic Report Generation
library(knitr)
if (!require(car)) install.packages('car') # Checking Multicollinearity
library(car)
if (!require(ROCR)) install.packages('ROCR') # Prediction: ROC Curve
library(ROCR)
if (!require(rpart)) install.packages('rpart') # Prediction: Decision Tree
library(rpart)
if (!require(rpart.plot)) install.packages('rpart.plot')# Plot Decision Tree
library(rpart.plot)


#Importing data
url<-'https://raw.githubusercontent.com/Derrick015/Predicting-Systemic-Crisis/main/african_crises.csv'
african_crises<-read.csv(url)

#Structure
str(african_crises)

#Data Preprocessing and cleaning.......................................................................
# Convert categorical variables from numeric to factors
african_crises$systemic_crisis<-as.factor(african_crises$systemic_crisis)
african_crises$domestic_debt_in_default<-as.factor(african_crises$domestic_debt_in_default)
african_crises$sovereign_external_debt_default<-as.factor(african_crises$sovereign_external_debt_default)
african_crises$independence<-as.factor(african_crises$independence)
african_crises$inflation_crises<-as.factor(african_crises$inflation_crises)
african_crises$banking_crisis<-as.factor(african_crises$banking_crisis)
african_crises$currency_crises<-as.factor(african_crises$currency_crises)

str(african_crises)

#Replacing erronous level 2 with 1 in currency crisis column
african_crises$currency_crises[african_crises$currency_crises==2]<-1
levels(african_crises$currency_crises)

#Dropping unused level 2
african_crises$currency_crises<-factor(african_crises$currency_crises)
levels(african_crises$currency_crises)


#Data exploration and visualization

#Number of total obervations per country
african_crises %>%
  group_by(country) %>%
  summarise(total=n()) %>%
  arrange(desc(total)) %>%
  ggplot(aes(x=reorder(country,+total),y=total, fill=country)) +
  geom_col() +
  theme_bw() + theme(legend.position="none") +
  labs(x = 'Country', y = '',
       title = 'Numbers of observations per country',
       subtitle = 'Period (1860-2014)') +
  coord_flip()

#Number of systematic crisis per country
african_crises %>% 
  mutate(systemic_crisis = ifelse(systemic_crisis == 1, 'yes', 'no')) %>% 
  group_by(country) %>% 
  count(systemic_crisis) %>% 
  spread(systemic_crisis, n) %>% 
  mutate(yes = ifelse(is.na(yes), 0, yes)) %>% 
  gather('no', 'yes', key = 'systemic_crisis', value = "cases") %>% 
  arrange(desc(cases)) %>% 
  kable()

african_crises %>% 
  mutate(systemic_crisis = ifelse(systemic_crisis == 1, 'yes', 'no')) %>% 
  group_by(country) %>% 
  count(systemic_crisis) %>% 
  spread(systemic_crisis, n) %>% 
  mutate(yes = ifelse(is.na(yes), 0, yes)) %>% 
  gather('no', 'yes', key = 'systemic_crisis', value = "cases") %>% 
  data.frame() %>% 
  ggplot(aes(x = reorder(country, +cases), y = cases, 
             fill = systemic_crisis)) + geom_col(position = 'dodge') +
  theme_bw() +
  labs(x = 'Country', y = '',
       title = 'Numbers of systemic crisis per country',
       subtitle = 'Period (1860-2014)') +
  theme(legend.position="bottom") +
  scale_fill_discrete(name = "Systemic Crisis", labels = c('No', 'Yes')) +
  coord_flip()


#Modeling.......................................................................

#selecting relevant variables for model
crisis<-african_crises %>% 
  dplyr::select(-c(case,cc3,country,year))


# class imbalance problem detected
prop.table(table(crisis$systemic_crisis))
barplot(prop.table(table(crisis$systemic_crisis)),
        col = rainbow(4),
        ylim = c(0,1),
        main = 'Class Distribution')



#Data Partitioning Train: 80%, Test: 20%
set.seed(46, sample.kind = "Rounding")
y<-crisis$systemic_crisis
test_index <- createDataPartition(y, times = 1, p = 0.2, list = FALSE)
test_set <- crisis[test_index,]
train_set <- crisis[-test_index,]


#Imbalance in training set
table(train_set$systemic_crisis)
barplot(prop.table(table(train_set$systemic_crisis)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Training Set Class Distribution')


#Imbalance in training set
prop.table(table(train_set$systemic_crisis))
barplot(prop.table(table(crisis$systemic_crisis)),
        col = rainbow(2),
        ylim = c(0,1),
        main = 'Train Set Class Distribution')



# Over sampling to correct class imbalance
new_train_set<- ovun.sample(systemic_crisis~., data = train_set, 'over',
                            seed =222,#set seed to 222 for repeatability
                            N=1562)$data

#Imbalance corrected
table(new_train_set$systemic_crisis)
barplot(prop.table(table(new_train_set$systemic_crisis)),
        col = rainbow(3),
        ylim = c(0,0.7),
        main = 'Balanced Class Distribution in Training Set')



#Logistic Regression

# correlation of numeric features
cor(crisis[,unlist(lapply(crisis,is.numeric))])

#syd<-chisq.test(crisis$systemic_crisis,crisis$domestic_debt_in_default)$p.value
sys<-chisq.test(crisis$systemic_crisis,crisis$sovereign_external_debt_default)$p.value
syind<-chisq.test(crisis$systemic_crisis,crisis$independence)$p.value
syc<-chisq.test(crisis$systemic_crisis,crisis$currency_crises)$p.value
syi<-chisq.test(crisis$systemic_crisis,crisis$inflation_crises)$p.value
syb<-chisq.test(crisis$systemic_crisis,crisis$banking_crisis)$p.value

dsys<-chisq.test(crisis$domestic_debt_in_default,crisis$systemic_crisis)$p.value
ds<-chisq.test(crisis$domestic_debt_in_default,crisis$sovereign_external_debt_default)$p.value
dind<-chisq.test(crisis$domestic_debt_in_default,crisis$independence)$p.value
dc<-chisq.test(crisis$domestic_debt_in_default,crisis$currency_crises)$p.value
di<-chisq.test(crisis$domestic_debt_in_default,crisis$inflation_crises)$p.value
db<-chisq.test(crisis$domestic_debt_in_default,crisis$banking_crisis)$p.value

ssys<-chisq.test(crisis$sovereign_external_debt_default,crisis$systemic_crisis)$p.value
sd<-chisq.test(crisis$sovereign_external_debt_default,crisis$domestic_debt_in_default)$p.value
sind<-chisq.test(crisis$sovereign_external_debt_default,crisis$independence)$p.value
sc<-chisq.test(crisis$sovereign_external_debt_default,crisis$currency_crises)$p.value
si<-chisq.test(crisis$sovereign_external_debt_default,crisis$inflation_crises)$p.value
sb<-chisq.test(crisis$sovereign_external_debt_default,crisis$banking_crisis)$p.value

indsys<-chisq.test(crisis$independence,crisis$systemic_crisis)$p.value
indd<-chisq.test(crisis$independence,crisis$domestic_debt_in_default)$p.value
inds<-chisq.test(crisis$independence,crisis$sovereign_external_debt_default)$p.value
indc<-chisq.test(crisis$independence,crisis$currency_crises)$p.value
indi<-chisq.test(crisis$independence,crisis$inflation_crises)$p.value
indb<-chisq.test(crisis$independence,crisis$banking_crisis)$p.value

csys<-chisq.test(crisis$currency_crises,crisis$systemic_crisis)$p.value
cd<-chisq.test(crisis$currency_crises,crisis$domestic_debt_in_default)$p.value
cs<-chisq.test(crisis$currency_crises,crisis$sovereign_external_debt_default)$p.value
cind<-chisq.test(crisis$currency_crises,crisis$independence)$p.value
ci<-chisq.test(crisis$currency_crises,crisis$inflation_crises)$p.value
cb<-chisq.test(crisis$currency_crises,crisis$banking_crisis)$p.value

isys<-chisq.test(crisis$inflation_crises,crisis$systemic_crisis)$p.value
id<-chisq.test(crisis$inflation_crises,crisis$domestic_debt_in_default)$p.value
is<-chisq.test(crisis$inflation_crises,crisis$sovereign_external_debt_default)$p.value
iind<-chisq.test(crisis$inflation_crises,crisis$independence)$p.value
ic<-chisq.test(crisis$inflation_crises,crisis$currency_crises)$p.value
ib<-chisq.test(crisis$inflation_crises,crisis$banking_crisis)$p.value

bsys<-chisq.test(crisis$banking_crisis,crisis$systemic_crisis)$p.value
bd<-chisq.test(crisis$banking_crisis,crisis$domestic_debt_in_default)$p.value
bs<-chisq.test(crisis$banking_crisis,crisis$sovereign_external_debt_default)$p.value
bind<-chisq.test(crisis$banking_crisis,crisis$independence)$p.value
bc<-chisq.test(crisis$banking_crisis,crisis$currency_crises)$p.value
bi<-chisq.test(crisis$banking_crisis,crisis$inflation_crises)$p.value

cormatrix <- matrix(c(0,syd,sys,syind,syc,syi,syb,
                      dsys,0,ds,dind,dc,di,db,
                      ssys,sd,0,sind,sc,si,sb,
                      indsys,indd,inds,0,indc,indi,indb,
                      csys,cd,cs,cind,0,ci,cb,
                      isys,id,is,iind,ic,0,ib,
                      bsys,bd,bs,bind,bc,bi,0),
                    7,7,byrow = TRUE)

row.names(cormatrix) = colnames(cormatrix) = c('systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default',
                                               'independence','currency_crises','inflation_crises','banking_crisis')
cormatrix



# Fitting Logistic Regression on Training set
set.seed(46, sample.kind = "Rounding")
fit_glm0<- glm(formula = systemic_crisis ~.,data = new_train_set ,family = 'binomial')
summary(fit_glm0)


# Fitting Logistic Regression on Training set without insignificant variables
set.seed(46, sample.kind = "Rounding")
fit_glm1<- glm(formula = systemic_crisis ~ domestic_debt_in_default+
                 sovereign_external_debt_default+
                 gdp_weighted_default+
                 currency_crises+
                 inflation_crises+
                 banking_crisis+
                 exch_usd
               ,data = new_train_set ,family = 'binomial')

summary(fit_glm1)


# Randomize training set order
set.seed(46, sample.kind = "Rounding")
new_train_set <- new_train_set[order(runif(nrow(new_train_set))), ]

#Fit model with radominzed data
fit_glm1<- glm(formula = systemic_crisis ~ domestic_debt_in_default+
                 sovereign_external_debt_default+
                 gdp_weighted_default+
                 currency_crises+
                 inflation_crises+
                 banking_crisis+
                 exch_usd
               ,data = new_train_set ,family = 'binomial')


#DurbinWatsonTest
set.seed(46, sample.kind = "Rounding")
durbinWatsonTest(fit_glm1)


# Predicting the test set results with the glm model
pred_glm_1<- predict(fit_glm1,test_set, type = 'response')
y_hat_glm1<- ifelse(pred_glm_1>0.5,1,0)
confusionMatrix(as.factor(y_hat_glm1),test_set$systemic_crisis)


# Use the predictions to build a ROC curve to assess the performance of our model
fitpred = prediction(pred_glm_1, test_set$systemic_crisis)
fitperf = performance(fitpred,"tpr","fpr")
plot(fitperf,col="green",lwd=2,main="ROC Curve")
abline(a=0,b=1,lwd=2,lty=2,col="gray")



#Decision Tree
#Fitting decision tree model to the training set 
set.seed(46, sample.kind = "Rounding")
fit_rpart<-rpart(systemic_crisis~., data = new_train_set,
                 method = 'class')
# Decision tree visualizaton
rpart.plot(fit_rpart, extra = 4)


#Predicting the test set results with the decision tree model
pred_rpart<- predict(fit_rpart,test_set,type = 'class')
confusionMatrix(pred_rpart,test_set$systemic_crisis)

# Using cross validation to choose cp
set.seed(46, sample.kind = "Rounding")
fit_rpart_cv<- train(systemic_crisis~., data = new_train_set,
                     method = 'rpart',
                     tuneGrid = data.frame(cp = seq(0, 0.1, len = 30)))

# Best tune
plot(fit_rpart_cv)
fit_rpart_cv$bestTune

# Tree Visualization
rpart.plot(fit_rpart_cv$finalModel, extra = 4)

#Prediction the test set results with cross validated decision tree model
pred_rpart_cv<- predict(fit_rpart_cv,test_set)
confusionMatrix(pred_rpart_cv,test_set$systemic_crisis)


##Random Forest

# Fitting Random Forest Classification to the Training set
set.seed(46, sample.kind = "Rounding")
fit_rf<-randomForest(systemic_crisis~., 
                     data= new_train_set)

#Predicting the test set results with the random forest model
pred_rf<- predict(fit_rf,test_set)
confusionMatrix(pred_rf,test_set$systemic_crisis)


#Choosing the number of trees
plot(fit_rf)

# Fitting to the training set and Using cross validation to choose the best parameter for the random forest classfication.
set.seed(46, sample.kind = "Rounding")
fit_rf_cv<-train(systemic_crisis~., 
                 method = 'rf', data= new_train_set,
                 tuneGrid = data.frame(mtry = seq(1, 7)), 
                 ntree = 100,
                 trControl=trainControl(method = "cv", number = 10, p = .9))

# Best mtry value that maximizes accuracy
plot(fit_rf_cv)
fit_rf_cv$bestTune

#Checking the prediction accuracy
pred_rf_cv<- predict(fit_rf_cv,test_set)
confusionMatrix(pred_rf_cv,test_set$systemic_crisis)

# Feature Importance
gini = as.data.frame(importance(fit_rf))
gini = data.frame(Feature = row.names(gini), 
                  MeanGini = round(gini[ ,'MeanDecreaseGini'], 2))
gini = gini[order(-gini[,"MeanGini"]),]

ggplot(gini,aes(reorder(Feature,MeanGini), MeanGini, group=1)) + 
  geom_point(color='red',shape=17,size=2) + 
  geom_line(color='blue',size=1) +
  scale_y_continuous(breaks=seq(0,500,50)) +
  xlab("Feature") + 
  ggtitle("Mean Gini Index of Features") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


