# Derrick Owusu Ofori
# s4113984

#################################################################################################
# 0. Installing and Loading Libraries
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
##  Install Package if required
if (!require(tidyverse)) install.packages('tidyverse')#Data manipulation and visuals
if (!require(factoextra)) install.packages('factoextra') # Extract and Visualize the Results of Multivariate Data Analyses
if (!require(usdm)) install.packages('usdm') # Used for variance inflation factor, multicollinearity.
if (!require(lubridate)) install.packages('lubridate') # dates and times.
if (!require(caret)) install.packages('caret') # Streamlines the process for creating predictive models
if (!require(randomForest)) install.packages('randomForest') # Random forest algorithm
if (!require(ggpubr)) install.packages('ggpubr') # Functions for creating and customizing 'ggplot2
if (!require(glmnet)) install.packages('glmnet') # Ridge, lasso and elastic net regularized generalized models
if (!require(Metrics)) install.packages('Metrics') # Evaluation metrics for supervised machine learning
if (!require(GGally)) install.packages('GGally') # Plotting system for R based on the grammar of graphics
if (!require(reshape2)) install.packages('reshape2') # makes it easy to transform data between wide and long formats
if (!require(knitr)) install.packages('knitr') # Provides a general-purpose tool for dynamic report generation in R 
if (!require(viridis)) install.packages('viridis') # Generating the color maps in base R 
if (!require(hrbrthemes)) install.packages('hrbrthemes') # A compilation of extra 'ggplot2' themes, scales and utilities
if (!require(car)) install.packages('car') # Provides numerous functions that perform tests, creates visualizations, and transform data
if (!require(gbm)) install.packages('gbm') # Implementation of Gradient Boosting Machine algorithm
if (!require(rpart.plot)) install.packages('rpart.plot')# Plot Decision Tree
if (!require(rpart)) install.packages('rpart')# Prediction: Decision Tree


## Load libraries
library(caret)
library(tidyverse)
library(lubridate)
library(factoextra)
library(glmnet)
library(ggpubr)
library(randomForest)
library(Metrics)
library(ggpubr)
library(GGally)
library(reshape2)
library(knitr)
library(viridis)
library(hrbrthemes)
library(car)
library(gbm)
library(rpart.plot)

# 1. Introduction
# 2. Hypothesis


#################################################################################################
# 3 Data importation and preprocessing 
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#############################################################
# 3.1 Data Importation, Integration and feature enginnering
#############################################################

# Set working directory to Dataset - Assignment folder
setwd("./Dataset - CPS")
# List data set csv names of files
fnames <- list.files(pattern = '.csv$', # $ mean .csv appears at end of string
                     recursive = T)

csv <- lapply(fnames, read.csv) # Apply read.csv to all paths
dat<- as.data.frame(csv[1]) # select first data set 
dat1 <- dat %>% mutate_all(as.character) # Mutate all columns to character 
crime_data <- dat1[0,] # Empty data frame with same columns as data set


for (i in 1:length(fnames)){
  filname <- substr(fnames[i],6,nchar(fnames[1])) # Extract csv file name
  yearname <- substr(fnames[i],1,4) # Extract folder name as year
  splitname <- str_split(filname,"_") # Split file name by "_"
  monthname <- as.data.frame(splitname)[4,] # convert to data frame and select month(4th row)
  date <- paste(monthname,yearname) # combine month and year
  datfram <- as.data.frame(csv[i]) # convert csv from list to data frame
  datfram$year <- rep(yearname,nrow(datfram)) # create year column from replication of year name
  datfram$date <- rep(date,nrow(datfram)) # create month column from replication of month name
  datfram$date <- my(datfram$date) # convert date into month year format with my form lubridate
  datfram$month <- month.abb[month(as.POSIXlt(datfram$date, format="Y-%m-%d"))] # Create column for month abbreviation
  datfram1 <- datfram %>% mutate_all(as.character) # convert all data variables to character for easy integration
  crime_data <- bind_rows(crime_data,
                          datfram1) # bind datafrmaes into empty dataset
}

# Head of the year, month and date variable
head(crime_data %>%
       dplyr::select(year,month,date))

dim(crime_data) # Data dimensions

#############################################################
## 3.2 Data Cleaning
#############################################################

date_var <- crime_data$date # Save date varible in date_var
clean_percent_func<- function(x){gsub("%","",x)} # Function to remove %
clean_comma_func<- function(x){gsub(",","",x)} # Funciton to remove ,
clean_underscore_func<- function(x){gsub("-","0.0",x)} # Function to remove -
clean_functions <- c(clean_percent_func,clean_comma_func,clean_underscore_func) # Combine all functions

for (i in clean_functions){
  crime_data <- lapply(crime_data, i) # Apply all clean functions to data
  crime_data <- as.data.frame(crime_data) # Convert back to dataframe since lapply outputs list.
  
}

crime_data$date <- date_var # overwrite with date variable

## Rename columns
names(crime_data)[names(crime_data) == 'X'] <- 'County.Name'

##Variable class conversion
crime_data[2:52] <- sapply(crime_data[2:52],as.numeric) # convert elements to numeric
crime_data$date <- as.Date(crime_data$date) # convert date variable to date format

crime_data[1:40,1:4] # show the first 40 rows and 4 columns show the clean data
sum(is.na(crime_data)) # Sum of all missing data


#################################################################################################
# 4 Descriptive analysis
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

# Create sub dataframes for later analysis
dat1 <- crime_data %>% 
  filter(County.Name != "National" ) # Remove national row as it is derived from all the other counties

percentage_conv <- dat1 %>%
dplyr::select(starts_with("Percentage")) %>%
  dplyr::select(contains("Convictions")) # Data frame for all percentage conviction for all crimes

number_df <- dat1 %>%
  dplyr::select(starts_with("Number")) # Data frame for all number of convictions for all crimes

number_conv <- dat1 %>%
  dplyr::select(starts_with("Number")) %>%
  dplyr::select(contains("Convictions")) # Data frame for all number of convictions for all crimes

df_summarised <- dat1 %>% 
  dplyr::select(year,month,
         date,County.Name) # Put the year, month, date and County name in a data fame

df_summarised$sum.all.num.convs <- apply(number_conv,1,sum) # create a variable for the sum of all the convictions
df_summarised$average.all.perct.convs <- apply(percentage_conv,1,mean) # create a variable for the mean of all percentage convictions 

#############################################################
## 4.1 Summary statistics
#############################################################
SD_df <- as.data.frame(lapply(number_df, sd)) %>% 
  gather(key = "Offense", value = "SD") %>%
  mutate(SD=round(SD,digits = 2))

mean_df <- as.data.frame(lapply(number_df, mean)) %>% 
  gather(key = "Offense", value = "mean") %>%
  mutate(mean=round(mean,digits = 2))

max_df <- as.data.frame(lapply(number_df, max)) %>% 
  gather(key = "Offense", value = "max") %>%
  mutate(max=round(max,digits = 2))

min_df <- as.data.frame(lapply(number_df, min)) %>% 
  gather(key = "Offense", value = "min")%>%
  mutate(min=round(min,digits = 2))

#put all data frames into list
df_list <- list(mean_df,SD_df, max_df, min_df)

#merge all data frames in list
df_list %>% 
  reduce(full_join, by='Offense') %>%
  kable()


#############################################################
## 4.2 Data visualization
#############################################################

### 4.2.1 Average percentage unsuccessful Offence (Boxplot & Circular Barplot)
# Boxplot
# Select Percentage unsuccessful columns except percentage of.L moto.offence.unsucessful
per_unsucc <- dat1 %>%
  dplyr::select(starts_with("Percentage")) %>%
  dplyr::select(contains("Unsuccessful"),
         -Percentage.of.L.Motoring.Offences.Unsuccessful)

per_unsucc_gathered <- per_unsucc %>%
  gather(key = "type",value="values") # gather variable names into types and values into values

per_unsucc_gathered  %>%
  group_by(type) %>% # group by type; variable names
  summarise(percentage = mean(values)) %>% # mean percentage per type
  arrange(desc(percentage)) %>% # arrange in decending order
  top_n(5) %>% # show top 5
  ggplot(aes(reorder(type,+percentage),percentage)) + # reorder type by percentage
  geom_col(aes(fill=type)) + # fill with type
  theme_classic()+ # set classic theme
  theme(legend.position="none") + # no legend
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + # Flip x asis labels 
  labs(x = 'Unsuccessful Offence Category', y = 'Percentage',
       title = 'Top five unsuccessful Offence categories (Percentage)') # title and axix labels

# Circular Barplot
per_unsucc_cir <- per_unsucc_gathered  %>%
  group_by(type) %>%
  summarise(percentage = mean(values)) 

per_unsucc_cir <- per_unsucc_gathered  %>%
  group_by(type) %>% # group by type; variable names
  summarise(percentage = mean(values)) # mean percentage per type

id <- 1:12 # set ids representing total number of unsucessful convictions
per_unsucc_cir2 <- cbind(id,per_unsucc_cir) # Bind id and percentage unsucessful


# ----- This section prepares a data frame for labels ---- #
# Get the name and the position of each label
label_data <- per_unsucc_cir2

# calculate the ANGLE of the labels
number_of_bar <- nrow(label_data)
angle <-  90 - 360 * (label_data$id-0.5) /number_of_bar # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)

# calculate the alignment of labels: right or left
# If I am on the left part of the plot, my labels have currently an angle < -90
label_data$hjust<-ifelse( angle < -90, 1, 0)

# flip angle BY to make them readable
label_data$angle<-ifelse(angle < -90, angle+180, angle)

# Start the plot
plot.gg <- ggplot(per_unsucc_cir2, aes(x=as.factor(id), y=percentage+50)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  # This add the bars with a blue color
  geom_bar(stat="identity", fill=alpha("skyblue", 0.7)) +
  
  # Limits of the plot = very important. The negative value controls the size of the inner circle, the positive one is useful to add size to each bar
  ylim(-100,120) +
  
  # Custom the theme: no axis title and no cartesian grid
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1,4), "cm")      # Adjust the margin to make in sort labels are not truncated!
  ) +
  
  # This makes the coordinate polar instead of cartesian.
  coord_polar(start = 0) +
  
  # Add the labels, using the label_data dataframe that I have created before
  geom_text(data=label_data, aes(x=id, y=percentage+10, label=type, hjust=hjust), 
            color="blue", fontface="bold",alpha=0.8,
            size=2.2, angle= label_data$angle, 
            inherit.aes = FALSE ) 

plot.gg


### 4.2.2 Distribution and trend of yearly sexual offence convictions. (Boxplot, Line graph & Line graph with Vline)
dat1  %>%
  dplyr::select(year,Number.of.Sexual.Offences.Convictions) %>%
  ggboxplot(x = "year", y = "Number.of.Sexual.Offences.Convictions", 
            color = "year",ylab = "Number of sexual offense convictions", xlab = "Year")

# time series of sexual convictions overtime
dat1 %>%
  group_by(date) %>%
  summarise(Total.sex.conv = sum(Number.of.Sexual.Offences.Convictions)) %>%
  ggplot(aes(x=date,y=Total.sex.conv)) +
  theme_classic() +
  geom_line(col="skyblue", size = 1) +
  labs(x = 'Total sexual offence convictions', y = 'date',
       title = 'Trend of sexual offense convictions',
       subtitle = "2014 - 2017")

dat1 %>%
  group_by(date) %>%
  summarise(Total.sex.conv = sum(Number.of.Sexual.Offences.Convictions)) %>%
  ggplot(aes(x=date,y=Total.sex.conv)) +
  theme_classic() +
  geom_line(col="skyblue", size = 1) +
  labs(x = 'Total sexual offence convictions', y = 'date',
       title = 'Trend of sexual offense convictions',
       subtitle = "2014 - 2017") +
  geom_vline(xintercept = as.Date(("2017-10-01"))) + # vertical line
  geom_text(
    label= "#Metoo, Oct 2017", 
    x = as.Date(("2017-05-01")), # set the position of label on the x-axis
    y = 1180,  # set the position of label on the y axis
    check_overlap = T
  )


### 4.2.3 Monthly and yearly total conviction pattern (Lollipop chart, Grouped line graph & Stacked area chart)
# Lollipop
df_conv <- dat1 %>% 
  dplyr::select(year,month,
         date,County.Name)

df_conv$total.all.num.convs <- apply(number_conv,1,sum) # sum all convicitons

df_conv_month <- df_conv %>%
  group_by(month) %>%
  summarise(tot.conv = sum(total.all.num.convs)) # total conviction per month

df_conv_month %>%
  ggplot(aes(x=month, y=tot.conv)) +
  # xend is what month the lolipop plot ends on, y will also start form 0 and end at max for the month
  geom_segment( aes(reorder(x=month,+tot.conv), xend=month, y=0, yend=tot.conv), color="skyblue") +
  # alpha is for transparency
  geom_point( color="blue", size=4, alpha=0.6) +
  theme_light() + # set theme to light
  coord_flip() + # flip plot for better visualization
  theme(
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  labs(x = 'Month', y = 'Total convictions',
       title = 'Monthly total convictions for all years',
       subtitle = "2014 - 2017")


# Grid Bar graphs
# function to plot 
viz <- function(i){ # function to plot based on year inputed
  df_conv %>%
    filter(year==i) %>%
    group_by(year,date) %>%
    summarise(tot.conv = sum(total.all.num.convs)) %>%
    ggplot(aes(x=date, y=tot.conv)) +
    theme_pubr()+
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + # Flip x asis labels 
    geom_line(color = "steelblue", size = 1) +
    geom_point(color="steelblue")
}


figure <- ggarrange(viz(2014), viz(2015), viz(2016) , viz(2017),
                    labels = c("2014", "2015","2016", "2017"),
                    ncol = 2, nrow = 2) # plot form 2014 to 2017 on a 2 by 2 figure
figure 

# Stacked Plot
df_conv_date <- dat1 %>%
  dplyr:: select(starts_with("Number"), date) %>%
  dplyr:: select(contains("Convictions"),date) %>%
  group_by(date) 

stack_line_df <- aggregate(. ~ date, df_conv_date, sum) # Row sum based on date groups

# Library
# Plot
stack_line_df %>%
  gather(key = "Offense_types",value="Convictions", -date) %>%
  ggplot(aes(x=date, y=Convictions, fill=Offense_types)) + 
  geom_area(alpha=0.5 , size=0.5, colour="white") +
  theme_classic()


#################################################################################################
# 5 Hypothesis testing
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#############################################################
## 5.1 An increase in the number of homicide and robbery convictions will result in an increase in the number of sexual offence convictions.
#############################################################
### 5.1.1 Linear relationship

number_sex <- number_conv

fit_lm <- lm(Number.of.Sexual.Offences.Convictions~.,
             data = number_sex) # lm 

plot(fit_lm,which = 1) #residuals vs fitted for linearity assumption

### 5.1.2 Multivariate Normality
plot(fit_lm, which = 2) # QQ plot of residuals normality

### 5.1.3. Homoscedasticity
plot(fit_lm, which = 3) # scale-location plot for homoscedasticity

### 5.1.4 Autocorrelation
durbinWatsonTest(fit_lm) # Autocorrelation

### 5.1.5 No multicollinearity assumption.
number_sex1 <- number_sex %>% dplyr::select(-Number.of.Sexual.Offences.Convictions)
vif.1 <- usdm::vif(number_sex1) # Variance inflation factor(VIF), multicolinearity
vif.1

### 5.1.6 Cube root transformation to meet linear assumptions
number_sex_trans <- number_sex^(1/3)

### 5.1.7 Multivariate normality assumption met
fit_lm_trans <- lm(Number.of.Sexual.Offences.Convictions~.,
                   data = number_sex_trans)


plot(fit_lm_trans, which=2) # data is more normally distributed

### 5.1.8  Homoscedasticity assumption met
plot(fit_lm_trans, which=3)  


### 5.1.9 Multicollinearity rectified
vif.2 <- usdm::vif(number_sex_trans %>% 
                     dplyr::select(-Number.of.Sexual.Offences.Convictions))
vif.2 

# select VIF less than 5
vif_var_lessthan_5 <- vif.2$Variables[which(vif.2$VIF < 5)]
number_sex_final <- number_sex_trans %>% dplyr::select(all_of(vif_var_lessthan_5))
number_sex_final<-cbind(number_sex_final,number_sex_trans %>% 
                          dplyr::select(Number.of.Sexual.Offences.Convictions))


fit_lm_final <- lm(Number.of.Sexual.Offences.Convictions~.,
                   data = number_sex_final)

summary(fit_lm_final)
#############################################################
##5.2 Ridge regression will result in a lower root mean squared error than linear regression in the prediction of sexual conviction.
#############################################################

### 5.2.1 Linear regression
set.seed(76) # Set seed for repeatability
y <- number_sex_final$Number.of.Sexual.Offences.Convictions
# Split into 70:30 ratio for train and test set
test_index <- caret::createDataPartition(y, times = 1, p = 0.3, list = FALSE)
test_set <- number_sex_final[test_index,]
train_set <- number_sex_final[-test_index,]


fit_lm2 <- lm(Number.of.Sexual.Offences.Convictions~.,
              data = train_set)
# R2 of 0.6298
summary(fit_lm2)

pred_lm <- predict(fit_lm2,newdata =  test_set) # predictions
actual <- test_set$Number.of.Sexual.Offences.Convictions # actuals
head(data.frame(actual,pred_lm)) # head of both
# 0.487 is close to 0. Good predictor
caret::RMSE(pred = pred_lm,obs =  actual)

# Plot
plot(pred_lm,                                # Draw plot using Base R
     actual,
     xlab = "Predicted Values",
     ylab = "Observed Values")

abline(a = 0,# Add straight line
       b = 1,
       col = "blue",
       lwd = 2)


### 5.2.2 Predicting with Ridge regression

set.seed(76)
y <- number_sex_trans$Number.of.Sexual.Offences.Convictions
test_index <- caret::createDataPartition(y, times = 1, p = 0.3, list = FALSE)
test_set <- number_sex_trans[test_index,]
train_set <- number_sex_trans[-test_index,]

xtrain <- train_set %>% dplyr::select(-Number.of.Sexual.Offences.Convictions)
ytrain <- train_set %>% dplyr::select(Number.of.Sexual.Offences.Convictions)
xtest <- test_set %>% dplyr::select(-Number.of.Sexual.Offences.Convictions)
ytest <- test_set %>% dplyr::select(Number.of.Sexual.Offences.Convictions)


lambda.array <- seq(from=0.001,to = 10, by = 0.01)

set.seed(76)
# Glmnet requires a matrix as input for both, X and y. So you need to define as.matrix() on all model inputs.
ridgeFit <- glmnet(as.matrix(xtrain),as.matrix(ytrain), alpha = 0,
                   lambda = lambda.array) # alpa = 0 means activate ridge regression

# Visualise the effect of lamba
plot(ridgeFit, xvar = "lambda", label = T)

# I will first use the meadian lambda
y_predicted <- predict(ridgeFit,s = median(lambda.array), 
                       newx = as.matrix(xtest))


actual <- ytest$Number.of.Sexual.Offences.Convictions

# sum of squares and sum of squares error
sst <- sum((actual - 
              mean(actual))^2)

sse <-  sum((y_predicted - actual)^2)

rsquare <- 1 - (sse/sst)
rsquare

# RMSE 0.5448699
Linear_RMSE = caret::RMSE(pred = y_predicted, obs=actual)
Linear_RMSE


### 5.2.3 Ridge regression with cross validation
set.seed(76)
cv_ridge <- cv.glmnet(as.matrix(xtrain), as.matrix(ytrain), 
                      alpha = 0, lambda = lambda.array)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

y_predicted_cv <- predict(ridgeFit,s = optimal_lambda ,
                          newx = as.matrix(xtest))

head(data.frame(actual = actual, pred_ridge = y_predicted_cv))

# sum of squares and sum of squares error
sse <-  sum((y_predicted_cv - actual)^2)
rsquare <- 1 - (sse/sst)

# 78% of variations in sexual are explained.
rsquare

# RMSE
# lower RMSE 0.39
Ridge_RMSE = caret::RMSE(pred = y_predicted_cv, obs=actual)
Ridge_RMSE

plot(y_predicted_cv, # Draw plot using Base R
     actual,
     xlab = "Predicted Values",
     ylab = "Observed Values")

abline(a = 0, # Add straight line
       b = 1,
       col = "red",
       lwd = 2)

### 5.2.4 Hypothesis testing results
data.frame(Model =c('Ridge Regression','Linear Regression'), 
           RMSE = c(Ridge_RMSE,Linear_RMSE))


#############################################################
## 5.3 Ridge regression will result in a lower root mean squared error than linear regression in the prediction of sexual conviction.
#############################################################

### 5.3.1 Metropolitan area and city mean conviction vs other counties (Barplot and Boxplot)

dat_county_convs <-  df_summarised %>%
  dplyr::select(County.Name,sum.all.num.convs)

dat_county_convs$County.Name <- ifelse(dat_county_convs$County.Name=="Metropolitan and City","Metropolitan and City","Other Regions")


# Visualization with bar plot.. it is limited use
dat_county_convs %>% 
  group_by(County.Name) %>%
  summarise(average.convictions = mean(sum.all.num.convs)) %>%
  ggplot(aes(x=County.Name,y=average.convictions)) + 
  theme(legend.position="none") +
  geom_col(aes(fill=County.Name))

ggboxplot(dat_county_convs, x = "County.Name", y = "sum.all.num.convs", 
          color = "County.Name",ylab = "Total Convictions", xlab = "Regions")


### 5.3.2 K-Means clustering


num_dat <- dat1 %>%
  dplyr::select(starts_with("Number"))

scaled_num_dat <- as.data.frame(scale(num_dat))

kmean_withinss <- function(k) { 
  cluster <- kmeans(scaled_num_dat, k)
  return (cluster$tot.withinss)
}

# Set maximum cluster 
max_k <-10

set.seed(76)
# Run algorithm over a range of k 
wss <- sapply(1:max_k, kmean_withinss)
# Create a data frame to plot the graph
elbow <-data.frame(1:max_k, wss)

# Plot the graph with ggplot
ggplot(elbow, aes(x = X1.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 20, by = 1))

set.seed(76)
# Optimal k =3
pc_cluster_3 <-kmeans(scaled_num_dat, 3)
fviz_cluster(pc_cluster_3, data = scaled_num_dat,
             geom = 'point', ellipse.type = 'convex',
             ggtheme = theme_classic())

# Find out where cluster belongs
df_summarised$cluster <- as.factor(pc_cluster_3$cluster)

# Cluster 3 has higher total convictions
df_summarised %>% 
  group_by(cluster) %>%
  ggplot(aes(cluster,sum.all.num.convs)) +
  geom_boxplot(aes(col=cluster))

# Mean total conviction per cluster 1
cluster1.mean <- df_summarised %>% 
  filter(cluster==1) %>%
  group_by(cluster) %>%
  summarise(
    average = mean(sum.all.num.convs))

cluster1.mean

# Metropolitan and City mean same as cluster 1
metro.mean <- df_summarised %>% 
  filter(County.Name == 'Metropolitan and City') %>%
  group_by(County.Name) %>%
  summarise(
    average = mean(sum.all.num.convs))

metro.mean 

# Create cluster as conviction level
dat1$conviction.level <-  as.factor(pc_cluster_3$cluster)


### 5.3.3 Wilcoxon signed-rank test and hypothesis testing results
# Wilcox test.. for difference
Wilcox <- wilcox.test(sum.all.num.convs ~ County.Name, 
                      data = dat_county_convs,
                      alternative = "greater") # greater to test if the mean for metro is greater
Wilcox

#############################################################
## 5.4 Random forest algorithm will result in better accuracy than gradient boosting machine and decision trees algorithm in the prediction of conviction level.
#############################################################

### 5.4.1 Data partitioning
#Data Partitioning Train: 75%, Test: 25%

df_modelling <- dat1 %>%
  dplyr::select(starts_with("Number"),conviction.level)

set.seed(76)
y <- df_modelling$conviction.level
test_index <- caret::createDataPartition(y, times = 1, p = 0.25, list = FALSE)
test_set <- df_modelling[test_index,]
train_set <- df_modelling[-test_index,]

### 5.4.2 Results with no hyperparameter tuning and cross-validation
###############################
## Decision Tree
###############################
#Fitting decision tree model to the training set 
set.seed(76)
fit_rpart<-rpart(conviction.level~., data = train_set,
                 method = 'class')

#Predicting the test set results with the decision tree model
pred_rpart<- predict(fit_rpart,test_set,type = 'class')
cm_rpart <- caret::confusionMatrix(pred_rpart,test_set$conviction.level)

###############################
## Random Forest
###############################
set.seed(76)
fit_rf<-randomForest(conviction.level~., data = train_set,
                     method = 'class')

#Predicting the test set results with the decision tree model
pred_rf<- predict(fit_rf,test_set)
cm_rf <- caret::confusionMatrix(pred_rf,test_set$conviction.level)

###############################
## GLM
###############################
set.seed(76)
fit_gbm<- gbm(conviction.level~., 
              data = train_set,
              distribution = "multinomial")


#Checking the prediction accuracy
pred_gbm <- predict.gbm(object = fit_gbm,
                        newdata = test_set,
                        type = "response")

labels = colnames(pred_gbm)[apply(pred_gbm, 1, which.max)]
result = data.frame(test_set$conviction.level, labels)

cm_gbm <- caret::confusionMatrix(test_set$conviction.level, as.factor(labels))


# Results
data.frame(Model=c("Gradient boosting machine","Random Forest","Decision Tree"),
           Accuracy=c(cm_gbm$overall['Accuracy'],
                      cm_rf$overall['Accuracy'],
                      cm_rpart$overall['Accuracy'])) %>% kable()


### 5.4.3 Decision tree with hyperparameter tunning of complexity parameter

###############################
##1 Decision Tree
###############################

# Using cross validation to choose cp
set.seed(76)
fit_rpart_cv<- caret::train(conviction.level~., data = train_set,
                            method = 'rpart', # method rpart
                            tuneGrid = data.frame(cp = seq(0, 0.1, len = 40)), # complexity parameter
                            trControl=caret::trainControl(method = "repeatedcv", repeats = 3, number = 10, p = .9)) #cv

# Best tune
plot(fit_rpart_cv)
fit_rpart_cv$bestTune

# Tree Visualization
rpart.plot(fit_rpart_cv$finalModel, extra = 4)

pred_rpart_cv<- predict(fit_rpart_cv,test_set)
cm_rpart_cv <- caret::confusionMatrix(pred_rpart_cv,test_set$conviction.level)


### 5.4.4 Random forest with hyperparameter tuning and cross-validation

plot(fit_rf)

set.seed(76)
fit_rf_cv<-caret::train(conviction.level~., 
                        method = 'rf', data= train_set,
                        tuneGrid = data.frame(mtry = seq(1,7)), 
                        ntree = 70,
                        trControl=trainControl(method = "repeatedcv", number = 10,repeats = 3, p = .9))

# Best mtry value that maximizes accuracy

plot(fit_rf_cv)
fit_rf_cv$bestTune

#Checking the prediction accuracy
pred_rf_cv<- predict(fit_rf_cv,test_set)
cm_rf_cv <- caret::confusionMatrix(pred_rf_cv,test_set$conviction.level)


### 5.4.5 GBM Tree with Preprocessing and repeated cross-validation

########################
# 3 GBM Tree
#######################
set.seed(76)
fit_gbm_cv <- caret::train(conviction.level~.,
                           data = train_set,
                           method = "gbm",
                           preProcess = c("scale", "center"),
                           trControl = trainControl(method = "repeatedcv", 
                                                    number = 10, 
                                                    repeats = 3, 
                                                    verboseIter = FALSE), # dont show traning logs for the train contrl
                           verbose = 0) # Dont show traing logs for train 

#Checking the prediction accuracy
pred_gbm_cv<- predict(fit_gbm_cv,test_set)
cm_gbm_cv <- caret::confusionMatrix(pred_gbm_cv,test_set$conviction.level)

### 5.2.6 Model evaluation and hypothesis testing results
data.frame(Model=c("Gradient boosting machine","Random Forest","Decision Tree"),
           Accuracy=c(cm_gbm_cv$overall['Accuracy'],
                      cm_rf_cv$overall['Accuracy'],
                      cm_rpart_cv$overall['Accuracy'])) %>% kable()

cm_gbm_cv

