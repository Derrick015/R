##############################################################################################
# Installing packages and Importing data
##############################################################################################
#Install and libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(recosystem)


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



##############################################################################################
#Data exploration and visualization
##############################################################################################

# Average movie rating
edx %>% group_by(movieId) %>%
  summarise(avg_rating = sum(rating)/n()) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(bins=35, color ="black", fill="green") +
  labs(x = "Average Rating", y = "Movies")


# Number of ratings per movie
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=35, color = "black", fill="green") +
  scale_x_log10() + labs(x = "Movies", y = "Number of Ratings")


# Average rating per user
edx %>% group_by(userId) %>%
  summarise(avg_rating = sum(rating)/n()) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(bins=35, color = "black", fill="green") +
  labs(x = "Average rating", y = "Number of users") 

# Number of ratings per user
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=35, color = "black",fill="green") +
  scale_x_log10() +
  labs(x = "Users", y = "Number of ratings") 


##############################################################################################
# Modeling approach
##############################################################################################

#1. Average Movie Rating Model

# Calculate mean rating 
mu <- mean(edx$rating)
mu

# Compute RMSE based on average rating
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

#Produce and save results in dataframe
rmse_results <- data_frame(method = "Average Movie Rating",
                           RMSE = naive_rmse)



# 2. Movie Effect Model

# Compute movie effect
movie_avgs<- edx %>% 
  group_by(movieId) %>%
  summarise(b_i=mean(rating-mu))

# Plot number of movies with movie effect estimate
movie_avgs %>% qplot(b_i, geom ="histogram",
                     bins = 10, data = ., 
                     color = I("black"),
                     fill=I("green"))


# Predict rating including the movie effect model
predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# Calculate and save rmse accounting for the movie effect
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse ))



# 3. Movie and user effect model

#Compute user effect
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Plot number of movies with user effect estimate
user_avgs%>% qplot(b_u, geom ="histogram", 
                   bins = 30, data = ., 
                   color = I("black"),
                   fill=I("green"))

# Predict rating including the movie and user effect model
predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Calculate and save rmse accounting for the movie and user effect
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model",  
                                     RMSE = model_2_rmse))


# 4.Regularizing movie and user effect

# Generate lambdas from 0 to 10 with 0.25 increments
lambdas<- seq(0,10,0.25)

# Regularize models, predict ratings and compute rmse for all lambdas
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot rmse for each lambda                                       
qplot(lambdas, rmses)  

# Select optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

# Calculate and save rmse accounting for the regularized movie and user effect model
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",
                                     RMSE = min(rmses)))


#5. Matrix factorization

# Select three columns: user (userId), item (movieId) and the value (rating)
edx_fc <- edx %>% select(movieId, userId, rating)
validation_fc <- validation %>% select(movieId, userId, rating)

# Transform variables into matrix format.
edx_fc <- as.matrix(edx_fc)
validation_fc <- as.matrix(validation_fc)

# Write datasets onto the hard disk as tables 
write.table(edx_fc, file = "trainingset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)

write.table(validation_fc, file = "validationset.txt", sep = " ", 
            row.names = FALSE, col.names = FALSE)

# Assign the written datasets to a train set (train_fc) and a validation set(valid_fc). 
#A supported data format will be utilized by calling the data_file() function.           
set.seed(76)
train_fc <- data_file("trainingset.txt")

valid_fc <- data_file("validationset.txt")


# Build a recommender object (r)
r = Reco()

# We utilize the $tune() approach to find the optimum tuning parameter.
opts <- r$tune(train_fc, opts = list(dim = c(5, 10, 15), lrate = 
                                       c(0.1,0.2), costp_l1 = 0,
                                     costq_l1 = 0, nthread = 1, niter = 10))
opts

# The recommender model will now be trained with $train().
r$train(train_fc, opts = c(opts$min, nthread = 1, niter = 20))

# Write the predictions to a temp file on the hard drive
saved_pred = tempfile()

# Make predictions with the validation set
r$predict(valid_fc, out_file(saved_pred))
actual_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
predicted_ratings <- scan(saved_pred)

# Calculate the RMSE.
rmse_fc <- RMSE(actual_ratings, predicted_ratings)
rmse_fc 

# Save and tabulate the RMSE for comparison with the previous models
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix factorization", RMSE = rmse_fc ))

# Results
rmse_results %>% knitr::kable()
