rm(list=ls())

library('tidyverse')
library(caret)

insur = read_csv('aug_train.csv')

insur$Region_Code = NULL
insur$Policy_Sales_Channel = NULL
insur$Driving_License = NULL

insur = insur %>%
  mutate(Gender = if_else(Gender=='Male', 1, 0),
         Vehicle_Age_1Year = if_else(Vehicle_Age=='< 1 Year', 1, 0),
         Vehicle_Age_1to2Year = if_else(Vehicle_Age=='1-2 Year', 1, 0),
         Vehicle_Age_2Year = if_else(Vehicle_Age=='> 2 Years', 1, 0),
         Vehicle_Damage = if_else(Vehicle_Damage=='Yes', 1, 0),
         Response = factor(Response, levels = c(0, 1)))

insur_down = downSample(select(insur, -one_of(insur$Response)), insur$Response)
insur_down = insur_down %>%
  mutate(id=1:nrow(insur_down))
  
set.seed(30)

train = insur_down %>%
  sample_frac(0.7)

val = insur_down %>%
  slice(-train$id)
  
  
mean_age = mean(train$Age)
sd_age = sd(train$Age)
mean_annual_premium = mean(train$Annual_Premium)
sd_annual_premium = sd(train$Annual_Premium)
mean_vintage = mean(train$Vintage)
sd_vintage = sd(train$Vintage)

train = train %>%
  mutate(Age_norm = (Age - mean_age) / sd_age,
         Annual_Premium_norm = (Annual_Premium - mean_annual_premium) / sd_annual_premium,
         Vintage_norm = (Vintage - mean_vintage) / sd_vintage)

val = val %>%
  mutate(Age_norm = (Age - mean_age) / sd_age,
         Annual_Premium_norm = (Annual_Premium - mean_annual_premium) / sd_annual_premium,
         Vintage_norm = (Vintage - mean_vintage) / sd_vintage)

train_input = train %>%
  select(Gender, Age_norm, Previously_Insured, Vehicle_Damage,
         Vehicle_Age_1Year, Vehicle_Age_1to2Year, Annual_Premium_norm, Vintage_norm)

val_input = val %>%
  select(Gender, Age_norm, Previously_Insured, Vehicle_Damage,
         Vehicle_Age_1Year, Vehicle_Age_1to2Year, Annual_Premium_norm, Vintage_norm)

train_actual = train$Response

val_actual = val$Response


## KNN
library(FNN)

prediction = knn(train_input, val_input, train_actual, 3)

val = val %>%
  mutate(prediction_knn = prediction)

library(caret)

cm = confusionMatrix(val$prediction_knn, val$Response)
cm
options(warn = -1)

for (k in 1:20)
{
  prediction = knn(train_input, val_input, train_actual, k)
  
  cm = confusionMatrix(prediction, val$Response)   # save confusion matrix output in object "cm"
  
  print(paste("k = ", k, ' Accuracy:', round(cm$overall[1], 4), 'Sensitivity: ', round(cm$byClass[1], 4), 'Specificity:', round(cm$byClass[2], 4))) # print the value of accuracy for each k
  
}
# k = 12
prediction = knn(train_input, val_input, train_actual, 12)

val = val %>%
  mutate(prediction_knn = prediction)

cm = confusionMatrix(val$prediction_knn, val$Response)
cm

## Logistic Regression

train.lr = glm(Response~Gender+Age_norm+Previously_Insured+Vehicle_Damage+
               Vehicle_Age_1Year+Vehicle_Age_1to2Year+Annual_Premium_norm+Vintage_norm, 
               train, 
               family = "binomial")
summary(train.lr)
prediction = predict(train.lr, val_input, type='response')

val = val %>%
  mutate(prediction_lr = prediction)
val = val %>%
  mutate(prediction_lr = if_else(prediction_lr >= 0.5, 1, 0))
val = val %>% 
  mutate(prediction_lr = factor(prediction_lr, levels = c(0, 1)))

confusionMatrix(val$prediction_lr, val$Response)

## decision tree
library(rpart)
library(rpart.plot)


insur.dt = rpart(Response~Gender+Age+Previously_Insured+
                Vehicle_Age+Vehicle_Damage+Annual_Premium+Vintage,
                data=train, method='class', cp=0.00001, minsplit=10, xval=10)
printcp(insur.dt)
plotcp(insur.dt)
# nsplit = 13
#prp(insur.dt, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

cp_table = as_tibble(insur.dt$cptable)
cp_table = cp_table%>%
  filter(nsplit == 13)

pruned.dt = prune(insur.dt, cp=cp_table$CP)
prp(pruned.dt, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)

prediction = predict(pruned.dt, val, type = "class")

val = val %>%
  mutate(prediction_t = prediction)

confusionMatrix(val$prediction_t, val$Response)





