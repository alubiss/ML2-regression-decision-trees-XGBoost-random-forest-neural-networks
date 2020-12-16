

# Celem pracy jest porównanie dokładności oszacowań modeli o różnych sposobach predykcji.
# Dane pochodzą z ankiety przeprowadzonej na portalach społecznościowych, głównie na grupach studenckich (Studenci UW, Studenci Politechniki Warszawskiej itp.). Badanie miało charakter przekrojowy, a grupą docelową były osoby w wieku 20-30 lat. Łącznie zgromadzono 430 obserwacji. Zmienną objaśnianą jest planowany wiek rodzenia pierwszego dziecka, natomiast zmiennymi onjaśniającymi:
#   - płeć respondenta- 0-kobieta, 1-mężczyzna, zmienna binarna,
# - wiek respondenta - zmienna ciągła,
# - wielkość zamieszkiwanej przez respondenta miejscowości oraz wielkość miejsowości urodzenia respondenta-
#   1	poniżej 10 tysięcy	
# 2	- małe miasto - 10-50 tys.	
# 3	- średnie miasto - 50-150 tys.	
# 4	- duże miasto - 150-500 tys.	
# 5	- bardzo duże miasto - 500-1000 tys.	
# 6	- wielkie miasto - pow. miliona	
# - poziom wykształcenia respondenta-
#   1	jestem uczennicą/uczniem szkoły średniej
# 2	ukończone średnie i nie kontynuuję nauki
# 3	jestem studentką/em
# 0	wyższe pełne
# - typ piekunku studiów -
#   1	Kierunki ekonomiczne
# 2	Kierunki medyczne
# 3	Kierunki humanistyczne
# 4	Kierunki społeczne
# 5	Kierunki prawa i administracji
# 6	Kierunki biologiczne i przyrodnicze
# 7	Kierunki ścisłe
# 8	Kierunki wychowania fizycznego
# - ocena swojego stanu zdrowia przez respondenta w skali od 0-5,
# - miejsce na którym respondent podaje "urodzenie dziecka" wśród innych priorytetów życiowych (zawarcie związku małżeńskiego, urodzenie pierwszego dziecka, ukończenie studiów, osiągnięcie satysfakcjonującej sytuacji zawodowej, realizacja innych planów nie związanych z karierą (podróże, sport,…), zdobycie pierwszej pracy związanej z ukończonym kierunkiem studiów, posiadanie własnego mieszkania),
# - liczbę dzieci jaką chciałby mieć respondent.
# Z pośród modeli zostanie wybrany ten o najdokładniejszej jakości predyckcji i najmniejszej wartości błędów.
# Wykorzystano następujące modele i metody :
# - walidacja krzyżowa modelu,
# - drzewo decyzyjne,
# - bagging,
# - sub-bagging,
# - random forest,
# - XGBoost,
# - neural networks.

# Potrzebne pakiety

library(MASS)
library(tidyverse)
library(tree)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(dplyr)
library(randomForest)
library(ranger)
library(gbm)
library(xgboost)
library(fastAdaboost)
library(adaStump)
#library(catboost)
library(neuralnet)
library(caret)
library(plyr) 
library(boot)

# Wczytanie danych

library(readxl)
setwd("/Users/alubis/Desktop/ML2/ML2projekt/")
model1 <- read_excel("~/Desktop/ML2/ML2projekt/model1.xlsx")
dane=model1

str(dane)
as_tibble(dane)

# brak braków danych

# Podział zbioru na część treningową i testową
set.seed(123456)

train_obs <- createDataPartition(dane$wiek_rodzenia_dziecka, 
                                 p = 0.7, 
                                 list = FALSE) 
dane_train <- dane[train_obs,]
dane_test  <- dane[-train_obs,]

# all predictors
modelformula <- wiek_rodzenia_dziecka ~ .


# MODEL I

# tree_basic
set.seed(123456)
dane.tree <-  rpart(modelformula,
                    data = dane_train)

fancyRpartPlot(dane.tree)


# WALIDACJA KRZYŻOWA MODELU I

tc <- trainControl(method = "cv",
                   number = 10)
cp.grid <- expand.grid(cp = seq(0, 0.03, 0.001))

set.seed(123456)
dane.tree.cv <- train(modelformula,
                      data = dane_train, 
                      method = "rpart", 
                      trControl = tc,
                      tuneGrid = cp.grid)
dane.tree.cv

set.seed(123456)
cp.grid <- expand.grid(cp = 0.024)

dane.tree.cv <- train(modelformula,
                      data = dane_train, 
                      method = "rpart", 
                      trControl = tc,
                      tuneGrid = cp.grid)
dane.tree.cv


# Porównanie błędów dla części treningowej i testowej zbioru

source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# train dataset
a= regressionMetrics(real = dane_train$wiek_rodzenia_dziecka,
                     predicted = predict(dane.tree.cv, dane_train))

# test dataset
b= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = predict(dane.tree.cv, dane_test))

table=rbind(a,b)
table=round(table,3)
table = table %>% mutate(model=c("dane_train", "dane_test"))
table

# Uzyskano wyższą wartość błędów dla próby testowej.

# W celu zmniejszenia wariancji prognozowania zastosowano bagging
# Powtórzono proces stosowania tego samego modelu dla różnych podpróbek danych 
# (przy użyciu losowego próbkowania). 
# Prognoza ostatecznej wersji modelu została stworzona przez połączenie wyników wszystkich pojedynczych prognoz.

# MODEL II

# bagging

n <- nrow(dane_train)
# we create an empty list to collect results 
results_regression <- list()
dane_sample <- list()

set.seed(123456)
for (sample in 1:1000) {
  message(sample)
  # we draw n-element sample (with replacement) 
  data_sample <- 
    dane_train[sample(1:n, 
                      size = n,
                      replace = TRUE),]
  # paste as the next element of the list 
  results_regression[[sample]] <-lm(modelformula,
                                    data_sample)
  dane_sample[[sample]] <- data_sample
}
# wynikiem jest 1000 przeprowadzonych regresji na tych samych danych ale różnych podpróbkach

# now we make predictions for all models

forecasts_bag <- 
  sapply(results_regression,
         function(x) 
           predict(object = x,
                   newdata = dane_test,
                   type = "response")) %>% 
  data.frame()


forecasts_bag = forecasts_bag %>% rowMeans()

source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
c= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = forecasts_bag)

b=round(b,3)
c=round(c,3)
b = b %>% mutate(model= "walidacja_krzyżowa")
c = c %>% mutate(model= "bagging")

table = rbind(b,c)
table %>% arrange(MSE)

# WNIOSKI:
# ZASTOSOWANIE BAGGINGU OBNIŻYŁO WARTOŚCI BŁĘDÓW OSZACOWAŃ

# MODEL III
# sub-bagging

results_regression2 <- list()
dane_sample2 <- list()

set.seed(123456)
for (sample in 1:1000) {
  message(sample)
  # we draw n-element sample (with replacement) 
  data_sample2 <- 
    dane_train[sample(1:n,  
                      # important difference!
                      size = n/2,
                      replace = FALSE),]
  # paste as the next element of the list 
  results_regression2[[sample]] <-lm(modelformula,
                                     data_sample2)
  dane_sample2[[sample]] <- data_sample
}

# now we make predictions for all models

forecasts_bag2 <- 
  sapply(results_regression2,
         function(x) 
           predict(object = x,
                   newdata = dane_test,
                   type = "response")) %>% 
  data.frame()


forecasts_bag2 = forecasts_bag2 %>% rowMeans()

source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
d= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = forecasts_bag2)
d=round(d,3)
d = d %>% mutate(model= "sub-bagging")

table = rbind(table,d)
table %>% arrange(MSE)

# WNIOSKI:
# ZASTOSOWANIE SUB-BAGGINGU OBNIŻYŁO WARTOŚCI BŁĘDÓW OSZACOWAŃ W PORÓWNANIU DO ZASTOSOWANIA WALIDACJI KRZYŻOWEJ, DAŁO BARDZO ZBLIŻONE REZULTATY DO BAGGINGU.

# MODEL IV
# random forest

set.seed(123456)
dane.random.forest <- randomForest(modelformula,
                                   data = dane_train)
print(dane.random.forest)
plot(dane.random.forest)


# limited number of trees
# model on bootstrap samples from the full data set

set.seed(123456)
dane.random.forest2 <- randomForest(modelformula,
                                    data = dane_train,
                                    ntree = 50,
                                    # however we cut down number of obs
                                    # in each tree to 100
                                    sampsize = 100,
                                    mtry = 6,
                                    # we also generate 
                                    # predictors importance measures,
                                    importance = TRUE)

plot(dane.random.forest2)

print(dane.random.forest2)

# cross validation

parameters_rf <- expand.grid(mtry = 1:10)

set.seed(123456)
dane.random.forest3_training <- 
  train(modelformula, 
        data = dane_train, 
        method = "rf", 
        ntree = 50,
        tuneGrid = parameters_rf, 
        trControl = tc,
        importance = TRUE)

plot(dane.random.forest3_training)

print(dane.random.forest3_training)


# optimal mtry = 3
# najniższa wartość błędu dla parametru 3

set.seed(123456)
dane.random.forest3 <- randomForest(modelformula,
                                    data = dane_train,
                                    ntree = 50,
                                    # however we cut down number of obs
                                    # in each tree to 100
                                    sampsize = 100,
                                    mtry = 3,
                                    # we also generate 
                                    # predictors importance measures,
                                    importance = TRUE)


# optimal mtry = 9
# najniższa wartość błędu dla parametru 9

set.seed(123456)
dane.random.forest4 <- randomForest(modelformula,
                                    data = dane_train,
                                    ntree = 50,
                                    # however we cut down number of obs
                                    # in each tree to 100
                                    sampsize = 100,
                                    mtry = 9,
                                    # we also generate 
                                    # predictors importance measures,
                                    importance = TRUE)


regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                  predicted = predict(dane.random.forest4, dane_test))


# Porównanie modeli:

# test dataset
e1= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.random.forest, dane_test))

e2= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.random.forest2, dane_test))

e3= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.random.forest3, dane_test))

e1=round(e1,3)
e2=round(e2,3)
e3=round(e3,3)
rf= rbind(e1,e2)
rf=rbind(rf,e3)
rf = rf %>% mutate(random_forest = c("basic", "bootstrap samples", "optimal mtry"))
rf %>% arrange(MSE)

# Największe wartości błędów otrzymano za pomocą podstawowej wersji modelu. 
# Po zastosowaniu walidacji krzyżowej wartości błędów spadły. 
# Najniższe wartości błędów przyjmuje model, w którym zastosowano optymalną wartość parametru mtry, 
# zatem tunning modelu poprawił jakość oszacowań.

source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
e= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = predict(dane.random.forest3, dane_test))

e=round(e,3)
e = e %>% mutate(model= "random forest")

table = rbind(table,e)
table %>% arrange(MSE)

# MODEL 5
# Boosting of decision trees

# XGBoost

#modelLookup("xgbTree")

set.seed(123456)

parameters_xgboost <- expand.grid(nrounds = seq(20, 80, 10),
                                  max_depth = c(5), # 5-8 w zależności od wielkości zbioru
                                  eta = c(0.25), 
                                  gamma = 1,
                                  colsample_bytree = c(0.3), # sqrt(p)/p p=10
                                  min_child_weight = c(21), # 5% obserwacji
                                  subsample = 0.8)

# cross validation
set.seed(123456)
dane.xgboost<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost)
dane.xgboost

# the best result has been obtained for:
#   nrounds = 40
# 
# The next step is to find optimal values of tree parameters:
# - max_depth
# - min_child_weight
# - colsample_bytree

parameters_xgboost2 <- expand.grid(nrounds = 40,
                                   max_depth = seq(3, 9, 1),
                                   eta = c(0.25), # low value so eta=0.25
                                   gamma = 1,
                                   colsample_bytree = c(0.3),
                                   min_child_weight = seq(15,50, 1),
                                   subsample = 0.8)

set.seed(123456)
dane.xgboost2<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost2)
dane.xgboost2

parameters_xgboost3 <- expand.grid(nrounds = 40, max_depth = 5, eta = 0.25, gamma = 1,
                                   colsample_bytree = 0.3, min_child_weight = 48, subsample = 0.8)

set.seed(123456)
dane.xgboost3<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost3)
dane.xgboost3

# colsample_bytree 

parameters_xgboost4 <- expand.grid(nrounds = 40, max_depth = 5, eta = 0.25, gamma = 1,
                                   colsample_bytree = seq(0.1, 0.8, 0.1), min_child_weight = 48, subsample = 0.8)

set.seed(123456)
dane.xgboost4<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost4)
dane.xgboost4

# The next step is to determine the optimal length of the subsample:

parameters_xgboost5 <- expand.grid(nrounds = 40, max_depth = 5, eta = 0.25, gamma = 1,
                                   colsample_bytree = 0.3, min_child_weight = 48, subsample = c(0.6,0.7,0.75,0.8,0.85,0.9))


set.seed(123456)
dane.xgboost5<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost5)
dane.xgboost5

parameters_xgboost6 <- expand.grid(nrounds = 40, max_depth = 5, eta = 0.25, gamma = 1,
                                   colsample_bytree = 0.3, min_child_weight = 48, subsample = 0.8)


set.seed(123456)
dane.xgboost6<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost6)
dane.xgboost6

# Now we will lower the learning rate 
# and proportionally increase number of trees.
# let us lower learning rate by half (up to 0.125)
# increase number of trees (up to 60)

parameters_xgboost7 <- expand.grid(nrounds = 60, 
                                   max_depth = 5, 
                                   eta = 0.125, 
                                   gamma = 1,
                                   colsample_bytree = 0.3, 
                                   min_child_weight = 48, 
                                   subsample = 0.8)


set.seed(123456)
dane.xgboost7<- 
  train(modelformula,
        data = dane_train,
        method = "xgbTree",
        trControl = tc,
        tuneGrid  = parameters_xgboost7)
dane.xgboost7


source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
f1= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.xgboost, dane_test))

f2= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.xgboost3, dane_test))

f3= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                      predicted = predict(dane.xgboost7, dane_test))


f1=round(f1,3)
f2=round(f2,3)
f3=round(f3,3)
xb= rbind(f1,f2)
xb=rbind(xb,f3)
xb = xb %>% mutate(xgboost = c("basic", "tunning", "final"))
xb %>% arrange(MSE)

# Podobnie jak w poprzednich modelach największe błędy wykazuje model podstawowy. Tunning i dopasowanie wielkości parametrów poprawia jakość oszacowań. Najlepsze wyniki osiągane są dla modelu, w którym dopasowano wielkości parametrów, zwiększono liczbę drzew i zmniejszono parametr uczenia.

source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
f= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = predict(dane.xgboost7, dane_test))
f=round(f,3)
f = f %>% mutate(model= "xgboost")

table = rbind(table,f)
table %>% arrange(MSE)

# MODEL VI
# neural networks

set.seed(123456)
# fitting the linear model, which will serve as the benchmark 
lm.fit <- glm(modelformula , data = dane_train)
summary(lm.fit)
# calculating predictions
lm.pred <- predict(lm.fit, dane_test)


source("/Users/alubis/Desktop/ML2/zaj1/regressionMetrics.R")

# test dataset
g= regressionMetrics(real = dane_test$wiek_rodzenia_dziecka,
                     predicted = lm.pred)
g=round(g,3)
g = g %>% mutate(model= "regression")

table = rbind(table,g)
table %>% arrange(MSE)


# training the neural network

# all the cointinuous variables must be first standardized to the common scale 
# determining max and min values in the training data (for each variable)
train.maxs <- apply(dane_train, 2, max)
train.mins <- apply(dane_train, 2, min)

# standardization
dane_train.scaled <- 
  as.data.frame(scale(dane_train, 
                      center = train.mins, 
                      scale  = train.maxs - train.mins))
dane_test.scaled <- 
  as.data.frame(scale(dane_test, 
                      center = train.mins, 
                      scale  = train.maxs - train.mins))

# variables in the testing dataset are standardized based only on the 
# information from the training set!

# defining formula of the model
variables <- names(dane_train)
modelformula2 <- as.formula(paste("wiek_rodzenia_dziecka ~", 
                                  paste(variables[!variables %in% "wiek_rodzenia_dziecka"], 
                                        collapse = " + ")))

# verification
modelformula2


set.seed(123456)
nn <- neuralnet(modelformula2, 
                data   = dane_train.scaled,
                hidden = c(1), 
                # T for regression, F for classification
                linear.output = T, 
                threshold = 0.01,
                learningrate.limit = NULL,
                learningrate.factor = list(minus = 0.5, plus = 1.2),
                algorithm = "rprop+")
plot(nn)


# to generate predictions we will use the compute() function 
nn.pred <- compute(nn, dane_test.scaled[, 1: 10 ])

nn.pred.unscaled <- 
  nn.pred$net.result * 
  (train.maxs["wiek_rodzenia_dziecka"] - train.mins["wiek_rodzenia_dziecka"]) + train.mins["wiek_rodzenia_dziecka"]

# calculating MSE for the testing set
nn.MSE <- sum((dane_test$wiek_rodzenia_dziecka - nn.pred.unscaled)^2)/nrow(dane_test)
nn.MSE


set.seed(123456)
# let us increase number of neurons 
nn2 <- neuralnet(modelformula2, 
                 data   = dane_train.scaled,
                 hidden = c(5), 
                 linear.output = T)
plot(nn2)


# generating predictions
nn2.pred <- compute(nn2, dane_test.scaled[, 1:10])

# transformation opposite to the standardization
nn2.pred.unscaled <- 
  nn2.pred$net.result * 
  (train.maxs["wiek_rodzenia_dziecka"] - train.mins["wiek_rodzenia_dziecka"]) + train.mins["wiek_rodzenia_dziecka"]

# calculating MSE for the testing set
nn2.MSE <- sum((dane_test$wiek_rodzenia_dziecka - nn2.pred.unscaled)^2)/nrow(dane_test)
nn2.MSE

set.seed(123456)
nn3 <- neuralnet(modelformula2, 
                 data   = dane_train.scaled,
                 hidden = c(12, 3), # neurony w warstwie ukrytej
                 linear.output = T, # T dla regresjii, F dla klasyfikacji
                 algorithm = "backprop",
                 learningrate = 0.0001,
                 threshold = 0.1,
                 stepmax = 1e+06,
                 rep = 1)
plot(nn3)

# compute() to generate predictions
nn3.pred <- compute(nn3, dane_test.scaled[, 1:10])

# scaling back
nn3.pred.unscaled <- 
  nn3.pred$net.result * 
  (train.maxs["wiek_rodzenia_dziecka"] - train.mins["wiek_rodzenia_dziecka"]) + train.mins["wiek_rodzenia_dziecka"]

# calculating MSE for the testing set
nn3.MSE <- sum((dane_test$wiek_rodzenia_dziecka - nn3.pred.unscaled)^2)/nrow(dane_test)
nn3.MSE


## Cross- validation nn

# cross-validation of the NN
# we prepare the vector to store prediction errors for of CV folds 
cv.nn.error <- NULL

# we use 10-fold CV
k <- 10

# next we prepare variables used in all folds 
set.seed(123456)
index.df <- 
  as.data.frame(split((1:200)[order(runif(200))], 1:10))

# progress bar on
pbar <- create_progress_bar('text')
pbar$init(k)

# and we repeat in the loop 10 folds of the CV process 
set.seed(123456)
for (i in 1:k) {
  dane_train.cv <- dane_train[-index.df[, i],]
  dane_test.cv  <- dane_train[ index.df[, i],]
  
  # scaling ------------------------------
  train.maxs <- apply(dane_train.cv, 2, max) 
  train.mins <- apply(dane_train.cv, 2, min)
  dane_train.cv.scaled <- 
    as.data.frame(scale(dane_train.cv, 
                        center = train.mins, 
                        scale  = train.maxs - train.mins))
  dane_test.cv.scaled <- 
    as.data.frame(scale(dane_test.cv, 
                        center = train.mins, 
                        scale  = train.maxs - train.mins))
  
  # training -------------------------
  set.seed(123456)
  nn <- neuralnet(modelformula2, 
                  data   = dane_train.scaled,
                  hidden = c(12, 3), # neurony w warstwie ukrytej
                  linear.output = T, # T dla regresjii, F dla klasyfikacji
                  algorithm = "backprop",
                  learningrate = 0.0001,
                  threshold = 0.1,
                  stepmax = 1e+06,
                  rep = 1)
  
  # generating predictions --------------------
  nn.pred <- compute(nn, dane_test.cv.scaled[, 1:10])
  
  # scaling back -----------------
  nn.pred.unscaled <- 
    nn.pred$net.result * 
    (train.maxs["wiek_rodzenia_dziecka"]-train.mins["wiek_rodzenia_dziecka"])+train.mins["wiek_rodzenia_dziecka"]
  
  # calculating the CV prediction error 
  (cv.nn.error[i] <- 
      sum((dane_test.cv$wiek_rodzenia_dziecka - nn.pred.unscaled)^2) / nrow(dane_test.cv))
  
  pbar$step()
}

# mean error
mean(cv.nn.error)

# Otrzymano wyższe wartości błędu w porówananiu do oszacowań poprzednich modeli.

# Porównanie z wcześniej omówionymi modelami
table %>% arrange(MSE)

# # PODSUMOWANIE
# Spośród przeprowadzonych metod, najlepiej sprawdził się bagging i sub-bagging. 
# Dla tych modeli otrzymaliśmy najmniejsze wartości błędów predykcji oraz najwyższy współczynnik R^2. 
# Najgorsze rezultaty widzimy dla modelu, w którym została przeprowadzona jedynie walidacja krzyżowa drzewa decyzyjnego.

