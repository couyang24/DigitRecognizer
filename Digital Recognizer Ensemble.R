library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(vtreat)
library(magrittr)

setwd("C:/Users/Owen/Dropbox/Analysis/Database/Digit Recognizer")
training1<-read_csv("train.csv")
inTrain<-createDataPartition(training1$label,p=.7,list=F)
training<-training1[inTrain,]
validing<-training1[-inTrain,]

vlabel<-validing[1]
validing<-validing[-1]

testing<-read_csv("test.csv")


#xgboost

vars<-colnames(training[-1])

treatplan<-designTreatmentsZ(training,vars,verbose = F)

new_vars<-treatplan%>%
  use_series(scoreFrame)%>%
  filter(code %in%  c('lev','clean'))%>%
  use_series(varName)

training.treat<-prepare(treatplan,training,varRestriction = new_vars)
validing.treat<-prepare(treatplan,validing,varRestriction = new_vars)
testing.treat<-prepare(treatplan,testing,varRestriction = new_vars)

cv<-xgb.cv(
  data = as.matrix(training.treat),
  label=training$label,
  nrounds = 100,
  nfold = 5,
  objective = "multi:softmax",
  num_class = 10,
  eta = .3,
  max_depth = 6,
  early_stopping_rounds = 10,
  verbose = 0    # silent
)

(elog <- cv$evaluation_log)

# Determine and print how many trees minimize training and test error
elog %>% 
  summarize(ntrees.mean = which.min(test_merror_mean),   # find the index of min(train_rmse_mean)
            ntrees.std  = which.min(test_merror_std))   # find the index of min(test_rmse_mean)


ntrees<-99

xgboostmodel<-xgboost(
  data=as.matrix(training.treat),
  label=training$label,
  nrounds=ntrees,
  objective = "multi:softmax",
  num_class = 10,
  eta=.3,
  max_depth=6,
  early_stopping_rounds = 10,
  verbose=2
)

vlabel$xgboostpred<-predict(xgboostmodel,newdata=as.matrix(validing.treat))

confusionMatrix(vlabel$label,vlabel$xgboostpred)


#randomforest

model<-randomForest(label~.,training)



