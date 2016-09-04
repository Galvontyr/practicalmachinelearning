setwd("G:/Libraries/My Documents/Data Science/Course 8 - Practical Machine Learning/Final Project")

packages <- c("caret", "ggplot2", "reshape2", "plyr", "randomForest", 
              "gbm", "parallel", "doParallel")
lapply(packages, require, character.only=TRUE)

validation <- read.csv("pml-testing.csv")
train <- read.csv("pml-training.csv")

set.seed(2618)

inTrain <- createDataPartition(train$classe, p=.75)[[1]]
training <- train[inTrain,]
testing <- train[-inTrain,]

## Clean Data: Consolidate columns to remove
# 1. Remove NAs
countNA <- sapply(training, function(y) sum(is.na(y)))
noise <- c(grep(1,as.vector(countNA)))

# 2. Remove unrelated predictors
## Column 1: X - Because this is just a unit ID and does not help predict classe
## Column 2 & 7: user_name & num_window because these makes the model too biased to the individual
##   performing the activity (which is not the intent of this prediction model)
## Column 3, 4 & 5: raw_timestamp_part 1 & 2 and cvtd_timestamp
##   Since the test subjects (individuals) were given specific
##   instructions on how to perform each type of "classe". Time is unrelated to their performance
##   Time is unrelated to the way the individuals work.
noise <- c(1:7, noise)

## PreProcess 1: Remove Near Zero Variables
nzv1 <- nearZeroVar(training, saveMetrics=TRUE)
nzv1[1:20,]
nzv <- nearZeroVar(training)
noise <- unique(c(noise,nzv))

#Multi-thread (faster cpu processing)
cluster <- makeCluster(detectCores() - 3) #left 3 cores for PC's sake
registerDoParallel(cluster)

#Train Control setup
ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, allowParallel=TRUE)

#Random Forest - RFx1 (with PCA)
set.seed(4006)
system.time(modRF1 <- train(classe ~., method="rf", data=training[,-noise], trControl=ctrl, 
              preProcess=c("center", "scale", "pca")))
#Random Forest - RFx2 (no PCA)
set.seed(4006)
system.time(modRF2 <- train(classe ~., method="rf", data=training[,-noise], trControl=ctrl, 
                             preProcess=c("center", "scale")))
#Stochastic Gradient Boosting
set.seed(4006)
system.time(modGBM <- train(classe ~., method="gbm", data=training[,-noise], trControl=ctrl, 
                preProcess=c("center", "scale"), verbose=FALSE))
stopCluster(cluster)
registerDoSEQ()

#Results
modResults <- resamples(list(RF1=modRF1,RF2=modRF2,GBM=modGBM))
modSumm <- rbind(modRF1=getTrainPerf(modRF1),modRF2=getTrainPerf(modRF2),modGBM=getTrainPerf(modGBM))

#Compare models using Accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(modResults, scales=scales)

#Model Results
modRF2$finalModel
confusionMatrix(modRF2)
modRF2Imp <- varImp(modRF2, scale=F)
plot(modRF2Imp, top = 20)

#Prediction Results (testing set)
predRF2 <- predict(modRF2, newdata=testing)
postResample(predRF2, testing$classe)
confusionMatrix(predRF2, testing$classe)

#Random Sampling Tests
t <- training[sample(1:nrow(training),5000, replace = TRUE),]

predt <- predict(modRF2, t)
predx <- predict(modRF1, t)
predg <- predict(modGBM, t)
postResample(predt, t$classe)
postResample(predx, t$classe)
postResample(predg, t$classe)

#Prediction Results (validation set)
predRFV <- predict(modRF2, newdata=validation)
postResample(predRFV, validation$problem_id)
confusionMatrix(predRFV, validation$classe)