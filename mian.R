# install package
if (!suppressWarnings(require(h2o))) {
    install.packages("h2o")
    library(h2o)
}
if (!suppressWarnings(require(data.table))) {
    install.packages("data.table")
    library(data.table)
}
# h2o init
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '15g') 
# read data
train_data <- h2o.importFile("data/train.csv")
train_data[, 1] <- as.factor(train_data[, 1])
test_data <- h2o.importFile("data/test.csv")
# h2o tarin
splits <- h2o.splitFrame(data = train_data, ratios = 0.70, seed =1)
train <- splits[[1]]
test <- splits[[2]]
# y
X_test <- test[, -1]
y_test <- test[, 1]
# buiding model
dp_model <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train_data, 
                             hidden = c(1024, 1024, 1024), activation = "RectifierWithDropout",
                             epochs = 8000, input_dropout_ratio = 0.2, train_samples_per_iteration = -1,
                             classification_stop = -1, l1 = 1e-5)
# predict train 25%
pred <- h2o.predict(dp_model, newdata = X_test)
predf <- h2o.performance(dp_model, y_test)
print(h2o.confusionMatrix(predf))
# final result
fpred <- h2o.predict(dp_model, newdata = test_data)
pred_data <- as.data.table(fpred)
pred_data <- data.table(ImageId = seq_along(pred_data$predict), label = pred_data$predict)
h2o.exportFile(as.h2o(pred_data), path = "result/result.csv", force = TRUE)
