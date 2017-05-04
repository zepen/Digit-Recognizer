# install package
if (!suppressWarnings(require(h2o))) {
    install.packages("h2o")
    library(h2o)
}
if (!suppressWarnings(require(caret))) {
    install.packages("caret")
    library(caret)
}
if (!suppressWarnings(require(data.table))) {
    install.packages("data.table")
    library(data.table)
}
# h2o init
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '8g')
# read data
train_data <- fread("data/train.csv")
train_data[, label := as.factor(label)]
test_data <- fread("data/test.csv")
# h2o tarin
train_h2o <- as.h2o(train_data)
test_h2o <- as.h2o(test_data)
splits <- h2o.splitFrame(data = train_h2o, ratios = c(0.75), seed = 1)
train <- splits[[1]]
test <- splits[[2]]
# y
X_test <- test[, -1]
y_test <- test[, 1]
# buiding model
dp_model <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train,
                             hidden = c(1024, 1024, 2048, 1024, 1024),
                             loss = c("CrossEntropy"), activation = "RectifierWithDropout",
                             epochs = 200, rate = 0.01, rate_decay = 1.0, momentum_start = 0.5,
                             momentum_stable = 0.99, input_dropout_ratio = 0.2, initial_weight_scale = 0.01)
# predict train 25%
pred <- h2o.predict(dp_model, newdata = X_test)
print(confusionMatrix(as.matrix(pred$predict), as.matrix(y_test)))
# predict complete test
fpred <- h2o.predict(dp_model, newdata = test_h2o)
pred_data <- as.data.frame(fpred)
# build predict data
pred_data <- data.frame(ImageId = seq_along(pred_data$predict), label = pred_data$predict)
# save result
readr::write_csv(pred_data, path = "result/result.csv")
