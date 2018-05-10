# devtools::install_github("dfalbel/capsnet")

if (!require("pacman")) install.packages("pacman") 
pacman::p_load(tidyverse, keras, tensorflow, capsnet, data.table)

train <- fread("train.csv") %>% data.matrix()
test <- fread("test.csv") %>% data.matrix()

train_label <- train[,1] %>% to_categorical(10)

train <- train[,-1] %>% 
  normalize() %>% 
  array_reshape(dim = c(nrow(train), sqrt(ncol(train)), sqrt(ncol(train)), 1))

test <- test %>% 
  normalize() %>% 
  array_reshape(dim = c(nrow(test), sqrt(ncol(test)), sqrt(ncol(test)), 1))


model <- create_capsnet(input_shape = c(28, 28, 1), n_class = 10L, num_routing = 3L)
