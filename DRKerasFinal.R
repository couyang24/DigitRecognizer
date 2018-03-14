library(tidyverse)
library(keras)
library(tensorflow)


train<-data.matrix(read_csv("train.csv"))
test<-data.matrix(read_csv("test.csv"))

dim(train)
dim(test)

colnames(train)

train[,1] %>% table() %>% barplot(col=1:10)

train[4,-1] %>% matrix(nrow=sqrt(dim(train)[2]-1),byrow=T) %>% apply(2,rev)%>% t() %>% image(col=grey.colors(255))

train.label<-train[,1] %>% to_categorical()

train.feature<-train[,-1] %>% normalize()
test.feature<-test %>% normalize()

dim(train.feature)<-c(nrow(train.feature),28,28,1)
dim(test.feature)<-c(nrow(test.feature),28,28,1)

model<-keras_model_sequential()

model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(5,5),padding = 'Valid',
                activation = 'relu', input_shape = c(28,28,1))%>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),padding = 'Same',
                activation = 'relu')%>%
  layer_batch_normalization()%>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3),padding = 'Same',
                activation = 'relu')%>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),padding = 'Same',
                activation = 'relu')%>%
  layer_batch_normalization()%>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides=c(2,2)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_flatten() %>% 
  layer_dense(units=1024,activation='relu')%>%
  layer_dense(units=512,activation='relu')%>%
  layer_dense(units=256,activation='relu')%>%
  layer_dense(units=10,activation='softmax')

model%>%compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics='accuracy'
)

datagen <- image_data_generator(
  featurewise_center = F,
  samplewise_center=F,
  featurewise_std_normalization = F,
  samplewise_std_normalization=F,
  zca_whitening=F,
  horizontal_flip = F,
  vertical_flip = F,
  width_shift_range = 0.10,
  height_shift_range = 0.10,
  zoom_range = 0.1,
  rotation_range = 10
)


datagen %>% fit_image_data_generator(train.feature)

history<-model %>%
  fit_generator(flow_images_from_data(train.feature, train.label, datagen, batch_size = 40),
                steps_per_epoch = nrow(train.feature)/40, epochs = 100)

plot(history)

pred<-model %>% predict_classes(test.feature,batch_size=40)

cnnsubmission<-data.frame(ImageId=1:nrow(test),Label=pred)

write.csv(cnnsubmission, file="cnnsubmission.csv", row.names=F)
