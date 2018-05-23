#Author    : Ajit Dharmarajan (L00131684)
#Lab No    : CA3
#Subject   :  Prediction on the Impact of House Prices depending on  
#             House Type and Number of Bedrooms             
#Date      : 23rd May 2018

#-------------------------------------------------#
#                                                 #
#    Create Data using Web Scrapping the site     #
#                WWW.DAFT.IE                      #
#                                                 #
#-------------------------------------------------#

#Set libraries for processing.
library(caret)
library(rvest)
library(randomForest)

#Define the URL from where the data to be scrapped
#daft <- read_html("http://www.daft.ie/donegal/property-for-sale/letterkenny/")
#daft <- read_html("http://www.daft.ie/donegal/houses-for-sale/")
daft <- read_html("http://www.daft.ie/donegal/houses-for-sale/letterkenny/")
daft
# html_nodes
#Get the listing number using CSS selector
ListNo <- daft %>% html_nodes(".sr_counter") %>% html_text()
ListNo

#Get the Title of the property
Title <- daft %>% html_nodes(".search_result_title_box a") %>% html_text()
Title
#Remove the newline (\n) comment from the Title
Title <- gsub("\n", "", Title)

#Remove excess space from the Title
Title <- gsub("  ", "", Title)

#Remove rows with data as 'Learnmore
Title_Fin <- data.frame(Variable1 = c(Title))
Title_Final1 <- Title_Fin[!Title_Fin$Variable1 %in% c('Learn more'),]
Title_Final1
Title_Final <- format(Title_Final1, justify = "left")
Title_Final

#Get the Price of the property
Price <- daft %>% html_nodes(".price") %>% html_text()
Price <- gsub("  ", "", Price)
Price <- gsub("€", "", Price)
Price <- gsub("AMV: ", "", Price)
Price <- gsub("Price On Application", "200,000", Price)
Price

#Get the House Type of the property
House_Type <- daft %>% html_nodes(".info li:nth-child(1)") %>% html_text()
#Remove the newline (\n) comment and spaces from the House_Type
House_Type <- gsub("\n", "", House_Type)
House_Type <- gsub("  ", "", House_Type)
House_Type

#Get the No of Bedrooms in the property
No_of_Beds <- daft %>% html_nodes(".info li:nth-child(2)") %>% html_text()
#Remove the newline (\n) comment and spaces from the No_of_Beds
No_of_Beds <- gsub("\n", "", No_of_Beds)
No_of_Beds <- gsub("  ", "", No_of_Beds)
No_of_Beds

#Get the No of Bathrooms in the property
No_of_Baths <- daft %>% html_nodes(".info li~ li+ li") %>% html_text()
#Remove the newline (\n) comment and spaces from the No_of_Baths
No_of_Baths <- gsub("\n", "", No_of_Baths)
No_of_Baths <- gsub("  ", "", No_of_Baths)
No_of_Baths

#Display the Dataframe
#Daft_Table <- data.frame(ListNo, Title_Final, Price, House_Type, No_of_Beds, No_of_Baths)
Daft_Table <- data.frame(House_Type, No_of_Beds, Price)
#Daft_Table <- data.frame(Title_Final, House_Type, No_of_Beds, No_of_Baths, Price)
Daft_Table


#-------------------------------------------------#
#                                                 #
#    Load the data from csv file and set the      #
#         Training and validation data            #
#                                                 #
#-------------------------------------------------#
#Set libraries for processing.
library(caret)
library(rvest)
library(MASS)
library(kernlab)

# Define and Load from CSV
Daft_Table <- read.csv("C:/Users/Owner/Documents/Visual Studio 2015/Projects/CA3/daft.csv", header = TRUE)

# Define and load from Load from CSV
dataset <- Daft_Table
dataset
#Set the column name in the csv dataset
#colnames(dataset) <- c("Title_Final", "House_Type", "No_of_Beds", "No_of_Baths", "Price")
colnames(dataset) <- c("House_Type", "No_of_Beds", "Price")

#-----   Create a Validation Dataset
#create a list of 80% of the rows from the original dataset for training purpose
validation_index <- createDataPartition(dataset$Price, p = 0.80, list = FALSE)

#Select 20% of the data for validation
validation <- dataset[-validation_index,]

#use the 80% of the data to train and test the models
dataset <- dataset[validation_index,]

#-------------------------------------------------#
#                                                 #
#             Summarise the Dataset               #
#                                                 #
#-------------------------------------------------#

#Lets see how many rows and columns are present in out dataset
dim(dataset)

#Lets see the attributes of the columns
sapply(dataset, class)

#Lets have an eyeball on the data by displaying the first few rows
head(dataset)


#---- Lets see the various class within our columns
#Lets see the level of  Price
levels(dataset$Price) 

#Lets see the distribution of the class for the rows in percentage and count
percentage <- prop.table(table(dataset$Price)) * 100
cbind(freq = table(dataset$Price), percentage = percentage)

#Lets get the Statistical Summary of the classes (mean, minimum and maximum values)
summary(dataset)

#-------------------------------------------------#
#                                                 #
#             Visualize the Dataset               #
#                                                 #
#-------------------------------------------------#

#---- Univariate Plots
#Split for input and output

dataset
x <- dataset[, 1:2]
y <- dataset[, 3]

# boxplot for each attribute on one image
par(mfrow = c(1, 2))
for (i in 1:2) {
    boxplot(x[, i], main = names(dataset)[i])
}

# generate a barplot
plot(y)


#---- Multivariate Plots
#generate Scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#generate box and whiskar plots for each attribute
featurePlot(x=x, y=y, plot="box")


# generate density plots for each attribute by class value
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
featurePlot(x = x, y = y, plot = "density", scales = scales)


#-------------------------------------------------#
#                                                 #
#           Evaluate using Algorithms             #
#                                                 #
#-------------------------------------------------#

#---- Test Harness
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

#---- Build Models
# a) linear algorithms
set.seed(7)
fit.lda <- train(Price ~ ., data = dataset, method = "lda", metric = metric, trControl = control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Price ~ ., data = dataset, method = "rpart", metric = metric, trControl = control)
# kNN
set.seed(7)
fit.knn <- train(Price ~ ., data = dataset, method = "knn", metric = metric, trControl = control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Price ~ ., data = dataset, method = "svmRadial", metric = metric, trControl = control)
# Random Forest
set.seed(7)
fit.rf <- train(Price ~ ., data = dataset, method = "rf", metric = metric, trControl = control)

#----Select best model
# summarize accuracy of models
results <- resamples(list(cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
summary(results)


# compare accuracy of models by plot
dotplot(results)

# summarize Best Model
print(fit.svm)

#-------------------------------------------------#
#                                                 #
#              Make the Prediction                #
#                                                 #
#-------------------------------------------------#


# estimate skill of LDA on the validation dataset
predictions <- predict(fit.knn, validation)
confusionMatrix(predictions, validation$Price)