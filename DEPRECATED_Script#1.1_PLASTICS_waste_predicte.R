
##### 16/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
# - Predict MSW from the RF models parameters determined in Part II of Script#1.0

### Last update: 17/01/23

### ------------------------------------------------------------------------------------------------------------------------------------------------

library("tidyverse")
library("reshape2")
library("scales")
library("RColorBrewer")
library("viridis")
library("ggsci")
library("ggthemes")
library("scales")
library("wesanderson")
library("randomForest")
library("rworldmap")

# Worldmap
worldmap <- getMap(resolution = "high")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

### Load some objects to perform predictions

### A°) Get PVs, remove outliers, scale them etc. Basically: prepare data for the models
setwd(paste(WD,"/data/complete_data/", sep = "")) ; dir()

MSW_collected_UN <- read.csv("MSW_collected_corrected_14_01_23.csv", na.strings = c("NA"), stringsAsFactors = F) # MSW = Municipal solid waste
colnames(MSW_collected_UN) <- c("Country", 1990:2019) # adjust colnames
young_pop <- read.csv("young_pop.csv", na.strings = c("NA"), stringsAsFactors = F)
colnames(young_pop) <- c("Country", 1990:2019)
share_urb_pop <- read.csv("share_urb_pop.csv", na.strings = c("NA"), stringsAsFactors = F)
colnames(share_urb_pop) <- c("Country", 1990:2019)
elec_acc <- read.csv("elec_acc.csv", na.strings = c("NA"), stringsAsFactors = F)
colnames(elec_acc) <- c("Country", 1990:2019)
GDP_per_capita <- read.csv("GDP.csv", na.strings = c("NA"), stringsAsFactors = F)
colnames(GDP_per_capita) <- c("Country", 1990:2019)
energy_consumption <- read.csv("energy_consumption.csv", stringsAsFactors = F)
colnames(energy_consumption) <- c("Country", 1990:2019)
greenhouse_gas_pP <- read.csv("greenhouse_gas.csv", stringsAsFactors = F)
colnames(greenhouse_gas_pP) <- c("Country", 1990:2019)
country.details <- read.csv("country_details.csv", stringsAsFactors = F)[,c(1,8)]
# Remove outliers
removed <- 0
n <- nrow(young_pop) - 1 
for(i in 1:n) {
      # Extract country
      temp <- as.numeric(MSW_collected_UN[i,-1])
      # select outlier
      remove <- boxplot.stats(temp)$out
      # add number of outliers to counter
      removed <- removed+length(remove)
      if( length(remove) > 0 ) { # if there are outliers:
          message(i)
          # replace outlier with NA
          temp[which(!is.na(match(temp,remove)))] <- NA
          # Write edited line to data frame
          MSW_collected_UN[i,-1] <- temp
      } # eo if loop
} # eo for loop 
# Set parameters: p = predictors, y = goal variable. Scale some PVs (GDP, energy_consumption etc.) to avoid errors in RF
p1 <- t(elec_acc[,-1])
p2 <- log(t(energy_consumption[,-1]))
p3 <- log(t(GDP_per_capita[,-1]))
p4 <- log(t(greenhouse_gas_pP[,-1]))
p5 <- t(share_urb_pop[,-1])
p6 <- t(young_pop[,-1])
nyears <- length(c(1990:2019))
ncountries <- length(unique(MSW_collected_UN$Country))
countries <- t(matrix(rep(elec_acc[,1], nyears), ncol = nyears))
years <- matrix(rep(1990:2019, ncountries), nrow = nyears)
GNI <- t(matrix(rep(country.details$GNI.Classification, nyears), ncol = nyears))
# Response variable y = MSW_collected_UN (logtransform it)
y <- log(t(MSW_collected_UN[,-1]))
y_org <- t(MSW_collected_UN[,-1])
# Create dataset with only complete data
p1_complete <- p1[!is.na(y)]
p2_complete <- p2[!is.na(y)]
p3_complete <- p3[!is.na(y)]
p4_complete <- p4[!is.na(y)]
p5_complete <- p5[!is.na(y)]
p6_complete <- p6[!is.na(y)]
y_complete <- y[!is.na(y)]
countries_complete <- countries[!is.na(y)]
year_complete <- years[!is.na(y)]
GNI_complete <- GNI[!is.na(y)]
te <- data.frame(y_complete, p1_complete, p2_complete, p3_complete, p4_complete, p5_complete, p6_complete)
names(te) <- c("y","p1","p2","p3","p4","p5","p6")
# Scale data for RF
max <- c(max(y_complete),max(p1),max(p2),max(p3),max(p4),max(p5),max(p6))
min <- c(min(y_complete),min(p1),min(p2),min(p3),min(p4),min(p5),min(p6))
scaled <- as.data.frame(scale(te, center = min, scale = max - min)) # scale...rather "range"
scaled$country <- countries_complete
scaled$year <- year_complete
scaled$GNI <- GNI_complete

nyears <- length(error_country)
ncountries <- nrow(error_country)

### Load complete datasets
scaled.H <- read.table("data_scaled_H.txt", h = T, sep = "\t")
scaled.UM <- read.table("data_scaled_UM.txt", h = T, sep = "\t")
scaled.LM <- read.table("data_scaled_LM.txt", h = T, sep = "\t")
scaled.L <- read.table("data_scaled_L.txt", h = T, sep = "\t")
error_country <- get(load("table_errors_country.Rdata"))
min <- get(load("vector_min_values_PVs.Rdata"))
max <- get(load("vector_max_values_PVs.Rdata"))

# Set a folder where all trained RF will be temporarily saved. Can be deleted once the 10 best models are selected.
temp.folder.path <- "/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/temp"

# Matrix with errors of the countries with highest impact according to Jambeck
#setwd("/net/kryo/work/public/ftp/plastics/scripts/data/complete/")
#country.details <- read.csv("country_details.csv", stringsAsFactors = F)
summary_percent <- matrix(NA, nrow = 1, ncol = 13)
colnames(summary_percent) <- c("Model nr.","Total","China","Indonesia","Philippines","Sri Lanka","Egypt",
                                 "Malaysia","Algeria","Turkey","Brazil","Morocco","United States")

summary.pos <- as.numeric(sapply(c("China","Indonesia","Philippines","Sri Lanka","Egypt","Malaysia","Algeria","Turkey","Brazil","Morocco","United States of America"),
                                 function(x) { which(country.details$Country == x) } # eo FUN which simply retunrs index of the country x
                     ) # eo sapply
) # eo numeric

### Set vectors of chosen predictors per income class (BEWARE: 1 = y, 2 = p1, 3 = p2, 4 = p3 etc.)
# H -> p2+p3+p4+p5
# UM -> p3+p4+p5+p6
# LM -> p1+p2+p4+p5
# L -> p2+p4+p5
var.selection.H <- c(1,3:6)
var.selection.UM <- c(1,4:7)
var.selection.LM <- c(1,2,3,5,6)
var.selection.L <- c(1,3,5,6)
form.H <- as.formula("y ~ p2 + p3 + p4 + p5")
form.UM <- as.formula("y ~ p3 + p4 + p5 + p6")
form.LM <- as.formula("y ~ p1 + p2 + p4 + p5")
form.L <- as.formula("y ~ p2 + p4 + p5")


### A) Train 10,000 RF models and save the best ones ------------------------------------------------------------------------------------------------

### Function that trains the RF
# For testing 
dat <- scaled.L
preds <- var.selection.L
form <- form.L
income <- "L"
mtry.best <- 3

train.RF <- function(dat, preds, form, income, mtry.best) {
  
                  # Set initial error
                  MSE.RF.old <- 999
                  MSE.RF.cv <- rep(NA,10)
                  MSE.RF <- rep(NA,1000)
                  r2 <- rep(NA,1000)
                  dat <- dat[,preds]
                  
                  message(paste("\nTraining 1,000 RF models for GNI-",income,"\n", sep = ""))
  
                  # Loop to get the best model
                  # pb <- txtProgressBar(min = 0, max = 10000, style = 3)
                  # k <- 23
                  for(k in 1:1000) {
                      
                          # Split into training and testing datasets
                          pos <- sample(1:10, nrow(dat), replace = T)
                          
                          while( length( table(pos) ) != 10 ) {
                               # Prevent that one number is not represented in small samples (e.g. scaled.L)
                                   pos <- sample(1:10, nrow(scaled.temp), replace = T)
                          } # while

                          # Create train dataset
                          trainRF <- dat[pos != 10,] # head(trainRF)
    
                          # Create model
                          RF <- randomForest(form, trainRF, mtry = mtry.best)
                          # summary(RF)
                          for(i in 1:10) {
                                  # Create train and test dataset
                                  testRF <- dat[pos==i,]
                                  # Prediction using neural network
                                  predict_testRF <- predict(RF, testRF[,-1])
                                  # Unscale for CV
                                  predict_testRF.cv <- exp((predict_testRF * (max[1] - min[1])) + min[1])
                                  testRF.cv <- exp((testRF$y * (max[1] - min[1])) + min[1])
                                  # Prevent negative values
                                  predict_testRF[which(predict_testRF<0)] <- 0
                                  # Calculate Mean Square Error (MSE)
                                  MSE.RF.cv[i] = (sum((testRF.cv - predict_testRF.cv)^2) / nrow(testRF))
                          } # eo for loop - i in 1:10
    
                          MSE.RF[k] <- MSE.RF.cv[10]
    
                          pred <- predict(RF, dat[,-1])
                          pred <- exp((pred*(max[1] - min[1])) + min[1])
                          pred[which(pred < 0)] <- 0
                          measure <- exp((dat$y * (max[1] - min[1])) + min[1])
    
                          r2[k] <- 1- sum((pred-measure)^2)/sum(pred^2)
    
                          # Save the RF if the error of unseen data isn't the largest. Calculate the error per country.
                          if( which.max(MSE.RF.cv) != 10 ) {
                              
                              setwd(temp.folder.path)
                              
                              save(RF, file = paste(income,".",k,".Rdata", sep = "") )
      
                              # If the RF is the "best" so far, save it for later
                              if( MSE.RF[k] < MSE.RF.old ) {
                                  
                                      setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/best")
                                      
                                      MSE.RF.old <- MSE.RF[k]
                                      save(RF, file = paste("model_RF_",income,"_best.Rdata", sep = "") )
        
                                      pdf(file = paste("pred_v_test_RF.",income,".best.pdf", sep = ""), width = 10, height = 5)
                                          par(mfrow=c(1,2))
                                          plot(measure, pred, main = "RF CV", col = ifelse(pos==10,"red","black") )
                                          abline(0,1)
                                          mtext(paste("r^2:",r2[k]))
                                          boxplot(MSE.RF.cv, main=paste0("MSE: ", MSE.RF[k]))
                                          points(rep(1,10), MSE.RF.cv, col=c(rep("grey",9),"red"))
                                      dev.off()
                                      cat(paste0(Sys.time()," New best: Model ",k,", MSE=",MSE.RF[k]," R2=",r2[k],"\n"))
                                      
                              } # if loop - MSE.RF[k] < MSE.RF.old
                              
                              pred <- matrix(NA, ncol = nyears+1, nrow = ncountries)
                              colnames(pred) <- c("Country", 1990:2019)
                              pred[,1] <- country.details$Country
      
                              # Calculate the predicted MSW for each country and get the error if measurements are available
                              # j <- 100 # country == 100
                              for(j in if(income == "all") {
                                   
                                        1:ncountries # should be the length of the n countries 
                                        
                                    } else { which(country.details$GNI_classification == income) }    
                              ) {       
                                  
                                        dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j] )
                                        names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6")
                                        dat.temp <- dat.temp[,preds]
                                        # Scale data
                                        scaled.temp <- as.data.frame(scale(dat.temp, center = min[preds], scale = max[preds] - min[preds]))
                                        scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                        predict_testRF <- exp((predict(RF, scaled.temp[,-1]) * (max[1] - min[1])) + min[1])
                                        predict_testRF[which(predict_testRF < 0)] <- 0
                                        error.pos <- which(!is.na(y_org[,j]))
                                        error_country[j,c(error.pos)] <- (predict_testRF[error.pos]/y_org[error.pos,j])*100-100
                                        pred[j,-1] <- predict_testRF
                                        
                                } # eo else if in the for loop
      
                                # Average error for each country
                                error_avg <- rowMeans(sqrt(error_country^2), na.rm = T)
                                    
                                # Write in summary file
                                summary_percent <- rbind(summary_percent, c(k, sum(error_avg, na.rm = T), error_avg[summary.pos]) )
                                
                        } # eo if loop - which.max(MSE.RF.cv) != 10
    
                } # eo for loop - k in 1:1000

                # save MSE
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/")
                write.csv(x = cbind(MSE.RF,r2), file = paste("table_MSE_r2_",income,"_17_01_23.csv", sep = ""), row.names = F, sep = "\t")
                write.csv(x = summary_percent, file = paste("table_error_perc_",income,"_17_01_23.csv", sep = ""), row.names = F, sep = "\t")

} # eo FUN - train.RF

### Run the training function above. Apply the selected variables and best mtry parameter value from previous scripts!
train.RF(dat = scaled.H, preds = var.selection.H, form = form.H, income = "H", mtry.best = 3) # done
train.RF(dat = scaled.UM, preds = var.selection.UM, form = form.UM, income = "UM", mtry.best = 2)
train.RF(dat = scaled.LM, preds = var.selection.LM, form = form.LM, income = "LM", mtry.best = 2)
train.RF(dat = scaled.L, preds = var.selection.L, form = form.L, income = "L", mtry.best = 3)



### B) Search for best model and plot ------------------------------------------------------------------------------------------------
# For testing the function below
income <- "H"
preds <- var.selection.H 

best.model <- function(income, preds) {
    
                  counter <- 0
                  
                  # Select only models where MSE and r2 are among the best
                  setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/")
                  ranks <- read.csv(paste("table_MSE_r2_",income,"_17_01_23.csv", sep = ""), sep = ",", h = T)
                  # head(ranks)
                  ranks$model.nb <- c(1:1000)
                  ranks$MSE_rank <- rank(ranks$MSE.RF)
                  ranks$r2_rank <- rank(-ranks$r2)
                  ranks$ranksum <- ranks$MSE_rank + ranks$r2_rank
                  # ranks
                  files <- paste(income,".",which(rank(ranks$ranksum) <= 20), sep = "")

                  # Copy the best models from the temp folder to the final folder
                  # ?file.copy
                  file.copy(from = paste(temp.folder.path,"/",files,".Rdata", sep = ""), to = "/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/best")
  
                  errors <- matrix(NA, ncol = ncountries, nrow = length(files))
  
                  # Load all models and calculate the error of each country
                  pb <- txtProgressBar(min = 0, max = length(files), style = 3)
                  # k <- files[2] # for testing for loop below
                  # test <- get(load(paste(k,".Rdata",sep="")))
                  for(k in files) {
                      
                          counter <- counter+1
                          # Df for estimates
                          pred <- matrix(NA, ncol = nyears+1, nrow = ncountries)
                          colnames(pred) <- c("Country",1990:2019)
                          pred[,1] <- country.details$Country
    
                          # Load the model
                          setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/best")
                          if( file.exists( paste(k,".Rdata",sep="")) ) {
                              
                                  RF <- get(load(paste(k,".Rdata", sep = "")))
                                  
                          } else { cat("\n",k,"does not exist" )
                          next
                          } # eo if else next loop    
    
                          # Use data from each country and calculate MSW pp
                          for(j in if( income == "all" ) { 
                                  1:ncountries
                              } 
                            else { which(country.details$GNI.Classification == income) 
                              }
                          ) # eo for loop
                          {
                              dat.temp <- data.frame(y[,j],p1[,j],p2[,j],p3[,j],p4[,j],p5[,j],p6[,j])
                              names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6")
                              dat.temp <- dat.temp[,preds]
                              # summary(dat.temp)
                              # Scale data 
                              scaled.temp <- as.data.frame(scale(dat.temp, center = min[preds], scale = max[preds] - min[preds]))
                              # scaled.temp
                              # Make predictions based on currently loaded RF from the top 20
                              predict_testRF <- exp((predict(RF, scaled.temp[,-1])* (max[1] - min[1])) + min[1])
                              # Turn negative values to zeros if tere are any predicted?
                              if( length(predict_testRF[which(predict_testRF < 0 )]) > 0 ) {
                                       predict_testRF[which(predict_testRF < 0 )] <- 0
                              } # eo if loop
                              
                              error.pos <- which(!is.na( y_org[,j] ) ) # y_org = original non logtransformed value of MSW for country j
                              error_country[j,c(error.pos)] <- c( (predict_testRF[error.pos]/y_org[error.pos,j])*100-100 )
                              pred[j,-1] <- predict_testRF
                          } #
    
                          errors[counter,] <- rowMeans(sqrt(error_country^2), na.rm = T)
    
                          setTxtProgressBar(pb, counter)
                          
                      } # eo for loop - k in files
                      
                      close(pb)
                      # dim(errors); str(errors)
                      errors[errors == "NaN"] <- NA
                      colnames(errors) <- pred[,1]
                      rownames(errors) <- files
                      
                      # Save errors
                      setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/")
                      write.table(errors, file = paste("table_model_errors_",income,".txt", sep = ""), sep = "\t")
  
                      # Get the position (nr) of these countries in the data frame
                      error.pos <- as.numeric(sapply(c("China","Indonesia","Philippines","Sri Lanka","Egypt","Malaysia",
                                      "Algeria","Turkey","Brazil","Morocco","United States of America"),
                                    function(x) {
                                        which(country.details$Country == x & country.details$GNI_classification == income) 
                                    } # eo FUN 
                            ) # eo sapply 
                      ) # eo numeric
                      error.factor <- c(11:1)[which(!is.na(error.pos))]
                      error.pos <- error.pos[which(!is.na(error.pos))]
                      #
                      adjusted.error <- sapply(1:length(files), function(x) { 
                              sum(errors[x,error.pos]*error.factor)
                          } # eo FUN
                      ) # eo sapply
  
                      ranks <- ranks[which(rank(ranks$ranksum) <= 20),]
                      ranks$error <- rowSums(errors, na.rm = T)
                      ranks$adjusted.error <- rank(adjusted.error)
                      ranks$Model_name <- files
                      
                      write.table(ranks, file = paste("table_ranks_best_models_",income,".txt", sep = ""), sep = "\t")
                      return(list(table = ranks, number = rownames(ranks)[which.min(ranks$adjusted.error)]) )
  
} # eo FUN - best.model

# Run the function for each income class. This creates a csv file with the pre-selected best models
best.H <- best.model("H", var.selection.H)$number      ; best.H     # Best model: 66
best.UM <- best.model("UM", var.selection.UM)$number   ; best.UM    # Best model: 66
best.LM <- best.model("LM", var.selection.LM)$number   ; best.LM    # Best model: 61
best.L <- best.model("L", var.selection.L)$number      ; best.L     # Best model: 15


### C) Run predictions based on the "best" model and plot spatial distribution of errors
# Function to make predictions and arguments to test it
income <- "L"
best.model <- best.L
preds <- var.selection.L

create.prediction <- function(income, best.model, preds) {
    
                            setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/best")
                            
                            RF <- get(load(paste(income,".",best.model,".Rdata",sep="")))
                            
                            # Use data from each country and calculate MSW pp
                            for(j in if(income == "all") { 
                                  1:ncountries 
                            } else { 
                                  which(country.details$GNI.Classification == income) 
                            } # eo else
                            ) # eo for loop 
                            {
                                dat.temp <- data.frame(y[,j],p1[,j],p2[,j],p3[,j],p4[,j],p5[,j],p6[,j])
                                names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6")
                                dat.temp <- dat.temp[,preds]
                                # dat.temp
                                # Scale data for neural network
                                scaled.temp <- as.data.frame(scale(dat.temp, center = min[preds], scale = max[preds] - min[preds]))
                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                # scaled.temp
                                predict_testRF <- exp((predict(RF, scaled.temp[,-1]) * (max[1] - min[1])) + min[1])
                                predict_testRF[which(predict_testRF < 0)] <- 0
                                
                                error.pos <- which( !is.na(y_org[,j]) )
                                
                                error_country[j,c(error.pos)] <- (predict_testRF[error.pos]/y_org[error.pos,j])*100-100
    
                                pred[j,-1] <- predict_testRF
    
                            } # eo - for loop
  
} # eo FUN - create.prediction

# data.frame to fill with RF estimates for estimates
pred <- matrix(NA, ncol = nyears+1, nrow = ncountries)
colnames(pred) <- c("Country", 1990:2019)
pred[,1] <- country.details$Country
# str(pred) ; head(pred) ; summary(pred)
# pred <- as.data.frame(pred)

# Create prediction using four models and save it
create.prediction("H", best.H, var.selection.H)
create.prediction("UM", best.UM, var.selection.UM)
create.prediction("LM", best.LM, var.selection.LM)
create.prediction("L", best.L, var.selection.L)

# CHECK
storage.mode(pred)
class(pred) ; str(pred) ; head(pred)
# NEED TO CONVERT TO NUM MATRIX 
pred2 <- data.matrix(pred[,c(2:nyears+1)], rownames.force = NA)
class(pred2) ; str(pred2)
#pred2 <- mapply(pred[,c(2:27)], FUN = as.numeric)

pred2 <- as.numeric( pred[,c(2:nyears+1)] )
summary(pred2)

predictions <- data.frame(Country = as.factor(pred[,1]))
predictions$MSW_1990 <- as.numeric(pred[,2])
predictions$MSW_1991 <- as.numeric(pred[,3])
predictions$MSW_1992 <- as.numeric(pred[,4])
predictions$MSW_1993 <- as.numeric(pred[,5])
predictions$MSW_1994 <- as.numeric(pred[,6])
predictions$MSW_1995 <- as.numeric(pred[,7])
predictions$MSW_1996 <- as.numeric(pred[,8])
predictions$MSW_1997 <- as.numeric(pred[,9])
predictions$MSW_1998 <- as.numeric(pred[,10])
predictions$MSW_1999 <- as.numeric(pred[,11])
predictions$MSW_2000 <- as.numeric(pred[,12])
predictions$MSW_2001 <- as.numeric(pred[,13])
predictions$MSW_2002 <- as.numeric(pred[,14])
predictions$MSW_2003 <- as.numeric(pred[,15])
predictions$MSW_2004 <- as.numeric(pred[,16])
predictions$MSW_2005 <- as.numeric(pred[,17])
predictions$MSW_2006 <- as.numeric(pred[,18])
predictions$MSW_2007 <- as.numeric(pred[,19])
predictions$MSW_2008 <- as.numeric(pred[,20])
predictions$MSW_2009 <- as.numeric(pred[,21])
predictions$MSW_2010 <- as.numeric(pred[,22])
predictions$MSW_2011 <- as.numeric(pred[,23])
predictions$MSW_2012 <- as.numeric(pred[,24])
predictions$MSW_2013 <- as.numeric(pred[,25])
predictions$MSW_2014 <- as.numeric(pred[,26])
predictions$MSW_2015 <- as.numeric(pred[,27])
predictions$MSW_2016 <- as.numeric(pred[,28])
predictions$MSW_2017 <- as.numeric(pred[,29])
predictions$MSW_2018 <- as.numeric(pred[,30])
predictions$MSW_2019 <- as.numeric(pred[,31])

summary(predictions)

### Save
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions")
write.table(predictions, file = paste("table_predictions_all_countries_RF.txt", sep = ""), sep = "\t")
save(predictions, file = paste("table_predictions_all_countries_RF.Rdata", sep = ""))

### Map errors per country
setwd("/net/kryo/work/fabioben/Inputs_plastics/plots")
error_avg <- rowMeans(sqrt(error_country^2), na.rm = T)
error_avg # vector of mean error per country
  
# create new list including carribbean & oceania islands
d <- data.frame(country = c(pred[c(-4,-7),1],
                paste(worldmap$GEOUNIT[which(worldmap$GEO3 == "Caribbean")]),
                paste(worldmap$GEOUNIT[which(worldmap$GEO3 == "South Pacific")])),
                value = c(error_avg[c(-4,-7)], rep(error_avg[4],23), rep(error_avg[7],22))
) # eo ddf
# Check content 
dim(d); head(d)
d

### Map, after removing Kyrgyzstan, palestine and Niger
d2 <- d[-which(d$country %in% c("Kyrgyzstan","Niger","Palestine")),]
# fit country codes
# ?joinCountryData2Map
n <- joinCountryData2Map(d2, joinCode = "NAME", nameJoinColumn = "country")
breaks <- c(0:100,1e99)
color <- c(rev(heat.colors(100)),"violet")
# plot map
pdf(file = paste("map_mean_error_RF_18_01_23.pdf", sep = ""), width = 36, height = 18)
             mapCountryData(n, nameColumnToPlot="value",  mapTitle="RF model error [mean(predicted-measured)]",
                        catMethod = breaks, colourPalette = color, oceanCol = "lightblue", missingCountryCol = "grey", 
                        borderCol = "black", lwd = 0.1, addLegend = F ) # eo mapCountryData
             coord <- coordinates(n)
             pos <- match(d$country,rownames(coord))
             text(coord[pos,1],coord[pos,2], paste0(round(d2$value,digits=0),"%"), cex = 0.5)
             legend.gradient(cbind(x =c(-250,-200,-200,-250), y = c(50,50,0,0)), cols = color, limits = c(0,100) )
dev.off()
### Nice, errors are lower than before !!

d2 <- d2[order(d2$value, decreasing = T),]
write.table(x = d2, file = "table_mean_error_allcountries_RF.txt", sep = "\t")

# Some macro-statistics
#error_avg <- rowMeans(sqrt(error_country^2), na.rm = T)
country.details2 <- country.details[-which(country.details$Country.Code %in% c("Kyrgyzstan","Niger","Palestine") ),]

sapply(unique(country.details2$Continent), function(x) { 
        median(abs(error_avg[which(country.details2$Continent == x)]), na.rm = T)
    } # eo FUN 
) # eo sapply
#    AS   EU    AF    OC   CAR   S-AM   N-AM 
#  5.71  3.14  9.44  3.40 12.95  3.73  8.94 

sapply(unique(country.details2$GNI.Classification), function(x) { 
        median(abs(error_avg[which(country.details2$GNI.Classification == x)]), na.rm = T)
    } # eo FUN 
) # eo sapply
#  L    UM   LM   H 
# 9.44 6.03 7.04 3.15 


### 09/12/19: Re-load tables of mean annual waste production and model errors to make better looking maps than the default one
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions")

if("dplyr" %in% (.packages())){
          detach("package:dplyr", unload=TRUE) 
          detach("package:plyr", unload=TRUE) 
} 
require("plyr")
require("rgdal")
require("maptools")
pred <- read.table("table_predictions_all_countries_RF_18_01_23.txt", sep = "\t", h = T)
errors <- read.table("table_mean_error_allcountries_RF.txt", sep = "\t", h = T)
head(pred); head(errors)
### Melt pred and compute average MSW
m.pred <- melt(pred, id.vars = "Country")
mean.pred <- data.frame(m.pred %>% group_by(Country) %>% summarize(MSW = mean(value,na.rm = T)) )
summary(mean.pred)

# Make discrete categories of mean annual MSW for better palette
mean.pred$bin <- factor(cut_interval(mean.pred$MSW,7))
levels(mean.pred$bin)
levels <- str_replace_all(levels(mean.pred$bin), ",", "-")
levels <- gsub("\\[|\\]", "", levels) ; levels <- gsub("\\(|\\)", "", levels)
levels
levels(mean.pred$bin) <- levels

### Try with the more classic tools
n <- joinCountryData2Map(mean.pred, joinCode = "NAME", nameJoinColumn = "Country")
breaks <- seq(from = 0, to = 3, by = 0.5)
color <- viridis::magma(10)[c(4:10)]
# str(n)
# plot map
pdf(file = paste("map_mean_error_RF_18_01_23.pdf", sep = ""), width = 10, height = 6)
    mapCountryData(n, nameColumnToPlot = "MSW", mapTitle = "",
    catMethod = breaks, colourPalette = color, oceanCol = "white", missingCountryCol = "grey65", 
    borderCol = "white", lwd = 0.1, addLegend = T) 
    #legend.gradient(cbind(x = c(-250,-200,-200,-250), y = c(50,50,0,0)), cols = color, limits = c(0,3) )
dev.off()
### Last one is kind of good :-)

### Map errors
head(errors); summary(errors)
n <- joinCountryData2Map(errors, joinCode = "NAME", nameJoinColumn = "country")
breaks <- seq(from = 0, to = 70, by = 10)
color <- viridis::magma(10)[c(2:10)]
# str(n)
# plot map
pdf(file = paste("map_mean_error_RF_18_01_23.pdf", sep = ""), width = 10, height = 6)
    mapCountryData(n, nameColumnToPlot = "value", mapTitle = "",
    catMethod = breaks, colourPalette = color, oceanCol = "white", missingCountryCol = "grey65", 
    borderCol = "white", lwd = 0.1, addLegend = T) 
    #legend.gradient(cbind(x = c(-250,-200,-200,-250), y = c(50,50,0,0)), cols = color, limits = c(0,3) )
dev.off()


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
