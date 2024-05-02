
##### 23/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the overarching project is to model and predict waste production per country and per year (1990-2015) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using random forest models (RF), but also neural networks (NNET) if robust enough, or GAMs.
##### Ultimately, you need to provide to C. Laufkötter a long table (countriesxyear) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train and optimize neural networks models based on the same predictors set as GAMs and RF
#   - finds the best initial model architecture for the variable selection (both layers and nodes) - 1000 NNET models

### Last update: 24/01/23

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
library("rworldmap")
library("neuralnet")

worldmap <- getMap(resolution = "high")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

### First, as usual, get the PVs, remove outliers, scale them etc. Basically: prepare data for the models
setwd(paste(WD,"/data/", sep = "")) ; dir()
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

### Load complete datasets
scaled.H <- read.table("data_scaled_H.txt", h = T, sep = "\t")
scaled.UM <- read.table("data_scaled_UM.txt", h = T, sep = "\t")
scaled.LM <- read.table("data_scaled_LM.txt", h = T, sep = "\t")
scaled.L <- read.table("data_scaled_L.txt", h = T, sep = "\t")
scaled.all <- rbind(scaled.H,scaled.UM,scaled.LM,scaled.L) # dim(scaled.all)
error_country <- get(load("table_errors_country.Rdata"))
min <- get(load("vector_min_values_PVs.Rdata"))
max <- get(load("vector_max_values_PVs.Rdata"))
names(min) <- c("y","p1","p2","p3","p4","p5","p6")
names(max) <- c("y","p1","p2","p3","p4","p5","p6")
nyears <- length(error_country)
ncountries <- nrow(error_country)

### Prepare tables for filling the predictions for missing countries (countries ith PV values but no MSW data)
missing.countries <- country.details$Country[!(country.details$Country %in% unique(scaled.all$country))]
# length(missing.countries)
missing <- data.frame(country = missing.countries)
missing$GNI <- NA
missing$index <- NA
# c <- "Aruba"
for(c in missing.countries) {
    paste(c, sep = "")
    missing[missing$country == c,"GNI"] <- country.details[country.details$Country == c,"GNI.Classification"]
    missing[missing$country == c,"index"] <- as.numeric(rownames(country.details[country.details$Country == c,]))
} # eo for loop - c in missing.countries

# And provide estimates of y/p1/p2/p3 etc. per year
missing$y <- NA
missing$p1 <- NA
missing$p2 <- NA
missing$p3 <- NA
missing$p4 <- NA
missing$p5 <- NA
missing$p6 <- NA
missing <- cbind(missing, as.data.frame(matrix(NA, ncol = nyears, nrow = length(missing.countries))) )
colnames(missing)[c(11:length(missing))] <- as.character(c(1990:2019))
#  head(missing)
m.missing <- melt(missing, id.vars = c("country","GNI","index","y","p1","p2","p3","p4","p5","p6"))
# head(m.missing) # drop last col
m.missing <- m.missing[,-c(length(m.missing))]
# colnames(m.missing)[11] <- "year"
# dim(m.missing)

### Fill estimates of PVs in a for loop
# c <- "Argentina"
for(c in missing.countries) {
     i <- missing[missing$country == c,"index"]
     m.missing[m.missing$country == c,"y"] <- y[,i] # should always be NA
     m.missing[m.missing$country == c,"p1"] <- p1[,i]
     m.missing[m.missing$country == c,"p2"] <- p2[,i]
     m.missing[m.missing$country == c,"p3"] <- p3[,i]
     m.missing[m.missing$country == c,"p4"] <- p4[,i]
     m.missing[m.missing$country == c,"p5"] <- p5[,i]
     m.missing[m.missing$country == c,"p6"] <- p6[,i]
} # eo for loop
# Check
# nrow(m.missing)/ length(unique(m.missing$country)) # 30 years, 93 countries ok


### Define predictors set for income classes
### High income 
list.preds.H <- list()
for(i in 1:nrow(t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))) ) {
    list.preds.H[[i]] <- t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))[i,]
}
# list.preds.H 
### Remove those vectors that have both p3 and p4: 1, 7, 8, 11, 12, 15
list.preds.H <- list.preds.H[-c(1,7,8,11,12,15)]
# Add the 2 sets with 5 PVs that include either p3 or p4
list.preds.H[[10]] <- c("p1","p2","p3","p5","p6")
list.preds.H[[11]] <- c("p1","p2","p4","p5","p6")

### Medium income: also issue between p3 and p4, so same list as H class
list.preds.UM <- list.preds.H


### Low-medium income: issue between p1/p2 & p3/p4 --> no sets with 5 PVs available.
list.preds.LM <- list()
for(i in 1:nrow(t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))) ) {
    list.preds.LM[[i]] <- t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))[i,]
}
# list.preds.LM
### Remove those vectors that have both p3 and p4 or p1 and p2: 1:8,11,12,15 (keep: 9,10,13,15)
list.preds.LM <- list.preds.LM[c(9,10,13,15)]
# Only 4 sets for now (whereas H and UM have 11 sets). For further tests, try also including sets with 3 PVs
list.preds.LM.2 <- list()
for(i in 1:nrow(t(combn(c("p1","p2","p3","p4","p5","p6"), m = 3))) ) {
    list.preds.LM.2[[i]] <- t(combn(c("p1","p2","p3","p4","p5","p6"), m = 3))[i,]
}
# list.preds.LM.2 
# Retain: 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20
list.preds.LM.2 <- list.preds.LM.2[c(6:10,12:16,19,20)]
list.preds.LM <- append(list.preds.LM,list.preds.LM.2)
# 16 sets


### Low income: issue between p1/p3, p1/p4, p1/p5, p3/p4 and p3/p5
list.preds.L <- list()
for(i in 1:nrow(t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))) ) {
    list.preds.L[[i]] <- t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))[i,]
}
# list.preds.L
### Keep: 14 only 
list.preds.L <- list.preds.L[14]
### Too few sets of parameters and very data points (n = 54) - might want to include sets with only 3 parameters too
list.preds.L.2 <- list()
for(i in 1:nrow(t(combn(c("p1","p2","p3","p4","p5","p6"), m = 3))) ) {
    list.preds.L.2[[i]] <- t(combn(c("p1","p2","p3","p4","p5","p6"), m = 3))[i,]
}
# list.preds.L.2
# From this one, keep: 4,13,14,15,16,20
list.preds.L.2 <- list.preds.L.2[c(4,13,14,15,16,20)]
list.preds.L <- append(list.preds.L,list.preds.L.2)
# 7 sets


### -----------------------------------------------------------------

### Make FUN to optimize the parameters of the NNET
# Arguments to test the fun
# inc <- "H"
# i <- 11
# params <- c(2:7)
# form <- y ~ p1 + p2 + p3 + p4 + p5 + p6

nnet.optimizer <- function(inc) {
                  
                  # Define the data 
                  if(inc == "H") {
                      
                        dat <- scaled.H
                        list.pred <- list.preds.H
                        l <- length(list.pred)
                        layers <- c(1:5)
                      
                  } else if(inc == "UM") {
                      
                        dat <- scaled.UM
                        list.pred <- list.preds.UM
                        l <- length(list.pred) 
                        layers <- c(1:5)
                      
                  } else if(inc == "LM") {
                      
                        dat <- scaled.LM
                        list.pred <- list.preds.LM
                        l <- length(list.pred)
                        layers <- c(1:4)
                      
                  } else {
                      
                        dat <- scaled.L
                        list.pred <- list.preds.L
                        l <- length(list.pred)
                        layers <- c(1:3)
                      
                  } # eo if else loop
                  
                  # Set initial error
                  MSE.NN.cv <- NA
                  MSE.NN <- matrix(999, ncol = length(layers), nrow = 10)
                  test <- matrix(layers, 1, length(layers) )
  
                  # For every set of predictors
                  require("parallel")
                  # l <- 3
                  # full <- TRUE # switch to test with all 6 PVs
                  # if(full == TRUE) {
   #
   #                    params <- c("p1","p2","p3","p4","p5","p6")
   #                    form1 <- paste(params, collapse = '+')
   #                    form2 <- as.formula(paste("y~", form1, sep = ""))
   #                    nlayers <- c(1:7)
   #
   #                    for(j in 1:10) {
   #
   #                            cat(paste0("Run ",j," of 10\n"))
   #
   #                            # Create train and test dataset
   #                            pos <- sample(1:10, nrow(dat), replace=T)
   #                            while(length(table(pos)) != 10) {
   #                                    pos <- sample(1:10, nrow(dat), replace=T)
   #                            } # eo while
   #                            trainNN <- dat[pos!=10 ,c("y",params)]
   #                            testNN <- dat[pos==10,c("y",params)]
   #
   #                            # Loop for each model architecture
   #                            # n <- 5
   #                            for(n in 1:length(nlayers)) {
   #
   #                                    hid <- test[!is.na(test[,n]),n]
   #                                    message(paste("Running model: ",inc," based on ",form1, " using ",n, " layers", sep = "") )
   #
   #                                    # Train NNET model
   #                                    NN <- neuralnet(form2, trainNN, rep = 10, hidden = hid )
   #                                    # summary(NN)
   #
   #                                    # Perform 10-fold CV
   #                                    for(k in 1:10) {
   #                                           # Create test dataset
   #                                           testNN <- dat[pos == k, ]
   #                                           # Prediction using neural network
   #                                           predict_testNN <- compute(NN, testNN[,params],rep=which.min(NN$result.matrix[1,]))
   #                                           predict_testNN <- predict_testNN$net.result
   #                                           # Unsacle for CV
   #                                           predict_testNN.cv <- exp((predict_testNN * (max[1] - min[1])) + min[1])
   #                                           testNN.cv <- exp((testNN$y * (max[1] - min[1])) + min[1])
   #                                           # Prevent negative values
   #                                           predict_testNN[which(predict_testNN<0)] <- 0
   #                                           # Calculate Mean Square Error (MSE)
   #                                           MSE.NN.cv[k] = (sum((testNN.cv - predict_testNN.cv)^2) / nrow(testNN))
   #                                    } # eo for loop - k in 1:10
   #
   #                                    # Get mean MSE of the 10 runs
   #                                    MSE.NN[j,i] <- mean(MSE.NN.cv)
   #
   #                                    cat(paste(", MSE: ",round(MSE.NN[j,i], digits = 5)))
   #
   #                                    if( MSE.NN[j,i] == min(MSE.NN[,i], na.rm = T) ) {
   #
   #                                        # Prediction for all data points
   #                                        pred <- compute(NN, dat[,params])
   #                                        pred <- pred$net.result
   #                                        pred <- exp((pred * (max[1] - min[1])) + min[1]) # unscale
   #                                        pred[which(pred<0)] <- 0
   #                                        measure <- exp((dat$y * (max[1] - min[1])) + min[1]) # unscale
   #                                        r2 <- summary(lm(pred~measure))$r.squared
   #                                        skillz <- data.frame(GNI = inc, formula = form1, R2 = r2, MSE = MSE.NN[j,i], nHL = n) # eo ddf
   #                                        save(skillz, file = paste("table_skills_",inc,"_",form1,"_",n,".Rdata",sep="") )
   #                                        # Plot pred v measure
   #                                        setwd(WD)
   #                                        pdf(file = paste(inc,"_",form1,"_",n,".pdf",sep=""), width = 10, height = 5)
   #                                              par(mfrow=c(1,2))
   #                                              plot(measure, pred, main=paste0("Model: 6-",paste(hid,collapse = "-"),"-1"), col=ifelse(pos==10,"red","black"))
   #                                              abline(0,1)
   #                                              boxplot(MSE.NN.cv, main=paste0("MSE: ", MSE.NN[j,i]))
   #                                              mtext(paste("r^2:",r2))
   #                                              points(rep(1,10),MSE.NN.cv, col=c(rep("grey",9),"red"))
   #                                         dev.off()
   #
   #                                         cat(", R^2: ", round(r2,digits=3), "\n", sep = "")
   #
   #                                     } else {
   #
   #                                         cat("\n")
   #
   #                                     } # eo if else loop
   #
   #                           } # eo for loop - i in layers
   #
   #                    } # eo for loop - j in 1:10
   #
   #                    # Select best architecture & create plot
   #                    avg <- colMeans(MSE.NN)
   #                    mini <- sapply(1:ncol(test),function(x){min(MSE.NN[,x])}) # get minimal MSE of each architecture
   #
   #                    # Re-arrange the data for plotting (order by the MSE)
   #                    MSE.NN.order <- MSE.NN[order(MSE.NN)]
   #                    MSE.NN.diff <- NA
   #                    avg.order <- avg[order(avg)]
   #                    avg.diff <- NA
   #                    mini.order <- mini[order(mini)]
   #                    mini.diff <- NA
   #
   #                    for(i in 1:(length(MSE.NN)-1)) {
   #                            MSE.NN.diff[i] <- (MSE.NN.order[i]/MSE.NN.order[i+1])
   #                    } # eo for loop - i in 1:(length(MSE.NN)-1
   #
   #                    for(i in 1:(length(avg)-1)) {
   #                            avg.diff[i] <- (avg.order[i]/avg.order[i+1])
   #                    } # eo for loop - i in 1:(length(avg)-1
   #
   #                    for(i in 1:(length(avg)-1)) {
   #                            mini.diff[i] <- (mini.order[i]/mini.order[i+1])
   #                    } #  eo for loop - i in 1:(length(avg)-1)
   #
   #                    # plot re-arranged data
   #                    setwd(WD)
   #                    pdf(file = paste("nnet_arch_",inc,"_",form1,".pdf",sep=""), width = 20, height = 15)
   #                        par(mfrow = c(3,2))
   #                        barplot(MSE.NN[order(MSE.NN)], las = 2, xlab = "NNET layers & nodes", ylab = "MSE ordered",
   #                            names = rep(c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers],each=10)[order(MSE.NN)])
   #                        plot(MSE.NN.diff, ylim = c(0,1),type = "b", xaxt="n", ylab = "Difference in ordered MSE")
   #                        axis(1, at = 1:length(MSE.NN), las = 2, labels = rep(c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1"),10)[order(MSE.NN)])
   #                        barplot(avg[order(avg)], las = 2, xlab = "NNET layers & nodes", ylab="Mean MSE ordered",xpd=F,
   #                            names = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers][order(avg)])
   #                        plot(avg.diff, ylim = c(0,1), type = "b", xaxt = "n", ylab = "Difference in mean MSE ordered")
   #                        axis(1, at = 1:length(avg), las = 2, labels = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[order(avg)])
   #                        barplot(mini[order(mini)],las = 2, xlab = "NNET layers & nodes", ylab="Min MSE ordered",xpd=F,
   #                            names = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers][order(mini)])
   #                        plot(mini.diff,ylim = c(0,1),type="b", xaxt = "n", ylab = "Difference in min MSE ordered")
   #                        axis(1, at = 1:length(mini), las = 2, labels = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[order(mini)])
   #                    dev.off()
   #
   #                } else {
                      
                    res <- mclapply(c(1:l), function(i) {
                      
                          # Get vector of pred names based on i
                          params <- list.pred[[i]]
                          form1 <- paste(params, collapse = '+')
                          form2 <- as.formula(paste("y~", form1, sep = ""))
                          # nlayers <- c(1:7)
                      
                          message(paste("\nTraining NNET for GNI-",inc," with ",paste("y~", form1, sep = ""), sep = ""))
                      
                          # Loop 10 times to avoid bias in training/test data
                          # j <- 3
                          for(j in 1:10) {
      
                                  cat(paste0("Run ",j," of 10\n"))
    
                                  # Create train and test dataset
                                  pos <- sample(1:10, nrow(dat), replace=T) 
                                  while(length(table(pos)) != 10) { 
                                          pos <- sample(1:10, nrow(dat), replace=T)
                                  } # eo while
                                  trainNN <- dat[pos != 10, c("y",params)] 
                                  testNN <- dat[pos == 10, c("y",params)]
    
                                  # Loop for each model architecture
                                  # n <- 5
                                  for(n in 1:length(layers) ) {
                              
                                          hid <- test[!is.na(test[,n]),n]
                                          message(paste("Running model: ",inc," based on ",form1, " using ",n, " layers", sep = "") )
                                  
                                          # Train NNET model
                                          NN <- neuralnet(form2, trainNN, rep = 10, hidden = hid)
      
                                          # Perform 10-fold CV
                                          for(k in 1:10) {
                                                 # Create test dataset
                                                 testNN <- dat[pos == k, ]
                                                 # Prediction using neural network
                                                 predict_testNN <- compute(NN, testNN[,params], rep = which.min(NN$result.matrix[1,]))
                                                 predict_testNN <- predict_testNN$net.result
                                                 # Unsacle for CV
                                                 predict_testNN.cv <- exp((predict_testNN * (max[1] - min[1])) + min[1])
                                                 testNN.cv <- exp((testNN$y * (max[1] - min[1])) + min[1])
                                                 # Prevent negative values
                                                 predict_testNN[which(predict_testNN < 0)] <- 0
                                                 # Calculate Mean Square Error (MSE)
                                                 MSE.NN.cv[k] = (sum((testNN.cv - predict_testNN.cv)^2) / nrow(testNN))
                                          } # eo for loop - k in 1:10
      
                                          # Get mean MSE of the 10 runs
                                          MSE.NN[j,n] <- mean(MSE.NN.cv)
      
                                          cat(paste(", MSE: ", round(MSE.NN[j,n], digits = 5)))
                                  
                                          if( MSE.NN[j,n] == min(MSE.NN[,n], na.rm = T) ) {
                                      
                                              # Prediction for all data points
                                              pred <- compute(NN, dat[,params])
                                              pred <- pred$net.result
                                              pred <- exp((pred * (max[1] - min[1])) + min[1]) # unscale
                                              pred[which(pred < 0)] <- 0
                                              measure <- exp((dat$y * (max[1] - min[1])) + min[1]) # unscale
                                              r2 <- summary(lm(pred ~ measure))$r.squared
                                              skillz <- data.frame(GNI = inc, formula = form1, R2 = r2, MSE = MSE.NN[j,n], nHL = n) # eo ddf
                                              
                                              setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/NNET_training/")
                                              save(skillz, file = paste("table_skills_",inc,"_",form1,"_",n,".Rdata",sep="") )
                                              
                                              # Plot pred v measure                                              
                                              pdf(file = paste(inc,"_",form1,"_",n,".pdf",sep=""), width = 10, height = 5)
                                                    par(mfrow=c(1,2))
                                                    plot(measure, pred, main=paste0("Model: 6-",paste(hid,collapse = "-"),"-1"), col=ifelse(pos==10,"red","black"))
                                                    abline(0,1)
                                                    boxplot(MSE.NN.cv, main=paste0("MSE: ", MSE.NN[j,n]))
                                                    mtext(paste("r^2:",r2))
                                                    points(rep(1,10),MSE.NN.cv, col=c(rep("grey",9),"red"))
                                               dev.off()
        
                                               cat(", R^2: ", round(r2,digits=3), "\n", sep = "")
        
                                           } else {
                                           
                                               cat("\n") 
                                           
                                           } # eo if else loop
                                   
                                 } # eo for loop - i in layers
                         
                          } # eo for loop - j in 1:10

                          # Select best architecture & create plot
                          avg <- colMeans(MSE.NN)
                          mini <- sapply(1:ncol(test), function(x) { min(MSE.NN[,x]) } ) # get minimal MSE of each architecture
  
                          # Re-arrange the data for plotting (order by the MSE)
                          MSE.NN.order <- MSE.NN[order(MSE.NN)]
                          MSE.NN.diff <- NA
                          avg.order <- avg[order(avg)]
                          avg.diff <- NA
                          mini.order <- mini[order(mini)]
                          mini.diff <- NA
                  
                          for(ii in 1:(length(MSE.NN)-1)) { 
                                  MSE.NN.diff[ii] <- (MSE.NN.order[ii]/MSE.NN.order[ii+1]) 
                          } # eo for loop - i in 1:(length(MSE.NN)-1
                  
                          for(ii in 1:(length(avg)-1)) { 
                                  avg.diff[ii] <- (avg.order[ii]/avg.order[ii+1]) 
                          } # eo for loop - i in 1:(length(avg)-1
                
                          for(ii in 1:(length(avg)-1)) { 
                                  mini.diff[ii] <- (mini.order[ii]/mini.order[ii+1])
                          } #  eo for loop - i in 1:(length(avg)-1)

                          # plot re-arranged data
 #                          setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/NNET_training/")
 #
 #                          pdf(file = paste("nnet_arch_",inc,"_",form1,".pdf",sep=""), width = 20, height = 15)
 #
 #                              # par(mfrow = c(3,2))
 # #                              barplot(MSE.NN[order(MSE.NN)], las = 2, xlab = "NNET layers & nodes", ylab = "MSE ordered",
 # #                                  names = rep(c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers],each=10)[order(MSE.NN)])
 # #                              plot(MSE.NN.diff, ylim = c(0,1),type = "b", xaxt="n", ylab = "Difference in ordered MSE")
 # #                              axis(1, at = 1:length(MSE.NN), las = 2, labels = rep(c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1"),10)[order(MSE.NN)])
 # #                              barplot(avg[order(avg)], las = 2, xlab = "NNET layers & nodes", ylab="Mean MSE ordered",xpd=F,
 # #                                  names = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers][order(avg)])
 # #                              plot(avg.diff, ylim = c(0,1), type = "b", xaxt = "n", ylab = "Difference in mean MSE ordered")
 # #                              axis(1, at = 1:length(avg), las = 2, labels = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[order(avg)])
 # #                              barplot(mini[order(mini)],las = 2, xlab = "NNET layers & nodes", ylab="Min MSE ordered",xpd=F,
 # #                                  names = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[layers][order(mini)])
 # #                              plot(mini.diff,ylim = c(0,1),type="b", xaxt = "n", ylab = "Difference in min MSE ordered")
 # #                              axis(1, at = 1:length(mini), las = 2, labels = c("6-1-1","6-2-1","6-3-1","6-4-1","6-5-1","6-6-1","6-7-1")[order(mini)])
 # #
 #
 #                              par(mfrow = c(3,2))
 #                              barplot(MSE.NN[order(MSE.NN)], las = 2, xlab = "NNET layers & nodes", ylab = "MSE ordered")
 #                              plot(MSE.NN.diff, ylim = c(0,1),type = "b", xaxt="n", ylab = "Difference in ordered MSE")
 #                              axis(1, at = 1:length(MSE.NN), las = 2)
 #                              barplot(avg[order(avg)], las = 2, xlab = "NNET layers & nodes", ylab="Mean MSE ordered", xpd = F)
 #                              plot(avg.diff, ylim = c(0,1), type = "b", xaxt = "n", ylab = "Difference in mean MSE ordered")
 #                              axis(1, at = 1:length(avg), las = 2)
 #                              barplot(mini[order(mini)],las = 2, xlab = "NNET layers & nodes", ylab="Min MSE ordered", xpd = F)
 #                              plot(mini.diff,ylim = c(0,1),type="b", xaxt = "n", ylab = "Difference in min MSE ordered")
 #                              axis(1, at = 1:length(mini), las = 2)
 #
 #                          dev.off()
                      
                      }, mc.cores = l
                  
            ) # eo mclapply
                      
                 # } # eo if else loop - full == TRUE

} # eo FUN - nnet.optimizer

# Run the function
nnet.optimizer("H")
nnet.optimizer("UM")
nnet.optimizer("LM")
nnet.optimizer("L")

### Check results 
setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/NNET_training/")
files <- dir()[grep('table_skills',dir())]
res <- lapply(files, function(f){d <- get(load(f)); return(d)}) # eo lapply
ddf <- bind_rows(res)
# dim(ddf); head(ddf)
data.frame(ddf %>% group_by(GNI,formula) %>% summarize(R2 = median(R2), MSE = median(MSE)))
data.frame(ddf %>% group_by(GNI,nHL) %>% summarize(R2 = median(R2), MSE = median(MSE)))

setwd("/net/kryo/work/fabioben/Inputs_plastics/plots/")

# Plot distribution of MSE and R2 per n layers and then facet per GNI
plot1 <- ggplot(aes(x = factor(nHL), y = R2), data = ddf) + geom_boxplot(fill = "grey55", colour = "black") + 
        xlab("Number of hidden layers") + ylab("R2 (from cross validation)") + theme_bw()
#
plot2 <- ggplot(aes(x = factor(nHL), y = MSE), data = ddf) + geom_boxplot(fill = "grey55", colour = "black") + 
        xlab("Number of hidden layers") + ylab("MSE (from cross validation)") + theme_bw()
        
# Save
ggsave(plot = plot1, filename = "boxplot_nnet_NHL_r2.pdf", dpi = 300, height = 5, width = 5)
ggsave(plot = plot2, filename = "boxplot_nnet_NHL_MSE.pdf", dpi = 300, height = 5, width = 5)

### And facet per GNI
plot1 <- ggplot(aes(x = factor(nHL), y = R2, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of hidden layers") + ylab("R2") + theme_bw() + 
        facet_wrap(~factor(ddf$GNI), ncol = 2, scales = "free")
#
plot2 <- ggplot(aes(x = factor(nHL), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of hidden layers") + ylab("MSE") + theme_bw() +
        facet_wrap(~factor(ddf$GNI), ncol = 2, scales = "free")
        
# Save
ggsave(plot = plot1, filename = "boxplot_nnet_NHL_r2_GNI.pdf", dpi = 300, height = 5.5, width = 6)
ggsave(plot = plot2, filename = "boxplot_nnet_NHL_MSE_GNI.pdf", dpi = 300, height = 5.5, width = 6)


### Conclusions on the nmber of hidden layers to choose:
# H: 3
# UM: 4
# LM: 3 
# L: 2


### 24/01/23: Compare MSE and R2 values across model types and per GNI: GAM vs. RF vs. NNET
# Get RF's skills table
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training")
# files <- dir()[grep('table_skills',dir())] #; files
# res <- lapply(files, function(f){ d <- get(load(f)); return(d)} ) # eo lapply
# ddf.rf <- bind_rows(res)
# # dim(ddf.rf); head(ddf.rf); summary(ddf.rf)
# ddf.rf$Model <- "RF"
#
# # Get GAM's skills table
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_models_training")
# files <- dir()[grep('table_skills',dir())]# ; files
# res <- lapply(files, function(f){ d <- get(load(f)); return(d)} ) # eo lapply
# ddf.gam <- bind_rows(res)
# # dim(ddf.gam); head(ddf.gam); summary(ddf.gam)
# ddf.gam$Model <- "GAM"
#
# # Get NNET's skills table
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/NNET_training/")
# files <- dir()[grep('table_skills',dir())] ; files
# res <- lapply(files, function(f){ d <- get(load(f)); return(d)} ) # eo lapply
# ddf.nnet <- bind_rows(res)
# # dim(ddf.nnet); head(ddf.nnet); summary(ddf.nnet)
# ddf.nnet$Model <- "NNET"
#
# # Merge by common colnames
# commons <- intersect(c(colnames(ddf.rf),colnames(ddf.gam)), colnames(ddf.nnet)) ; commons
# ddf.all.models <- rbind(ddf.rf[,commons], ddf.gam[,commons], ddf.nnet[,commons])
# # dim(ddf.all.models)
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots/")
# plot <- ggplot(aes(x = factor(Model), y = MSE, fill = factor(GNI)), data = ddf.all.models) + geom_boxplot(colour = "black") +
#         scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#         xlab("Model type") + ylab("MSE") + theme_bw() + facet_wrap(~factor(ddf.all.models$GNI), ncol = 2, scales = "free_y")
#
# ggsave(plot = plot, filename = "boxplot_all_models_MSE_GNI.pdf", dpi = 300, height = 7, width = 7)
#
# plot <- ggplot(aes(x = factor(Model), y = sqrt(MSE), fill = factor(GNI)), data = ddf.all.models) + geom_boxplot(colour = "black") +
#         scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#         xlab("Model type") + ylab("RMSE") + theme_bw() + facet_wrap(~factor(ddf.all.models$GNI), ncol = 2, scales = "free_y")
#
# ggsave(plot = plot, filename = "boxplot_all_models_RMSE_GNI.pdf", dpi = 300, height = 7, width = 7)
#
# plot <- ggplot(aes(x = factor(Model), y = R2, fill = factor(GNI)), data = ddf.all.models) + geom_boxplot(colour = "black") +
#         scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#         xlab("Model type") + ylab("R2") + theme_bw() + facet_wrap(~factor(ddf.all.models$GNI), ncol = 2, scales = "free_y")
#
# ggsave(plot = plot, filename = "boxplot_all_models_R2_GNI.pdf", dpi = 300, height = 7, width = 7)
#
# ### Examine median values
# data.frame(ddf.all.models %>% group_by(GNI,Model) %>% summarize(R2 = median(R2), MSE = median(MSE), RMSE = median(sqrt(MSE))))

### R2 of NNET models << GAM & RF BUT MSE << GAM and RF too...
### --> Use parameters determined above to generate some NNET predictions and compare errors !

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------

