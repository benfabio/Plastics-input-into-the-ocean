
##### 19/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train 1'000 GAM models per GNI class by using all potential PV set with 4 PVs
#   - Use the k values and the lists of PVs chosen from previous script a
#   - Predict annual MSW per country usinng either all 1000 GAMs (ensemble mean) or the top 20 or something
#   - Calculate ensemble predictions (means across the 1000 predictions, for each country and year) and compute error again       

### Last update: 23/01/23

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
library("mgcv")
library("lme4")
#library("MuMIn")
#library("SDMTools")
#library("merTools")
library("rworldmap")

worldmap <- getMap(resolution = "high")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

### First, as usual, get the PVs, remove outliers, scale them etc. Basically: prepare data for the models
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
# m.missing
#  Ok, drop Niger & Kyrgyzstan
# m.missing <- m.missing[!(m.missing$country %in% c("Niger","Kyrgyzstan")),]
# dim(m.missing)
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


### Create FUN to train and predict the GAMs
# inc <- "L"
train.predict.gams <- function(inc) {
                  
                  if(inc == "H") {
                      
                      dat <- scaled.H
                      list.pred <- list.preds.H
                      l <- length(list.pred)
                      k.values <- c(15:25)
                      
                  } else if(inc == "UM") {
                      
                      dat <- scaled.UM
                      list.pred <- list.preds.UM
                      l <- length(list.pred)
                      k.values <- c(15:25)
                      
                  } else if(inc == "LM") {
                      
                      dat <- scaled.LM
                      list.pred <- list.preds.LM
                      l <- length(list.pred)
                      k.values <- c(6:10)
                      
                  } else {
                      
                      dat <- scaled.L
                      list.pred <- list.preds.L
                      l <- length(list.pred)
                      k.values <- c(4:8)
                      
                  } # eo if else loop
                  
                  # Convert 'Country' from Char to Factor
                  dat$country <- as.factor(dat$country)
                  
                  message(paste("\nTraining 1,000 GAMs for GNI-",inc,"\n", sep = ""))
  
                  ### For every PV set in list.pred, run N models, N being 1000/number of PV sets (l)
                  # i <- 2 # for testing
                  require("parallel")
                  res <- mclapply(c(1:l), function(i) {
                              
                              # Get vector of pred names based on i
                              preds <- list.pred[[i]]
                              form <- paste(preds, collapse = '+')
                              # Useless message
                              message( paste("Training GAMs for GNI-",inc," based on ",form,sep = "") )
                              
                              # Define the number of n times to run the PV set 
                              N <- round(1000/l ,0.1) # convert to integer
                              
                              for(n in c(1:N)) {
      
                                  # Create train and test dataset to run the GAM and compute r2 and MSE from CV
                                  
                                  # If else loop
                                  if( length(preds) == 5 ) {
                                      
                                      training <- dat[,c("y",preds,"country")]
                                      colnames(training) <- c("y","x1","x2","x3","x4","x5","country")
                                  
                                      # train GAM
                                      gam1 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp") + s(x4,k=sample(k.values,1),bs="tp") + 
                                                  s(x5,k=sample(k.values,1),bs="tp") + s(country,bs="re"), data = training)
                                              
                                      # And one for the countries not in the training set
                                      gam2 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp") + s(x4,k=sample(k.values,1),bs="tp") + 
                                                  s(x5,k=sample(k.values,1),bs="tp"), data = training)
          
                                      # Predict on FULL set
                                      newdata <- dat[,c(preds,"country")]
                                      colnames(newdata) <- c("x1","x2","x3","x4","x5","country")
                                      pred.full <- mgcv::predict.gam(object = gam1, newdata = newdata)
                                      pred.full <- exp((pred.full*(max[1] - min[1])) + min[1])
                                      pred.full[which(pred.full < 0)] <- 0
                         
                                  } else if( length(preds) == 4 ) {
                                      
                                      training <- dat[,c("y",preds,"country")]
                                      colnames(training) <- c("y","x1","x2","x3","x4","country")
                                  
                                      # train GAM
                                      gam1 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp") + s(x4,k=sample(k.values,1),bs="tp") + 
                                                  s(country,bs="re"), data = training)
                                              
                                      # And one for the countries not in the training set
                                      gam2 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp") + s(x4,k=sample(k.values,1),bs="tp"), data = training)
          
                                      # Predict on FULL set
                                      newdata <- dat[,c(preds,"country")]
                                      colnames(newdata) <- c("x1","x2","x3","x4","country")
                                      pred.full <- mgcv::predict.gam(object = gam1, newdata = newdata)
                                      pred.full <- exp((pred.full*(max[1] - min[1])) + min[1])
                                      pred.full[which(pred.full < 0)] <- 0                
          
                                  } else {
                                      
                                      training <- dat[,c("y",preds,"country")]
                                      colnames(training) <- c("y","x1","x2","x3","country")
                                  
                                      # train GAM
                                      gam1 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp") + s(country,bs="re"), data = training)
                                              
                                      # And one for the countries not in the training set
                                      gam2 <- gam(y ~ s(x1,k=sample(k.values,1),bs="tp") + s(x2,k=sample(k.values,1),bs="tp") + 
                                                  s(x3,k=sample(k.values,1),bs="tp"), data = training)
          
                                      # Predict on FULL set
                                      newdata <- dat[,c(preds,"country")]
                                      colnames(newdata) <- c("x1","x2","x3","country")
                                      pred.full <- mgcv::predict.gam(object = gam1, newdata = newdata)
                                      pred.full <- exp((pred.full*(max[1] - min[1])) + min[1])
                                      pred.full[which(pred.full < 0)] <- 0  
          
                                  } # eo if else loop
                                           
                                  # Compute R2 of full model
                                  measure <- exp((dat$y * (max[1] - min[1])) + min[1])
                                  r2 <- 1- sum((pred.full-measure)^2)/sum(pred.full^2)
                                  # Compute mse of full model
                                  mse <- (sum((dat$y - pred.full)^2) / nrow(newdata))
                                  # & AIC of full model
                                  aic <- AIC(gam1)
                                  # Deviance explained
                                  devexpl <- round(summary(gam1)$dev.expl,4)
                                  # Summarize in ddf
                                  skillz <- data.frame(GNI = inc, n = n, formula = form, R2 = r2, MSE = mse, AIC = aic, DevExpl = devexpl)
                                  
                                  # Save GAM object
                                  setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_models_training")
                                  save(gam1, file = paste("gam.full_",inc,"_",form,"_",n,".Rdata", sep = "") )
                                  save(gam2, file = paste("gam.fill_no_re_",inc,"_",form,"_",n,".Rdata", sep = "") )
                                  save(skillz, file = paste("table_skills_gam.full_",inc,"_",form,"_",n,".Rdata", sep = "") )
                                  
                                  ### Make prediction and compute error to original data 
                                  all.countries <- c( unique(as.character( m.missing[m.missing$GNI == inc,"country"] )) , as.character(unique(dat$country)) )
                                  pred <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                                  pred[,1] <- unique(all.countries)
                                  colnames(pred) <- c("Country", 1990:2019)
                                  # class(pred)
                                  # Same for errors
                                  error_country_perc <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                                  error_country_perc[,1] <- unique(all.countries)
                                  colnames(error_country_perc) <- c("Country", 1990:2019)

                                  # Fill error and pred data.frame with for loop. Make if else loop to account for the data in m.missing too !
                                  # land <- "Ukraine"
                                  for( land in all.countries ) { 
                                        
                                        message( paste("Saving predictions and errors for ", land, sep = "") )
                                        
                                        if( land %in% unique(dat$country) ) {
                                            
                                            # get the corresponding j index from country.details
                                            j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                                            # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                                            dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                                            names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                                            dat.temp <- dat.temp[,c(preds,"country")]
                                            
                                            # Prepare scaled data for prediction
                                            if( length(preds) == 5 ) {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:5)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:5)] <- c("x1","x2","x3","x4","x5")
                                                
                                            } else if( length(preds) == 4 ) {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:4)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:4)] <- c("x1","x2","x3","x4")
                                                
                                            } else {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:3)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:3)] <- c("x1","x2","x3")
                                                
                                            }
                                            
                                            # Predict y for each year for country y
                                            predict_test_GAM <- exp((predict(gam1, scaled.temp) * (max[1] - min[1])) + min[1])
                                            # If ever negatove values, convert to zero
                                            predict_test_GAM[which(predict_test_GAM < 0)] <- 0
                                        
                                            # Find the position (YEARs between 1990:2015) to fill in the empty error dataset
                                            error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
                                        
                                            # Compute and supply error 
                                            # ( (predict_test_GAM[error.pos] - y_org[error.pos,j])/y_org[error.pos,j] )*100
                                            er <- ( (predict_test_GAM[error.pos] - y_org[error.pos,j])/y_org[error.pos,j] )*100
                                            error_country_perc[error_country_perc$Country == land,c(error.pos)] <- er
                                            # error_country_perc[error_country_perc$Country == land,c(error.pos)] <- (predict_test_GAM[error.pos] / y_org[error.pos,j]) *100-100
                                            
                                            # And supply prediction to pred matrix
                                            pred[pred$Country == land,-1] <- predict_test_GAM
                                            
                                        } else if ( land %in% unique(m.missing$country) ) {
                                            
                                            # get the corresponding j index from country.details
                                            j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                                            # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                                            dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                                            names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                                            dat.temp <- dat.temp[,c(preds,"country")]
                                            
                                            # Prepare scaled data for prediction
                                            if( length(preds) == 5 ) {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:5)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:5)] <- c("x1","x2","x3","x4","x5")
                                                
                                            } else if( length(preds) == 4 ) {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:4)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:4)] <- c("x1","x2","x3","x4")
                                                
                                            } else {
                                                
                                                scaled.temp <- as.data.frame(scale(dat.temp[,c(1:3)], center = min[preds], scale = max[preds] - min[preds]))
                                                scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                                scaled.temp$country <- dat.temp$country                                            
                                                colnames(scaled.temp)[c(1:3)] <- c("x1","x2","x3")
                                                
                                            }
                                                      
                                            # Predict y for each year for country y
                                            predict_test_GAM2 <- exp((predict(gam2, scaled.temp) * (max[1] - min[1])) + min[1])
                                            # If ever negatove values, convert to zero
                                            predict_test_GAM2[which(predict_test_GAM2 < 0)] <- 0
                                            
                                            # error not possible to measure because not data
                                            #error_country_perc[error_country_perc$Country == land,] <- NA

                                            # And supply prediction to pred matrix
                                            pred[pred$Country == land,-1] <- predict_test_GAM2
                                            
                                        } # eo else if loop
                                        
                                    } # for loop
                                    
                                    # Check error_country & pred data.frames
                                    # error_country ; pred
                                    error_avg <- rowMeans(error_country_perc[,c(2:nyears+1)], na.rm = T)
                                    error_country_perc$mean <- error_avg
                                    # summary(error_country_perc)
                                   
                                    # Save outputs 
                                    setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_predictions")
                                    write.table(x = error_country_perc, file = paste("table_error_perc_",inc,"_",form,"_",n,".txt", sep = ""), sep = "\t")
                                    write.table(x = pred, file = paste("table_pred_",inc,"_",form,"_",n,".txt", sep = ""), sep = "\t") 
                                    
                                    
                                } #
                                
                        }, mc.cores = 10
                          
                  ) # eo lapply - i in l
  
} # eo FUN - train.predict.gams()

train.predict.gams("H")
train.predict.gams("UM")
train.predict.gams("LM")
train.predict.gams("L")

### Check the skills of the trained models
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_models_training")
length(dir()[grep("table_skills_gam.full_H",dir())]) 
length(dir()[grep("table_skills_gam.full_UM",dir())]) 
length(dir()[grep("table_skills_gam.full_LM",dir())])
length(dir()[grep("table_skills_gam.full_L_",dir())]) 
# Ok all have 1000 models
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
ddf <- bind_rows(res)
dim(ddf); summary(ddf)

### R2
plot <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "R2", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("R2") + theme_bw()
            
ggsave(plot = plot, filename = "boxplot_full.gams_R2_GNI_23_01_23.pdf", dpi = 300, height = 5, width = 4)

### MSE
plot <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("MSE") + theme_bw()
            
ggsave(plot = plot, filename = "boxplot_full.gams_MSE_GNI_23_01_23.pdf", dpi = 300, height = 5, width = 4)

### Dev Explained
plot <- ggplot(aes(x = factor(GNI), y = DevExpl, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("Deviance explained (%)") + theme_bw()
#
ggsave(plot = plot, filename = "boxplot_full.gams_DevExpl_GNI_23_01_23.pdf", dpi = 300, height = 5, width = 4)


### And check % error for each class/ country?
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_predictions")
# length(dir()[grep("table_error_perc_H",dir())]) 
# length(dir()[grep("table_error_perc_UM",dir())]) 
# length(dir()[grep("table_error_perc_LM",dir())])
# length(dir()[grep("table_error_perc_L_",dir())]) 
files <- dir()[grep("table_error_perc",dir())]
# d <- read.table("table_error_perc_UM_p3+p4+p5+p6_91.txt", sep = "\t", h = T)
res <- lapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) })
# Rbind
ddf <- bind_rows(res)
colnames(ddf)
# Melt
m.ddf <- melt(ddf[,c("Country","mean")], id.vars = "Country")
dim(m.ddf)
head(m.ddf)
summary(m.ddf$value) # from -19% to +42%, but median error is 0.37% only ! 
sum.error <- data.frame(m.ddf %>% group_by(Country) %>% summarize(median = median(value)) ) # eo ddf
# summary(sum.error) 



### 23/01/23: Compute mean annual ensemble predictions. Re-compute error of the ensemble predictions to the y_org. 
### Compare to the mean error from the 1'000 predictions (data in the "table_error_perc" tables).
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_predictions")
# Load all predictions and compute mean/median annual MSW per country
files <- dir()[grep(paste("table_pred_", sep = ""),dir())] # length(files)
res <- lapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) } ) # eo lapply
# Rbind
ddf <- bind_rows(res)
#dim(ddf); head(ddf) 
colnames(ddf)[c(2:length(ddf))] <- as.character(c(1990:2019))
# unique(ddf$Country) # should be 217 - YES 
# Melt to put years as vector
m.ddf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
colnames(m.ddf)[c(2:3)] <- c("Year","MSW")
summary(m.ddf[m.ddf$Country == "Botswana",])

# Convert Inf to NAs
is.na(m.ddf) <- sapply(m.ddf, is.infinite)
summary(m.ddf[m.ddf$Country == "Botswana",])

# Compute ensemble metrics 
ensemble.range <- data.frame(m.ddf %>% group_by(Country,Year) %>% 
        summarize(Median = median(MSW, na.rm = T), IQR = IQR(MSW, na.rm = T), 
                Q25th = quantile(MSW, na.rm = T)[2], Q75th = quantile(MSW, na.rm = T)[4],
                Mean = mean(MSW, na.rm = T), Stdev = sd(MSW, na.rm = T)) 
) # eo ddf
# Check
summary(ensemble.range)
# ensemble.range[ensemble.range$Mean > 4,]
# ensemble.range[ensemble.range$Median > 4,]

# Inf values in average values?
# ensemble.range[is.na(ensemble.range$Mean),]
# ensemble.range[is.na(ensemble.range$Stdev),] # Botswana
# summary(m.ddf[m.ddf$Country == "Botswana",])

# Dcast to have years as columns
ensemble <- dcast(ensemble.range, Country ~ Year)
dim(ensemble) # 217x31 - good
summary(ensemble)

### Ok, save ensemble predictions of MSWc
write.table(x = ensemble.range, "table_ranges_GAM_median_predictions+ranges_22_02_23.txt", sep = "\t")


### Compute error of predictions to y_org
error_ensemble <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(unique(ensemble$Country)) ) )
error_ensemble[,1] <- unique(ensemble$Country)
colnames(error_ensemble) <- c("Country", 1990:2019)
# Calculate the predicted MSW for each country and get the error if measurements are available
# land <- "Armenia"
for(land in unique(ensemble$Country)) { 
      message( paste("Compute error of ensembles means for ", land, sep = "") )  
      # get the corresponding j index from country.details
      country.details2 <- country.details[country.details$Country %in% ensemble$Country,]
      j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
      # Find the position (YEARs between 1990:2015) to fill in the empty error dataset
      error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
      # Get corresponding predictions
      predictions <- ensemble[ensemble$Country == land,error.pos]
      # Compute % error and provide to "error_ensemble"      
      er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
      error_ensemble[error_ensemble$Country == land,c(error.pos)] <- er
} # for loop
# Check
#summary(error_ensemble)
#dim(error_ensemble) # good
### Save error table
write.table(error_ensemble, file = "table_GAM_median_errors_percentage_23_01_23.txt", sep = ";")

### And derive pluriannual mean of errors
m.error <- melt(error_ensemble, id.vars = "Country")
# dim(m.error)
# head(m.error)
# summary(m.error$value) # from -49% to +109%, but median error is -0.27% only ! 
sum.error.ensembles <- data.frame(m.error %>% group_by(Country) %>% summarize(median = median(value, na.rm = T)) ) # eo ddf
sum.error.ensembles[order(sum.error.ensembles$median, decreasing = T),]
# Ok exactly the same of course :D (you numb numb)

# ### Map this error
# n <- joinCountryData2Map(sum.error.ensembles, joinCode = "NAME", nameJoinColumn = "Country")
# # plot map
# pdf(file = paste("map_ensemble_error_GAM_13_12_19.pdf", sep = ""), width = 10, height = 6)
#     mapCountryData(n, nameColumnToPlot = "median", mapTitle = "",
#     catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#     borderCol = "white", lwd = 0.1, addLegend = T)
# dev.off()


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------