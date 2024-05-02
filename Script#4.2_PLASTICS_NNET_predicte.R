
##### 24/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the overarching project is to model and predict waste production per country and per year (1990-2015) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using random forest models (RF), but also neural networks (NNET) if robust enough, or GAMs.
##### Ultimately, you need to provide to C. Laufkötter a long table (countriesxyear) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train and NNETs based on all 4 PV sets and the values of nHL chosen based on Script#4.1

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
# m.missing
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

### -----------------------------------------------------------------

### Write FUN to predict MSW using NNET
# inc <- "L"
nnet.predicter <- function(inc) {
    
                  # Define the data 
                  if(inc == "H") {
                      
                      dat <- scaled.H
                      list.pred <- list.preds.H
                      l <- length(list.pred)
                      nLayers <- 3
                      
                  } else if(inc == "UM") {
                      
                      dat <- scaled.UM
                      list.pred <- list.preds.UM
                      l <- length(list.pred) 
                      nLayers <- 4
                      
                  } else if(inc == "LM") {
                      
                      dat <- scaled.LM
                      list.pred <- list.preds.LM
                      l <- length(list.pred)
                      nLayers <- 3
                      
                  } else {
                      
                      dat <- scaled.L
                      list.pred <- list.preds.L
                      l <- length(list.pred)
                      nLayers <- 2
                      
                  } # eo if else loop
  
                  # For every set of predictors
                  require("parallel")
                  # i <- 1
                  res <- mclapply(c(1:l), function(i) {
                      
                          # Get vector of pred names based on i
                          params <- list.pred[[i]]
                          form1 <- paste(params, collapse = '+')
                          form2 <- as.formula(paste("y~", form1, sep = ""))
                      
                          message(paste("\nTraining full NNET for GNI-",inc," with ",paste("y~", form1, sep = ""), sep = ""))
                      
                          # Train NNET model
                          NN <- neuralnet(form2, dat, rep = 25, hidden = nLayers)
      
                          # Prediction using neural network
                          predict.NN <- predict(NN, dat[,params], rep = which.min(NN$result.matrix[1,]) )[,1]
                          # Compute MSE and R2
                          predict.NN.unscaled <- exp((predict.NN * (max[1] - min[1])) + min[1])
                          # Prevent negative values
                          predict.NN.unscaled[which(predict.NN.unscaled < 0)] <- 0
                          # Get response var
                          MSW.obs <- exp((dat$y * (max[1] - min[1])) + min[1])
                          # Calculate Mean Square Error (MSE)
                          mse <- (sum((MSW.obs - predict.NN.unscaled)^2) / nrow(dat))
                          # Compute r2
                          r2 <- summary(lm(predict.NN.unscaled ~ MSW.obs))$r.squared
                          
                          # Save model and skillz
                          setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/models/NNET_models_training/")
                          skillz <- data.frame(GNI = inc, formula = form1, R2 = r2, MSE = mse) 
                          save(skillz, file = paste("table_skills_",inc,"_",form1,".Rdata", sep = "") )
                          save(NN, file = paste("NNET.full_",inc,"_",form1,".Rdata", sep = "") )
                          
                          ### Make prediction and compute error to original data 
                          all.countries <- c( unique(as.character( m.missing[m.missing$GNI == inc,"country"] )) , as.character(unique(dat$country)) )
                          pred <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                          pred[,1] <- unique(all.countries)
                          colnames(pred) <- c("Country", 1990:2019)
                          # Same for errors
                          error_country_perc <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                          error_country_perc[,1] <- unique(all.countries)
                          colnames(error_country_perc) <- c("Country", 1990:2019)
                          
                          # land <- "Mali"
                          # land <- "Togo"
                          
                          for( land in all.countries ) { 
                                    
                                    message( paste("Saving predictions and errors for ", land, sep = "") )
                                    
                                    if( land %in% unique(dat$country) ) {
                                        
                                        # get the corresponding j index from country.details
                                        j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                                        # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                                        dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                                        names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                                        dat.temp <- dat.temp[,c(params,"country")]
                                        
                                        scaled.temp <- as.data.frame(scale(dat.temp[,c(1:length(params))], center = min[params], scale = max[params] - min[params]))
                                        scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                        scaled.temp$country <- dat.temp$country
                                        # Predict y for each year for country y
                                        predicted <- exp( (predict(NN, scaled.temp[,params], rep = which.min(NN$result.matrix[1,]))[,1] * (max[1] - min[1])) + min[1])
                                        # If ever negatove values, convert to zero
                                        predicted[which(predicted < 0)] <- 0
                                        # Find the position (YEARs between 1990:2015) to fill in the empty error dataset
                                        error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
                                        # Compute and supply error 
                                        er <- ( (predicted[error.pos] - y_org[error.pos,j])/y_org[error.pos,j] )*100
                                        error_country_perc[error_country_perc$Country == land,c(error.pos)] <- er
                                        # And supply prediction to pred matrix
                                        pred[pred$Country == land,-1] <- predicted
                                        
                                    } else if ( land %in% unique(m.missing$country) ) {
                                        
                                        # get the corresponding j index from country.details
                                        j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                                        # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                                        dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                                        names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                                        dat.temp <- dat.temp[,c(params,"country")]
                                        
                                        scaled.temp <- as.data.frame(scale(dat.temp[,c(1:length(params))], center = min[params], scale = max[params] - min[params]))
                                        scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                        scaled.temp$country <- dat.temp$country
                                        # Predict y for each year for country y
                                        predicted <- exp( (predict(NN, scaled.temp[,params], rep = which.min(NN$result.matrix[1,]))[,1] * (max[1] - min[1])) + min[1])
                                        # If negative values, convert to zero
                                        predicted[which(predicted < 0)] <- 0
                                        # And supply prediction to pred matrix
                                        pred[pred$Country == land,-1] <- predicted
                                        
                                    } # eo else if loop
                                    
                          } # for loop - l in land
                                
                          # Check error_country & pred data.frames
                          error_avg <- rowMeans(error_country_perc[,c(2:nyears+1)], na.rm = T)
                          error_country_perc$mean <- error_avg
                               
                          # Save outputs 
                          setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/models/NNET_predictions")
                          write.table(x = error_country_perc, file = paste("table_error_perc_",inc,"_",form1,".txt", sep = ""), sep = "\t")
                          write.table(x = pred, file = paste("table_pred_",inc,"_",form1,".txt", sep = ""), sep = "\t")          
                                                
                      }, mc.cores = l
                  
            ) # eo mclapply

} # eo FUN - nnet.predicter

# Run the function
nnet.predicter("H")
nnet.predicter("UM")
nnet.predicter("LM")
nnet.predicter("L")

### Check the skills of the trained models
setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/models/NNET_models_training/")
files <- dir()[grep("table_skills",dir())] # files
res <- lapply(files, function(f) { d <- get(load(f)); d$file <- f; return(d) })
ddf <- bind_rows(res) ; rm(res); gc()

### Save plots 
setwd("/net/kryo/work/fabioben/Inputs_plastics/plots/")
# R2
plot <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "R2", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("R2") + theme_bw()    
ggsave(plot = plot, filename = "boxplot_full.nnet_R2_GNI_24_01_23.pdf", dpi = 300, height = 5, width = 4)

# MSE
plot <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("MSE") + theme_bw()
ggsave(plot = plot, filename = "boxplot_full.nnet_MSE_GNI_24_01_23.pdf", dpi = 300, height = 5, width = 4)


### Extract predictions and errors and compute ensembles
setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/models/NNET_predictions") #; dir()
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
# Compute median 
ensemble.med <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(MSW = median(MSW)) ) # eo ddf
summary(ensemble.med)
# Dcast
ensemble <- dcast(ensemble.med, Country ~ Year)
dim(ensemble) # 217x31 - good
summary(ensemble)

### Ok, save ensemble predictions of MSW and apply the same prediction fun above to RF
write.table(x = ensemble, "table_ensemble_predictions_NNET_24_01_23.txt", sep = "\t")


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
# summary(error_ensemble)
# dim(error_ensemble) 
### Save error table
write.table(error_ensemble, file = "table_NNET_median_errors_percentage_24_01_23.txt", sep = ";")


# ### Look at distributions of errors in the predictions
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/NNET_predictions")
# files <- dir()[grep("table_error",dir())]
# res <- lapply(files, function(f) {
#             d <- read.table(f, sep = "\t", h = T)
#             # Melt and add GNI and form
#             m.d <- melt(d, id.vars = c("Country","mean"))
#             m.d$GNI <- do.call(rbind,strsplit(as.character(f),"_"))[,4]
#             m.d$form <- do.call(rbind,strsplit(as.character(f),"_"))[,5]
#             m.d$form <- str_replace_all(m.d$form, ".txt", "")
#             colnames(m.d)[c(3:4)] <- c("Year","error")
#             m.d$Year <- str_replace_all(m.d$Year, "X", "")
#             # head(m.d)
#             return(m.d[,c(1,3:6)])
#         } # eo lapply
# ) # eo lapply
# # Rbind
# ddf <- bind_rows(res)
# head(ddf); dim(ddf)
# rm(res); gc()
# # For each formula and GNI, compute mean(error)
# data.frame(ddf %>% group_by(GNI) %>% summarize(mean = mean(error, na.rm=T), sd = sd(error,na.rm=T)))
# data.frame(ddf %>% group_by(GNI, form) %>% summarize(mean = mean(error, na.rm=T), sd = sd(error,na.rm=T)))
# ### All have seemingly low mean error but large std variations...examine min and max errors
# data.frame(ddf %>% group_by(GNI) %>% summarize(max = max(error, na.rm=T), min = min(error,na.rm=T)))
# data.frame(ddf %>% group_by(GNI, form) %>% summarize(max = max(error, na.rm=T), min = min(error,na.rm=T)))
# na.omit(ddf)[order(na.omit(ddf$error), decreasing = T),]
# na.omit(ddf[ddf$Country == "China",])


# ### Very very high errors :-/ I actually would not keep those predictions in the 1st place
# ### Compare to RF predictions
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/NNET_predictions")
#
# files <- c("table_pred_L_p1+p2+p4+p6.txt", "table_pred_L_p2+p3+p4+p6.txt",
#         "table_pred_LM_p1+p3+p4+p5.txt", "table_pred_LM_p2+p3+p4+p5.txt","table_pred_UM_p3+p4+p5+p6.txt",
#         "table_pred_H_p1+p2+p5+p6.txt","table_pred_H_p2+p3+p4+p5.txt","table_pred_H_p2+p3+p4+p6.txt",
#         "table_pred_H_p2+p3+p5+p6.txt","table_pred_H_p2+p4+p5+p6.txt","table_pred_H_p3+p4+p5+p6.txt"
# ) #
# # f <- "table_pred_L_p1+p2+p4+p6.txt"
# res <- lapply(files, function(f) {
#             d <- read.table(f, sep = "\t", h = T)
#             d$GNI <- do.call(rbind,strsplit(as.character(f),"_"))[,3]
#             d$form <- do.call(rbind,strsplit(as.character(f),"_"))[,4]
#             d$form <- str_replace_all(d$form,".txt","")
#             return(d)
#     }
# ) # eo lapply
# ddf <- bind_rows(res)
# dim(ddf); head(ddf)
#
# ### Melt to put ALL MSW estimates as a vector and detect those that exceed 5
# m.ddf <- melt(ddf, id.vars = c("Country","GNI","form"))
# colnames(m.ddf)[c(4,5)] <- c("Year","MSW")
# summary(m.ddf)
# m.ddf$Year <- str_replace_all(m.ddf$Year,"X","")
# # Compute mean
# ens.ddf <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(MSW = mean(MSW,na.rm = T)) ) # eo ddf
# # Dcast
# ens.nnet <- dcast(Country ~ Year, data = ens.ddf)
# dim(ens.nnet)
# summary(ens.nnet)
#
# # Now get the RF ensembles
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/RF_predictions2")
# ens.rf <- read.table("table_ensemble_predictions_RF2_13_12_19.txt", sep = "\t", h = T)
# colnames(ens.rf)[c(2:27)] <- str_replace_all(as.character(colnames(ens.rf)[c(2:27)]),"X","")
# dim(ens.nnet); dim(ens.rf)
# head(ens.nnet) ; head(ens.rf)
# # Compare them
#
# ### Awesome, no do the same with MSW predictions :-)
# m.ens.rf <- melt(ens.rf, id.vars = "Country")
# m.ens.nnet <- melt(ens.nnet, id.vars = "Country")
# head(m.ens.rf) ; head(m.ens.nnet)
# # Cbind
# m.preds <- data.frame(Country = m.ens.nnet$Country, Year = m.ens.nnet$variable, RF = m.ens.rf$value, NNET = m.ens.nnet$value)
# m.preds$GNI <- NA
# for(c in unique(m.preds$Country)) {
#         m.preds[m.preds$Country == c,"GNI"] <- country.details[country.details$Country == c,"GNI_classification"]
# } # for loop
# m.preds$label <- paste(m.preds$Country, m.preds$Year, sep = "_")
# summary(m.preds)
#
# require("ggrepel")
# plot <- ggplot() + geom_point(aes(x = NNET, y = RF, fill = factor(GNI)), pch = 21, colour = "black", data = m.preds) +
#             geom_text_repel(aes(x = NNET, y = RF, label = label),
#                     data = m.preds[which(m.preds$NNET > 3 & m.preds$RF < 1.5),], size = 1.5) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("NNET predictions (t/pers./year)") + ylab("RF predictions (t/pers./year)") + theme_classic()
#
# ggsave(plot = plot, filename = "plot_predictions_NNETxRF.pdf", dpi = 300, height = 5, width = 7)
#
# plot <- ggplot() + geom_point(aes(x = log(NNET), y = log(RF), fill = factor(GNI)), pch = 21, colour = "black", data = m.preds) +
#             #geom_text_repel(aes(x = NNET, y = RF, label = label),
#                     #data = m.preds[which(m.preds$NNET > 3 & m.preds$RF < 1.5),], size = 1.5) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("NNET predictions log(t/pers./year)") + ylab("RF predictions log(t/pers./year)") + theme_classic()
#
# ggsave(plot = plot, filename = "plot_predictions_NNETxRF_log.pdf", dpi = 300, height = 5, width = 7)
#
#
#
# summary(lm(NNET ~ RF, data = m.preds)) # Adjusted R-squared: 0.0003145 - n.s

### ANd correaltion?
# cor(na.omit(m.preds$NNET), na.omit(m.preds$RF), method = "spearman") # 0.83

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------