
##### 30/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train full GLM models per GNI class by using all potential PV set based on Gaussian-type models
#   - Predict annual MSW per country (can't be 1000 full models since GLM - one model per )
#   (- Compare to RF/GAM.NNET outputs in Script#6.1)

### Last update: 30/01/23

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
library("gbm")

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
m.missing <- m.missing[,-c(length(m.missing))]

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
# OK

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

### Subset based on results from Script#5.1.
list.preds.H2 <- list.preds.H[c(1,3,4,9:11)]
list.preds.UM2 <- list.preds.UM[c(2,3,4,6,7,9,11)]
list.preds.LM2 <- list.preds.LM[c(1:5,7,9,10,12,14:16)]
list.preds.L2 <- list.preds.L[c(1,4:6)]


### Write master function to predict MSW per country and per year
# inc <- "H"

glm.predicter <- function(inc) {
        
        if(inc == "H") {

            dat <- scaled.H
            list.pred <- list.preds.H
            l <- length(list.pred)

        } else if(inc == "UM") {
    
            dat <- scaled.UM
            list.pred <- list.preds.UM
            l <- length(list.pred)

        } else if(inc == "LM") {
    
            dat <- scaled.LM
            list.pred <- list.preds.LM
            l <- length(list.pred)

        } else {

            dat <- scaled.L
            list.pred <- list.preds.L
            l <- length(list.pred)

        } # eo if else loop
        
        require("parallel")
        # i <- 1
        mclapply(c(1:l), function(i) {
            
                # Get vector of pred names based on i
                preds <- list.pred[[i]]
                form <- as.formula(paste('y', paste(preds, collapse = " + "), sep = " ~ "))
                
                # Useless message
                message(paste("Running GLMs for GNI-",inc," based on ",paste(preds, collapse = '+'),sep = ""))
                message("   ")
                
                glm.mod <- glm(formula = form, data = dat, family = "gaussian")

                # Predict SR fpr the testing set and derive RMSE
                fit <- predict(object = glm.mod, newdata = dat[,preds])
                            
                # Extract predictions and errors - save in proper dir
                # Compute MSE and R2
                predict.unscaled <- exp((fit * (max[1] - min[1])) + min[1])
                # Prevent negative values
                if( length(predict.unscaled[which(predict.unscaled < 0)]) >= 1 ){
                        predict.unscaled[which(predict.unscaled < 0)] <- 0    
                }
                            
                # Get response var
                MSW.obs <- exp((dat$y * (max[1] - min[1])) + min[1])
                # Calculate Mean Square Error (MSE)
                MSE <- (sum((MSW.obs - predict.unscaled)^2) / nrow(dat))
                # Compute r2
                R2 <- summary(lm(predict.unscaled ~ MSW.obs))$r.squared
                # r2 <- rsq(x = MSW.obs, y = predict.unscaled) # same output so good
                            
                # Return the table with all the info and results
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_models_training")
                skillz <- data.frame(GNI = inc, n = n, formula = paste(preds, collapse = "+"), R2 = R2, MSE = MSE)
                save(x = glm.mod, file = paste("gbm.full_",inc,"_",paste(preds, collapse = '+'),"_",n,".Rdata", sep = "") )
                save(x = skillz, file = paste("table_skills_gbm.full_",inc,"_",paste(preds, collapse = '+'),"_",n,".Rdata", sep = "") )
                               
                ### Make prediction and compute error to original data 
                all.countries <- c( unique(as.character( m.missing[m.missing$GNI == inc,"country"] )) , as.character(unique(dat$country)) )
                pred <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                pred[,1] <- unique(all.countries)
                colnames(pred) <- c("Country", 1990:2019)
                # Same for errors
                error_country_perc <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                error_country_perc[,1] <- unique(all.countries)
                colnames(error_country_perc) <- c("Country", 1990:2019)
                          
                # land <- "Portugal"
                for( land in all.countries ) { 
                                    
                    message( paste("Saving predictions and errors for ", land, sep = "") )
                                    
                    if( land %in% unique(dat$country) ) {
                                        
                        # get the corresponding j index from country.details
                        j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                        # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                        dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                        names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                        dat.temp <- dat.temp[,c(preds,"country")]
                                        
                        scaled.temp <- as.data.frame(scale(dat.temp[,c(1:length(preds))], center = min[preds], scale = max[preds] - min[preds]))
                        scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                        scaled.temp$country <- dat.temp$country
                        # Predict y for each year for country y                                    
                        fit.2 <- predict(object = glm.mod, newdata = scaled.temp[,preds])
                        # Extract predictions and errors - save in proper dir
                        predict.unscaled <- exp((fit.2 * (max[1] - min[1])) + min[1])                      
                        # If ever negative values, convert to zero
                        predict.unscaled[which(predict.unscaled < 0)] <- 0
                        names(predict.unscaled) <- rownames(scaled.temp)
                        # Find the position (Years between 1990:2019) to fill in the empty error dataset
                        error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)   
                        # Compute and supply error 
                        er <- ( (predict.unscaled[error.pos] - y_org[error.pos,j])/y_org[error.pos,j] )*100
                        error_country_perc[error_country_perc$Country == land,c(error.pos)] <- er
                        # And supply prediction to pred matrix
                        pred[pred$Country == land,-1] <- predict.unscaled
                                        
                    } else if ( land %in% unique(m.missing$country) ) {
                                        
                        j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                        dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                        names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                        dat.temp <- dat.temp[,c(preds,"country")]
                        scaled.temp <- as.data.frame(scale(dat.temp[,c(1:length(preds))], center = min[preds], scale = max[preds] - min[preds]))
                        scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                        scaled.temp$country <- dat.temp$country
                        # Predict y for each year for country y
                        fit.2 <- predict(object = glm.mod, newdata = scaled.temp[,preds])
                        predicted <- exp( (fit.2 * (max[1] - min[1])) + min[1] ) 
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
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_predictions")
                write.table(x = error_country_perc, file = paste("table_error_perc_",inc,"_",paste(preds, collapse = '+'),"_",n,".txt", sep = ""), sep = "\t")
                write.table(x = pred, file = paste("table_pred_",inc,"_",paste(preds, collapse = '+'),"_",n,".txt", sep = ""), sep = "\t")   
                                
            } # eo fun of the mclapply
            
            , mc.cores = l
            
        ) # eo mclapply - i in length(preds)    
    
} # eo FUN - glm.predicter

# Run for each GNI class
glm.predicter("H")
glm.predicter("UM")
glm.predicter("LM")
glm.predicter("L")

# Check number of file (should be ~8000)
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_models_training")
# length(dir())
# setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_predictions")
# length(dir())
# All good

### Check the skills of the trained models
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_models_training")
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
ddf <- bind_rows(res)
dim(ddf); summary(ddf)

### R2
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
plot <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "R2", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("R2") + theme_bw()
ggsave(plot = plot, filename = "boxplot_full.glms_R2_GNI_30_01_23.pdf", dpi = 300, height = 5, width = 4)

### MSE
plot <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("MSE") + theme_bw()          
ggsave(plot = plot, filename = "boxplot_full.glms_MSE_GNI_30_01_23.pdf", dpi = 300, height = 5, width = 4)


### And check % error for each class/ country?
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_predictions")
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
summary(m.ddf$value) # from -92 to +506% & median error is 2.5%? Does not look good. 
sum.error <- data.frame(m.ddf %>% group_by(Country) %>% summarize(median = median(value)) ) # eo ddf
# summary(sum.error) 

### Compute mean annual ensemble predictions for GBMs. Re-compute error of the ensemble predictions to the y_org. 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_predictions")
# Load all predictions and compute mean/median annual MSW per country
files <- dir()[grep(paste("table_pred_", sep = ""),dir())] # length(files)
res <- lapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) } ) # eo lapply
# Rbind
ddf <- bind_rows(res)
# dim(ddf); head(ddf) 
colnames(ddf)[c(2:length(ddf))] <- as.character(c(1990:2019))
# Melt to put years as vector
m.ddf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
colnames(m.ddf)[c(2:3)] <- c("Year","MSW")
# summary(m.ddf$MSW)
# Compute median 
ensemble.med <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(MSW = median(MSW)) ) # eo ddf
summary(ensemble.med)
# Dcast
ensemble <- dcast(ensemble.med, Country ~ Year)
dim(ensemble) # 217x31 - good
summary(ensemble)

### Ok, save ensemble predictions of MSW and apply the same prediction fun above to RF
write.table(x = ensemble, "table_ensemble_predictions_GLM_30_01_23.txt", sep = "\t")


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
# dim(error_ensemble) # good
### Save error table
write.table(error_ensemble, file = "table_GLM_median_errors_percentage_30_01_23.txt", sep = ";")

### And derive pluriannual mean of errors
m.error <- melt(error_ensemble, id.vars = "Country")
# summary(m.error$value) # from -90% to +874%, and median error is -1.5%
sum.error.ensembles <- data.frame(m.error %>% group_by(Country) %>% summarize(median = median(value, na.rm = T)) ) # eo ddf
sum.error.ensembles[order(sum.error.ensembles$median, decreasing = T),]
### Does not look good, like GBMs

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------

