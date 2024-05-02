
##### 26/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the overarching project is to model and predict waste production per country and per year (1990-2015) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using random forest models (RF), but also neural networks (NNET) if robust enough, or GAMs.
##### Ultimately, you need to provide to C. Laufkötter a long table (countriesxyear) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train and optimize GLMs models (Generalized Linear Models) based on the same predictors set as GAMs/RF/NNET/GBMs
#   - Need to find the optimal PVs sets and GLM parameters (family) for each GNI class by running 1000 GLMs models - boxplots

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

### A°) Setting up data and writing the master function to test parameters of various GBMs (10 models per combinations)

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

### Range of parameters to test in GLMs:
# ?glm

### Write function to test this grid of parameters for each GNI class and list of predictors - return R2 and MSE/RMSE
# inc <- "H" # for testing function below

# vector of families of GLMs to run
families <- c("gaussian","poisson","quasi","quasibinomial","quasipoisson")

glm.optimizer <- function(inc) {

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
        # i <- 4
        res <- mclapply(c(1:l), function(i) {
            
                # Get vector of pred names based on i
                preds <- list.pred[[i]]
                form <- as.formula(paste('y', paste(preds, collapse = " + "), sep = " ~ "))
            
                ### Use another lapply,to tets all parameters 
                # fam <- families[3]
                res.params <- lapply(families, function(fam) {                    
                        
                            # Useless message
                            message(paste("Testing 10 GLMs of family ",fam," for GNI-",inc," with ",paste(preds, collapse = '+'),sep = ""))
                            message("   ")
                            
                            skillz <- data.frame(GNI = inc, Nmodel = c(1:10), formula = paste(preds, collapse = "+"), family = fam, R2 = NA, MSE = NA, rho = NA)
                            
                            # table.res <- GBM_params[p,]
                            # table.res <- table.res %>% slice(rep(row_number(), 10))
                            # table.res$N_model <- c(1:N_models)
                             
                            for(n in c(1:10)) {
                                
                                # Split the global susbet into XX% training set and XX% testing set randomly
                                nobs <- nrow(dat)
                                ind <- sample(c(T,F), size = nobs, replace = T, prob = c(0.9,0.1))
                                train <- dat[ind,]
                                test <- dat[!ind,]
                            
                                # Run GLM
                                glm.test <- glm(formula = form, data = train, family = fam)
                                # summary(glm.test)
                            
                                # Predict SR fpr the testing set and derive RMSE
                                test$fit <- predict(object = glm.test, newdata = test[,preds])

                                # Compute MSE & R2
                                mse <- (sum((test$y - test$fit)^2) / nrow(test))
                                cor.spear <- cor(test$fit, test$y, method = "spearman")
                                rsq <- function (x,y) cor(x,y)^2
                                r2 <- rsq(x = test$y, y = test$fit)
                        
                                # Fill 'skillz' table
                                skillz[skillz$Nmodel == n,"MSE"] <- mse
                                skillz[skillz$Nmodel == n,"R2"] <- r2
                                skillz[skillz$Nmodel == n,"rho"] <- cor.spear
                        
                            } # eo for loop
                        
                            # Return the table with all the info and results
                            return(skillz)
                        
                        } # eo FUN
                        
                ) # eo lapply - res.params
                # Rbind
                table.params <- bind_rows(res.params)
                rm(res.params) ; gc()    
                # dim(table.params); str(table.params); summary(table.params)   
                
                return(table.params)
            
                } # eo fun of the mclapply
            
            , mc.cores = l
            
        ) # eo mclapply - i in length(preds)    
    
} # eo FUN - glm.optimizer

# Run for each GNI class
table.opti.H <- bind_rows( glm.optimizer("H") )
table.opti.UM <- bind_rows( glm.optimizer("UM") )
table.opti.LM <- bind_rows( glm.optimizer("LM") ) # tends to fail
table.opti.L <- bind_rows( glm.optimizer("L") )
# Make sure they all have same dimensions and rbind
# dim(table.opti.H) ; dim(table.opti.UM) ; dim(table.opti.LM) ; dim(table.opti.L) # OK, rbind
table.all <- rbind(table.opti.H, table.opti.UM, table.opti.LM, table.opti.L)
# dim(table.all)
# summary(table.all)

### Save output 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GLM_optimization_per_GNI")
save(x = table.all, file = "table_parameters_tests_GLM_v3_27_01_23.RData")


### ------------------------------------------------------------------------------------------------------------------------------------------------

### B°) Examine outputs (MSE and R2 per GNI and parameters)
library("ggpubr")
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots") 

# colnames(table.all)
#groups <- data.frame(table.all %>% group_by(GNI) %>% summarize(MSE = median(MSE, na.rm = T)))
#groups[order(groups$MSE, decreasing = T),]
#groups <- data.frame(table.all %>% group_by(GNI) %>% summarize(R2 = median(R2, na.rm = T)))
#groups[order(groups$R2, decreasing = T),]

p1 <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("GNI") + ylab("MSE") + theme_bw() 

p2 <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("GNI") + ylab("R2") + theme_bw() 

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GLM_tests_R2_MSE_GNI_27_01_23.pdf", dpi = 300, height = 4, width = 7)
    
### Early conclusions: Large variations in scores, but UM seems like harder to model than the other classes

### Look at variations across preds x GNI (does PVs drive the patterns?)
p1 <- ggplot(aes(x = factor(formula), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("MSE") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    
p2 <- ggplot(aes(x = factor(formula), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("R2") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GLM_tests_R2_MSE_PVs_GNI_27_01_23.pdf", dpi = 300, height = 10, width = 15)


### Examine R2 and MSE per GNI x families
p1 <- ggplot(aes(x = factor(family), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Family") + ylab("MSE") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    
p2 <- ggplot(aes(x = factor(family), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Family") + ylab("R2") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GLM_tests_R2_MSE_families_GNI_27_01_23.pdf", dpi = 300, height = 10, width = 15)


### Conclusions: choose some PVs set as clear variability in R2 and pick Gaussian GLMs


### New PV sets to select: 
# For H: exclude p1,p3,p5,p6  p1,p4,p5,p6   p1,p2,p5,p6   p2,p3,p5,p6    p1,p2,p3,p6
# For UM: exclude p1,p2,p3,p5,p6        p1,p2,p3,p5       p2,p3,p5,p6     p1,p2,p5,p6
# For LM: exclude p1,p3,p6      p1,p4,p6        p2,p3,p6        p2,p4,p6
# For L: exclude p1,p2,p6       p2,p3,p6        p4,p5,p6
list.preds.H2 <- list.preds.H[c(1,3,4,9:11)]
list.preds.UM2 <- list.preds.UM[c(2,3,4,6,7,9,11)]
list.preds.LM2 <- list.preds.LM[c(1:5,7,9,10,12,14:16)]
list.preds.L2 <- list.preds.L[c(1,4:6)]

### Based on this new sets of PVs and the selected family (Gaussian), train 1'000 GLMs per GNI and assess R2 and MSE to obs. Compare to GAM and RF (Script 6.1).
### Make new master function.

# inc <- "H"
glm.tester <- function(inc) {
    
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
        # i <- 4
        res <- mclapply(c(1:l), function(i) {
            
                # Get vector of pred names based on i
                preds <- list.pred[[i]]
                form <- as.formula(paste('y', paste(preds, collapse = " + "), sep = " ~ "))
            
                # Define the number of n times to run the PV set 
                N <- round(1000/l, 0.1) 
                
                # Useless message
                message(paste("Running ",N," GLMs for GNI-",inc," based on ",paste(preds, collapse = '+'),sep = ""))
                message("   ")
                 
                ### Use another lapply,to tets all parameters 
                # fam <- families[3]
                res.params <- lapply(c(1:N), function(n) {                    
                            
                            skillz <- data.frame(GNI = inc, formula = paste(preds, collapse = "+"), R2 = NA, MSE = NA, rho = NA)
                                       
                            # Split the global susbet into XX% training set and XX% testing set randomly
                            nobs <- nrow(dat)
                            ind <- sample(c(T,F), size = nobs, replace = T, prob = c(0.9,0.1))
                            train <- dat[ind,]
                            test <- dat[!ind,]
                            
                            # Run GLM
                            glm.test <- glm(formula = form, data = train, family = "gaussian")
                            
                            # Predict SR fpr the testing set and derive RMSE
                            test$fit <- predict(object = glm.test, newdata = test[,preds])

                            # Compute MSE & R2
                            mse <- (sum((test$y - test$fit)^2) / nrow(test))
                            cor.spear <- cor(test$fit, test$y, method = "spearman")
                            rsq <- function (x,y) cor(x,y)^2
                            r2 <- rsq(x = test$y, y = test$fit)
                        
                            # Fill 'skillz' table
                            skillz$Nmodel <- n
                            skillz[skillz$Nmodel == n,"MSE"] <- mse
                            skillz[skillz$Nmodel == n,"R2"] <- r2
                            skillz[skillz$Nmodel == n,"rho"] <- cor.spear
                                                
                            # Return the table with all the info and results
                            return(skillz)
                        
                        } # eo FUN
                        
                ) # eo lapply - res.params
                # Rbind
                table.params <- bind_rows(res.params)
                rm(res.params) ; gc()    
                # dim(table.params); str(table.params); summary(table.params)   
                
                return(table.params)
            
                } # eo fun of the mclapply
            
            , mc.cores = l
            
        ) # eo mclapply - i in length(preds)    
    
} # eo FUN - glm.tester

# Run for each GNI class
table.opti.H <- bind_rows( glm.tester("H") )
table.opti.UM <- bind_rows( glm.tester("UM") )
table.opti.LM <- bind_rows( glm.tester("LM") ) 
table.opti.L <- bind_rows( glm.tester("L") )
table.all <- rbind(table.opti.H, table.opti.UM, table.opti.LM, table.opti.L)
summary(table.all)

### Save output 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GLM_optimization_per_GNI")
save(x = table.all, file = "table_parameters_tests_GLM_models_30_01_23.RData")

### 30/01/23: Check outputs of models 
library("ggpubr")
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots") 
colnames(table.all)

p1 <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("MSE") + theme_bw() 
    
p2 <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("R2") + theme_bw() 

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GLM_R2_MSE_30_01_23.pdf", dpi = 300, height = 4, width = 7)


### Compare to R2 and MSE of 'full' RF and GAM models (couldn't run 1000 GLMs)
table.all.glm <- table.all

# Get the GAMs' scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_models_training")
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.gam <- bind_rows(res)
rm(res); gc()

# Get the RF's scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training")
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.rf <- bind_rows(res)
rm(res); gc()

# Get the GBM scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GBM_models_training")
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.gbm <- bind_rows(res)
rm(res); gc()

# Get the NNET scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/NNET_models_training")
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.nnet <- bind_rows(res)
rm(res); gc()


### Rbind by common colnames
# colnames(table.all.rf) ; colnames(table.all.gam) ; colnames(table.all.gbm)
table.all.glm$Model <- "GLM"
table.all.gam$Model <- "GAM"
table.all.rf$Model <- "RF"
table.all.gbm$Model <- "GBM"
table.all.nnet$Model <- "NNET"
commons <- c("Model","GNI","formula","MSE","R2")
table <- rbind(table.all.glm[,commons], table.all.gam[,commons], table.all.rf[,commons], table.all.gbm[,commons], table.all.nnet[,commons])
dim(table); head(table)

### Plot MSE and R2 per Modle and per GNI 
p1 <- ggplot(aes(x = factor(Model), y = MSE, fill = factor(GNI)), data = table) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Model type") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
# Same, but with R2
p2 <- ggplot(aes(x = factor(Model), y = R2, fill = factor(GNI)), data = table) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Model type") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
    
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots") 
panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GLMxGAMxRFxGBMxNNET_R2_MSE_30_01_23.pdf", dpi = 300, height = 8, width = 10)

### Conclusion: GLMs show the lowest R2 but also the lowest MSE - MSE is weirdly extremely low

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------