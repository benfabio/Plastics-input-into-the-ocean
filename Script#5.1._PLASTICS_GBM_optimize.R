
##### 26/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the overarching project is to model and predict waste production per country and per year (1990-2015) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using random forest models (RF), but also neural networks (NNET) if robust enough, or GAMs.
##### Ultimately, you need to provide to C. Laufkötter a long table (countriesxyear) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train and optimize BRT models (Gradient Boosting Machines, GBM) based on the same predictors set as GAMs and RF
#   - Needs to find the optimal GBM architecture for each GNI class by running 1000 BRT models - boxplots

### Last update: 27/01/23

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

### Range of parameters to test in GBMs:
# n.minobsinnode (integer specifying the minimum number of observations in the terminal nodes of the trees) --> 5,10,25,50,100 (has to vary per class)
# interaction.depth (Integer specifying the maximum depth of each tree (i.e. the highest level of variable interactions allowed. value == 1 implies an additive model) --> 1,2,3
# shrinkage (learning rate or step-size reduction): usually between 0.01 and 0.1 --> 0.01 & 0.1
# n.trees (This is equivalent to the number of iterations and the number of basis functions in the additive expansion) --> 25, 50, 75, 100, 200
### according to LG: interaction.depth & n.trees are the 2 main parameters controlling model complexity and overfitting


### Write function to test this grid of parameters for each GNI class and list of predictors - return R2 and MSE/RMSE
# inc <- "LM" # for testing function below

gbm.optimizer <- function(inc) {
    
        # Choose range of features to explore as a function of GNI class. Some features must be lower for L and LM income class because of the lower number of points available
        if(inc == "H") {
        
            dat <- scaled.H
            list.pred <- list.preds.H
            l <- length(list.pred)
            # Define parameters grid
            min_obs_node <- c(5,10,25,50)
            # inter_depth <- c(1:3)
            inter_depth <- c(1:3,5,7,10) # v3
            shrinkage <- c(0.01,0.1)
            #Ntree_gbm <- c(25,50,75,100,200,500) # v1
            #Ntree_gbm <- c(2,3,5,10,25,50,75,100) # v2
            Ntree_gbm <- c(2,3,5,10,25,50) # v3
            N_models <- 10 # 10 random CV runs
            GBM_params <- expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm)
            colnames(GBM_params) <- c("min_obs_node","inter_depth","shrinkage","Ntree_gbm")
            GBM_params$combination <- apply(expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm), 1, paste, collapse = "_")
            # Set OOB fraction (the fraction of the training set observations randomly selected to propose the next tree in the expansion. This
            # introduces randomnesses into the model fit. If ‘bag.fraction’ < 1 then running the same model twice will result in similar
            # but different fits.)
            bag <- 0.5 # enough data in H to do this
        
        } else if(inc == "UM") {
            
            dat <- scaled.UM
            list.pred <- list.preds.UM
            l <- length(list.pred)
            # Define parameters grid
            min_obs_node <- c(5,10,25,50)
            # inter_depth <- c(1:3)
            inter_depth <- c(1:3,5,7,10) # v3
            shrinkage <- c(0.01,0.1)
            #Ntree_gbm <- c(25,50,75,100,200) # v1
            #Ntree_gbm <- c(2,3,5,10,25,50,75,100) # v2
            Ntree_gbm <- c(2,3,5,10,25,50) # v3
            N_models <- 10 # 10 random CV runs
            GBM_params <- expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm)
            colnames(GBM_params) <- c("min_obs_node","inter_depth","shrinkage","Ntree_gbm")
            GBM_params$combination <- apply(expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm), 1, paste, collapse = "_")
            # Set OOB fraction (the fraction of the training set observations randomly selected to propose the next tree in the expansion. This
            # introduces randomnesses into the model fit. If ‘bag.fraction’ < 1 then running the same model twice will result in similar
            # but different fits.)
            bag <- 0.5 # enough data in UM to do this
        
        } else if(inc == "LM") {
            
            dat <- scaled.LM
            list.pred <- list.preds.LM
            l <- length(list.pred)
            # Define parameters grid
            min_obs_node <- c(3,5,10,25)
            # inter_depth <- c(1:3)
            inter_depth <- c(1:3,5,7,10) # v3
            shrinkage <- c(0.01,0.1)
            #Ntree_gbm <- c(25,50,75,100) # v1
            #Ntree_gbm <- c(2,3,5,10,25,50,75,100) # v2
            Ntree_gbm <- c(2,3,5,10,25,50) # v3#
            N_models <- 10 # 10 random CV runs
            GBM_params <- expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm)
            colnames(GBM_params) <- c("min_obs_node","inter_depth","shrinkage","Ntree_gbm")
            GBM_params$combination <- apply(expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm), 1, paste, collapse = "_")
            # Set OOB fraction (the fraction of the training set observations randomly selected to propose the next tree in the expansion. This
            # introduces randomnesses into the model fit. If ‘bag.fraction’ < 1 then running the same model twice will result in similar
            # but different fits.)
            bag <- 0.5 # enough data in LM to do this
       
        } else {
        
            dat <- scaled.L
            list.pred <- list.preds.L
            l <- length(list.pred)
            # Define parameters grid
            min_obs_node <- c(3,5,10)
            # inter_depth <- c(1:3)
            inter_depth <- c(1:3,5,7,10) # v3
            shrinkage <- c(0.01,0.1)
            #Ntree_gbm <- c(25,50,75,100) # v1
            #Ntree_gbm <- c(2,3,5,10,25,50,75,100) # v2
            Ntree_gbm <- c(2,3,5,10,25,50) # v3
            N_models <- 10 # 10 random CV runs
            GBM_params <- expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm)
            colnames(GBM_params) <- c("min_obs_node","inter_depth","shrinkage","Ntree_gbm")
            GBM_params$combination <- apply(expand.grid(min_obs_node, inter_depth, shrinkage, Ntree_gbm), 1, paste, collapse = "_")
            # Set OOB fraction (the fraction of the training set observations randomly selected to propose the next tree in the expansion. This
            # introduces randomnesses into the model fit. If ‘bag.fraction’ < 1 then running the same model twice will result in similar
            # but different fits.)
            bag <- 0.8 # Needs to be higher because smaller training set than for the other 3 GNI classes
        
        } # eo if else loop
        
        require("parallel")
        # i <- 4
        res <- mclapply(c(1:l), function(i) {
            
                # Get vector of pred names based on i
                preds <- list.pred[[i]]
                form <- as.formula(paste('y', paste(preds, collapse = " + "), sep = " ~ "))
            
                ### Use another lapply,to tets all parameters 
                # p <- 1
                res.params <- lapply(c(1:nrow(GBM_params)), function(p) {                    
                         
                        table.res <- GBM_params[p,]
                        table.res <- table.res %>% slice(rep(row_number(), 10))
                        table.res$N_model <- c(1:N_models)
                        
                        ### Run 10 CV runs
                        # n <- 3
                        for(n in c(1:N_models)) {
                        
                            # Useless message
                            message(paste("Testing GMs 10 GBMs for GNI-",inc," with ",paste(preds, collapse = '+'),sep = ""))
                            message("   ")
                            
                            # Split the global susbet into XX% training set and XX% testing set randomly
                            nobs <- nrow(dat)
                            ind <- sample(c(T,F), size = nobs, replace = T, prob = c(0.85,0.15))
                            train <- dat[ind,]
                            test <- dat[!ind,]
                            
                            # Perform GBM
                            message(paste("Running GBM based on parameters from row ", p, sep = ""))
                         
                            gbm.test <- gbm(formula = form, data = train, distribution = "gaussian",
                                    n.minobsinnode = GBM_params[p,"min_obs_node"], interaction.depth = GBM_params[p,"inter_depth"],
                                    n.trees = GBM_params[p,"Ntree_gbm"], shrinkage = GBM_params[p,"shrinkage"], bag.fraction = bag,
                                    cv.folds = 0, verbose = F, n.cores = 1)

                            # Predict SR fpr the testing set and derive RMSE
                            test$fit <- predict(object = gbm.test, newdata = test[,preds], n.trees = GBM_params[i,"Ntree_gbm"])

                            # Compute MSE & R2
                            mse <- (sum((test$y - test$fit)^2) / nrow(test))
                            # rmse <- sqrt(mse)
                            # Compute corr coeff between preds and obs
                            cor.spear <- cor(test$fit, test$y, method = "spearman")
                            rsq <- function (x,y) cor(x,y)^2
                            r2 <- rsq(x = test$y, y = test$fit)
                        
                            # Return the table with all the info and results
                            table.res[table.res$N_model == n,"R2"] <- r2
                            table.res[table.res$N_model == n,"MSE"] <- mse
                            table.res[table.res$N_model == n,"rho"] <- cor.spear
                            
                        } # eo for loop - n in c(1:N_models)
                        
                        return(table.res)
                    
                    } # eo FUN - i in nrows(GBM_params)
                    
                ) # eo lapply - res.params
                # Rbind
                table.params <- bind_rows(res.params)
                rm(res.params) ; gc()    
                # dim(table.params); str(table.params); summary(table.params)   
                
                # Inform PVs used
                table.params$preds <- paste(preds, collapse = "+")
                table.params$GNI <- inc
                
                return(table.params)
            
                } # eo fun of the mclapply
            
            , mc.cores = l
            
        ) # eo mclapply - i in length(preds)    
    
} # eo FUN - gbm.optimizer

# Run for each GNI class
table.opti.H <- bind_rows( gbm.optimizer("H") )
table.opti.UM <- bind_rows( gbm.optimizer("UM") )
table.opti.LM <- bind_rows( gbm.optimizer("LM") ) # tends to fail
table.opti.L <- bind_rows( gbm.optimizer("L") )
# Make sure they all have same dimensions and rbind
# dim(table.opti.H) ; dim(table.opti.UM) ; dim(table.opti.LM) ; dim(table.opti.L) # OK, rbind
table.all <- rbind(table.opti.H, table.opti.UM, table.opti.LM, table.opti.L)
# dim(table.all)
# str(table.all)
# summary(table.all)

### Save output 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GBM_optimization_per_GNI")
save(x = table.all, file = "table_parameters_tests_GBM_v3_27_01_23.RData")

### 26/01/23: Note: 'v2' for Ntrees ranging from 2 to 100
### 27/01/23: Note: 'v3' for Ntrees ranging from 2 to 50 and interaction.depth varying from c(1,2,3,5,7,10)


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
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_GNI_26_01_23.pdf", dpi = 300, height = 4, width = 7)
    
### Early conclusions: Large variations in scores, but UM seems like harder to model than the other classes

### Look at variations across preds x GNI (does PVs drive the patterns?)
p1 <- ggplot(aes(x = factor(preds), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("MSE") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    
p2 <- ggplot(aes(x = factor(preds), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") +
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("") + ylab("R2") + theme_bw() + facet_wrap(~factor(table.all$GNI), ncol = 2, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_PVs_GNI_26_01_23.pdf", dpi = 300, height = 10, width = 15)

### Some PV sets seem to lead to lower skill (p1+p2+p3+p6 for H, or p1+p3+p6 for LM for instance). But unsure about the interaction with GBM parameters yet. Could be due to certain combinations of parameters to avoid. Also, seems only to have an impact for LM, and not L, H or UM


### B.1) MSE across min_obs_node
# ggplot(aes(x = factor(min_obs_node), y = MSE), data = table.all) +
#     geom_boxplot(fill = "grey65", colour = "black") +
#     xlab("min_obs_node") + ylab("MSE") + theme_bw()
# + Facet per group
p1 <- ggplot(aes(x = factor(min_obs_node), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("min_obs_node") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
# Same, but with R2
p2 <- ggplot(aes(x = factor(min_obs_node), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("min_obs_node") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_min_obs_node_GNI_v3_27_01_23.pdf", dpi = 300, height = 10, width = 15)

### Conclusions: 
# H: min_obs_node can span between 5 and 25
# UM: min_obs_node should span between 5 and 25
# LM: min_obs_node should span between 3 and 10
# L: min_obs_node should span between 3 and 5

#groups <- data.frame(table.all %>% group_by(GNI,min_obs_node) %>% summarize(MSE = mean(MSE, na.rm = T), sd = sd(MSE, na.rm = T)))
#groups[order(groups$MSE, decreasing = T),]
#groups <- data.frame(table.all %>% group_by(GNI,min_obs_node) %>% summarize(R2 = mean(R2, na.rm = T), sd = sd(R2, na.rm = T)))
#groups[order(groups$R2, decreasing = T),]


### B.2) MSE across inter_depth
# ggplot(aes(x = factor(inter_depth), y = MSE), data = table.all) +
#     geom_boxplot(fill = "grey65", colour = "black") +
#     xlab("inter_depth") + ylab("MSE") + theme_bw()
#
p1 <- ggplot(aes(x = factor(inter_depth), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("inter_depth") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
# Same, but with R2
p2 <- ggplot(aes(x = factor(inter_depth), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("inter_depth") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_inter_depth_GNI_v3_27_01_23.pdf", dpi = 300, height = 10, width = 15)

### Conclusions: 
# H: 3 better than 1
# UM: 3 better than 1
# LM: no impact - keep 1
# L: no impact - keep 1


### B.3) MSE across shrinkage
# ggplot(aes(x = factor(shrinkage), y = MSE), data = table.all) +
#     geom_boxplot(fill = "grey65", colour = "black") +
#     xlab("shrinkage") + ylab("MSE") + theme_bw()

p1 <- ggplot(aes(x = factor(shrinkage), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("shrinkage") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
# Same, but with R2
p2 <- ggplot(aes(x = factor(shrinkage), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("shrinkage") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_shrinkage_GNI_v3_27_01_23.pdf", dpi = 300, height = 8, width = 6)

### Conclusions: shrinkage = 0.1 for all !


### B.4) MSE across Ntree
# ggplot(aes(x = factor(Ntree_gbm), y = MSE), data = table.all) +
#     geom_boxplot(fill = "grey65", colour = "black") +
#     xlab("N trees") + ylab("MSE") + theme_bw()

p1 <- ggplot(aes(x = factor(Ntree_gbm), y = MSE, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Ntree") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")
# Same, but with R2
p2 <- ggplot(aes(x = factor(Ntree_gbm), y = R2, fill = factor(GNI)), data = table.all) +
    geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Ntree") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(GNI), scales = "free")

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_Ntree_GNI_v3_27_01_23.pdf", dpi = 300, height = 10, width = 15)


### Assess interactions between Ntree_gbm and inter_depth?
p1 <- ggplot(aes(x = factor(Ntree_gbm), y = MSE), data = table.all) +
    geom_boxplot(colour = "black", fill = "grey75") + 
    #scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Ntree") + ylab("MSE") + theme_bw() +
    facet_wrap(~ factor(inter_depth), scales = "free")
    
p2 <- ggplot(aes(x = factor(Ntree_gbm), y = R2), data = table.all) +
    geom_boxplot(colour = "black", fill = "grey75") + 
    #scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    xlab("Ntree") + ylab("R2") + theme_bw() +
    facet_wrap(~ factor(inter_depth), scales = "free")

panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "hv", common.legend = T)
ggsave(panel, filename = "boxplots_GBM_tests_R2_MSE_Ntreexinter.depth_v3_27_01_23.pdf", dpi = 300, height = 10, width = 15)


### Conclusions: no impact of Ntree on scores! Re-run master function above with same Ntree range for all GNI ( c(10,25,50,75,100,200,300,500) ) 
### --> No change - choose ntree = 201 for all (after consulting Luke Gregor)
### No interactions between N trees and interaction depth


### So far actually, the most impacting factor was the PV set, especially except for UM, who has lower R2 and very low MSE

### FINAL conclusions: 
# shrinkage = 0.1
# ntree = 201
# inter_depth = 5 for H and UM, 2 for LM and L
# min_obs_node: 5-25 for H and UM, 3-10 for LM, 5 or 3-5

### New PV sets to select: 
# For H: exclude p1,p2,p3,p6 and p1,p2,p4,p6
# For UM: keep all as before 
# For LM: exclude p1,p3,p6 ; p1,p4,p6 ; p2,p3,p6 and p2,p4,p6
# For L: exclude p1,p2,p6 and p2,p3,p6
list.preds.H2 <- list.preds.H[c(1,3,5:11)]
list.preds.UM2 <- list.preds.UM
list.preds.LM2 <- list.preds.LM[c(1:5,7,9,10,12,14:16)]
list.preds.L2 <- list.preds.L[c(1,4:7)]

### Based on this new sets of PVs and the selected features, train 'full' 1000 GBMs per GNI and assess R2 and MSE to obs. Compare to GAM and RF (Script 6.1).
### Make new master fucntion.

# inc <- "H"
gbm.tester <- function(inc) {
    
        # Choose range of features to explore as a function of GNI class. Some features must be lower for L and LM income class because of the lower number of points available
        if(inc == "H") {
        
            dat <- scaled.H
            list.pred <- list.preds.H2
            l <- length(list.pred)
            min_obs_node <- c(5,10,25)
            inter_depth <- 5
            shrinkage <- 0.1
            Ntree_gbm <- 99
            bag <- 0.5 
        
        } else if(inc == "UM") {
            
            dat <- scaled.UM
            list.pred <- list.preds.UM2
            l <- length(list.pred)
            # Define sets of parameters to choose from, based on tests above 
            min_obs_node <- c(5,10,25)
            inter_depth <- 5
            shrinkage <- 0.1
            Ntree_gbm <- 99
            bag <- 0.5 
        
        } else if(inc == "LM") {
            
            dat <- scaled.LM
            list.pred <- list.preds.LM2
            l <- length(list.pred)
            # Define sets of parameters to choose from, based on tests above 
            min_obs_node <- c(3,5,10)
            inter_depth <- 2
            shrinkage <- 0.1
            Ntree_gbm <- 99
            bag <- 0.5 
       
        } else {
        
            dat <- scaled.L
            list.pred <- list.preds.L2
            l <- length(list.pred)
            # Define sets of parameters to choose from, based on tests above 
            min_obs_node <- c(3,5)
            inter_depth <- 2
            shrinkage <- 0.1
            Ntree_gbm <- 99
            bag <- 0.8
        
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
                message(paste("Running ",N," GBMs for GNI-",inc," based on ",paste(preds, collapse = '+'),sep = ""))
                message("   ")
                
                ### Use another lapply,to tets all parameters 
                # n <- 1
                res <- lapply(c(1:N), function(n) {                    
                      
                            gbm.test <- gbm(formula = form, data = dat, distribution = "gaussian",
                                    n.minobsinnode = sample(min_obs_node, 1), interaction.depth = inter_depth,
                                    n.trees = Ntree_gbm, shrinkage = shrinkage, bag.fraction = bag,
                                    cv.folds = 0, verbose = F, n.cores = 1)

                            # Predict SR fpr the testing set and derive RMSE
                            fit <- predict(object = gbm.test, newdata = dat[,preds], n.trees = Ntree_gbm)

                            # Compute MSE & R2
                            mse <- (sum((dat$y - fit)^2) / nrow(dat))
                            # rmse <- sqrt(mse)
                            # Compute corr coeff between preds and obs
                            cor.spear <- cor(fit, dat$y, method = "spearman")
                            rsq <- function (x,y) cor(x,y)^2
                            r2 <- rsq(x = dat$y, y = fit)
                        
                            # Return the table with all the info and results
                            skillz <- data.frame(GNI = inc, n = n, formula = paste(preds, collapse = "+"), R2 = r2, MSE = mse, rho = cor.spear)
                            
                            return(skillz)
                    
                    } # eo FUN - i in nrows(GBM_params)
                    
                ) # eo lapply - res.params
                # Rbind
                table <- bind_rows(res)
                rm(res) ; gc()    
                # dim(table); summary(table)   
        
                return(table)
            
                } # eo fun of the mclapply
            
            , mc.cores = l
            
        ) # eo mclapply - i in length(preds)    
    
} # eo FUN - gbm.tester


# Run for each GNI class
table.opti.H <- bind_rows( gbm.tester("H") )
table.opti.UM <- bind_rows( gbm.tester("UM") )
table.opti.LM <- bind_rows( gbm.tester("LM") ) 
table.opti.L <- bind_rows( gbm.tester("L") )
# dim(table.opti.H) ; dim(table.opti.UM) ; dim(table.opti.LM) ; dim(table.opti.L) # OK, ~1000 each
table.all <- rbind(table.opti.H, table.opti.UM, table.opti.LM, table.opti.L)
summary(table.all)

### Save output 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GBM_optimization_per_GNI")
save(x = table.all, file = "table_parameters_tests_GBM_full_models_27_01_23.RData")

### 27/01/23: Check outputs of full models 
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
ggsave(panel, filename = "boxplots_GBM_full_R2_MSE_27_01_23.pdf", dpi = 300, height = 6, width = 4)


### Compare to R2 and MSE of RF and GAM models 
table.all.gbm <- table.all
# dim(table.all.gbm)

# Get the GAMs' scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_models_training")
# length(dir()[grep("table_skills_gam.full_H",dir())])
# length(dir()[grep("table_skills_gam.full_UM",dir())])
# length(dir()[grep("table_skills_gam.full_LM",dir())])
# length(dir()[grep("table_skills_gam.full_L_",dir())])
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.gam <- bind_rows(res)
dim(table.all.gam); summary(table.all.gam)
rm(res); gc()

# Get the RF's scores
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training")
# length(dir()[grep("table_skills_RF_H_",dir())])
# length(dir()[grep("table_skills_RF_UM_",dir())])
# length(dir()[grep("table_skills_RF_LM_",dir())])
# length(dir()[grep("table_skills_RF_L_",dir())])
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
# Rbind
table.all.rf <- bind_rows(res)
dim(table.all.rf); summary(table.all.rf)
rm(res); gc()

### Rbind by common colnames
# colnames(table.all.rf) ; colnames(table.all.gam) ; colnames(table.all.gbm)
table.all.rf$Model <- "RF"
table.all.gam$Model <- "GAM"
table.all.gbm$Model <- "GBM"
commons <- c("Model","GNI","formula","MSE","R2")
table <- rbind(table.all.rf[,commons], table.all.gam[,commons], table.all.gbm[,commons])
dim(table)

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
ggsave(panel, filename = "boxplots_RFxGAMxGBM_R2_MSE_27_01_23.pdf", dpi = 300, height = 8, width = 8)

### Much lower MSE (like NNET before) and slightly lower R2 (unlike NNET which had much lower R2). GBMs could be kept for the ensemble predictions.
### In Script#5.2, run full models to predict MSWc and check errors to original obs - compare predictions and errors to RF and GAM again then. 


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------