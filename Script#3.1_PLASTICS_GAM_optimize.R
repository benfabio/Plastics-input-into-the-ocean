
##### 18/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#  - Train & tune GAM models per income classes 

### Last update: 19/01/23

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
library("merTools")
library("rworldmap")
#install.packages("MuMIn")
#install.packages("merTools")

worldmap <- getMap(resolution = "high")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

### First, load the data 
setwd(paste(WD,"/data/complete_data/", sep = "")) ; dir()
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

### Step 1) Explore various GAMs set up at global and GNI to choose best one 
# ?gam
# Fits a generalized additive model (GAM) to data, the term `GAM' being taken to include any quadratically penalized GLM and a
# variety of other models estimated by a quadratically penalised likelihood type approach (see ‘family.mgcv’).
# The degree of smoothness of model terms is estimated as part of fitting. ‘gam’ can also fit any GLM subject to multiple quadratic penalties
# (including estimation of degree of penalization). Confidence/credible intervals are readily available for any quantity predicted using a fitted model.
# Smooth terms are represented using penalized regression splines (or similar smoothers) with smoothing parameters selected by
# GCV/UBRE/AIC/REML or by regression splines with fixed degrees of freedom (mixtures of the two are permitted). Multi-dimensional
# smooths are available using penalized thin plate regression splines (isotropic) or tensor product splines (when an isotropic
# smooth is inappropriate), and users can add smooths.  Linear functionals of smooths can also be included in models. For an
# overview of the smooths available see ‘smooth.terms’. For more on specifying models see ‘gam.models’, ‘random.effects’ and
# ‘linear.functional.terms’. For more on model selection see ‘gam.selection’. Do read ‘gam.check’ and ‘choose.k’.

# gam1 <- gam(y ~ s(p1) + s(p2) + s(p3) + s(p4) + s(p5) + s(p6), data = scaled.all, method = "REML")
# summary(gam1) # R-sq.(adj) =  0.782   Deviance explained = 78.7%
# AIC(gam1) # -5729.892 # better than in 2019
# # https://campus.datacamp.com/courses/nonlinear-modeling-with-generalized-additive-models-gams-in-r/interpreting-and-visualizing-gams?ex=8
# gam.check(gam1) # Converged after 11 iterations, BUT: signif patterns in residuals (pvalue < 0.01) for several smooths: 1, 2, 4 and 5. Not enough basis functions (number 'k') for those. Re-run gam with higher k for these smooth terms
# # plot(gam1)
#
# # Change k values according to gam.check
# gam2 <- gam(y ~ s(p1,k=20,bs="tp") + s(p2,k=20,bs="tp") + s(p3,bs="tp") + s(p4,k=20,bs="tp") + s(p5,k=20,bs="tp") + s(p6,bs="tp"),
#     data = scaled.all, method = "REML")
#
# summary(gam2) # Increased skill
# AIC(gam2) # -6285.707; lower AIC, good
# gam.check(gam2) # k still too low for p1, p3, p4 and p5
#
# # Re-change k values
# gam3 <- gam(y ~ s(p1,k=30,bs="tp") + s(p2,k=20,bs="tp") + s(p3,k=10,bs="tp") + s(p4,k=30,bs="tp") + s(p5,k=30,bs="tp") + s(p6,bs="tp"),
#     data = scaled.all, method = "REML")
#
# summary(gam3) # Still very good skill
# AIC(gam3) # -6483.425 slightly lower, good
# gam.check(gam3) #
#
# ### Keep latest formula, increase k for p1 and p5 and add year and country as random effects with + s(X,"re")
# gam4 <- gam(y ~ s(p1,k=40,bs="tp") + s(p2,k=20,bs="tp") + s(p3,k=10,bs="tp") + s(p4,k=30,bs="tp") + s(p5,k=40,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re") + s(year, bs = "re"), data = scaled.all, method = "REML")
#
# summary(gam4) # R-sq.(adj) =  0.98 || Deviance explained = 98.3%
# AIC(gam4) # -7257.103
# gam.check(gam4) #
# # NICE ! Try with just country or year
#
# # gam5, with years as RE
# gam5 <- gam(y ~ s(p1,k=50,bs="tp") + s(p2,k=30,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=20,bs="tp") + s(p6,bs="tp") +
#             s(year, bs = "re"), data = scaled.all, method = "REML")
#
# summary(gam5) # R-sq.(adj) = 0.838 || Deviance explained =  84.7%
# AIC(gam5) # -4503.542
# gam.check(gam5) #
# ### Actuallt similar to no random effects at all...so country should be the random effect
#
# # gam5, with country as RE
# gam6 <- gam(y ~ s(p1,k=50,bs="tp") + s(p2,k=30,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=20,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re"), data = scaled.all, method = "REML")
# #
# summary(gam6) # R-sq.(adj) =  0.98  || Deviance explained = 98.3%
# AIC(gam6) # -7257.103
# gam.check(gam6) #
# ### OK, so use countries as RE
#
#
# ### Next, explore gams like gam6 above but for each GNI class
# ### A°) High income
# # Start without choosing k
# gam.H <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re"), data = scaled.H, method = "REML")
# #
# summary(gam.H) # R-sq.(adj) = 0.951 || Deviance explained = 95.5%
# AIC(gam.H) # -5407.095
# gam.check(gam.H) # p1 and p5 need higher k
#
# gam.H <- gam(y ~ s(p1,k=35,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re"), data = scaled.H, method = "REML")
# #
# summary(gam.H) # R-sq.(adj) = 0.959  || Deviance explained = 96.4%
# AIC(gam.H) # -5547.712
# gam.check(gam.H) # this should be ok
#
# ### try w/out country as re?
# gam.H <- gam(y ~ s(p1,k=35,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(p6,bs="tp")
#             , data = scaled.H, method = "REML")
# #
# summary(gam.H) # R-sq.(adj) = 0.858 || Deviance explained =  87%
# AIC(gam.H) # -4471.823
# gam.check(gam.H) # this should be ok
# ### Nope, should keep countrries as RE!
# gam.H <- gam(y ~ s(p1,k=35,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(p6,bs="tp") +
#             s(year, bs = "re"), data = scaled.H, method = "REML")
# #
# summary(gam.H) # R-sq.(adj) = 0.858 || Deviance explained = 87%
# AIC(gam.H) # -4471.823
# ### --> keep country as RE !
#
#
# ### B°) UM
# gam.UM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re"), data = scaled.UM, method = "REML")
# #
# summary(gam.UM) # R-sq.(adj) = 0.916 || Deviance explained = 93.1%
# AIC(gam.UM) # -1414.24
# gam.check(gam.UM) # actually already ok !
#
#
# ### C°) LM
# gam.LM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
#             s(country, bs = "re"), data = scaled.LM, method = "REML")
# #
# summary(gam.LM) # R-sq.(adj) = 0.968 || Deviance explained = 97.5%
# AIC(gam.LM) # -708.9213
# gam.check(gam.LM) # actually already ok !
#
#
# ### D°) L - L has only got 35 data points so manually reduce k values
# gam.L <- gam(y ~ s(p1,k=3,bs="tp") + s(p2,k=3,bs="tp") + s(p3,k=3,bs="tp") + s(p4,k=3,bs="tp") + s(p5,k=3,bs="tp") + s(p6,k=3,bs="tp") +
#             s(country, k=3, bs = "re"), data = scaled.L, method = "REML")
# #
# summary(gam.L) # R-sq.(adj) = 0.989 || Deviance explained = 99.3%
# AIC(gam.L) # -176.3701
# gam.check(gam.L) # actually already ok !
# # Remove country as RE?
# gam.L <- gam(y ~ s(p1,k=2,bs="tp") + s(p2,k=2,bs="tp") + s(p3,k=2,bs="tp") + s(p4,k=2,bs="tp") + s(p5,k=2,bs="tp") + s(p6,k=2,bs="tp")
#             , data = scaled.L, method = "REML")
# #
# summary(gam.L) # R-sq.(adj) = 0.963  || Deviance explained = 97.4%
# AIC(gam.L) # -135.7791
# ### Only a slight decrease actually. Keep it


### ------------------------------------------------------------------------

### Step 2) Test the GAMs with the selected variables from the RF optimization procedure
# form.H <- as.formula("y ~ p2 + p3 + p4 + p5")
# form.UM <- as.formula("y ~ p1 + p3 + p4 + p5")
# form.LM <- as.formula("y ~ p1 + p2 + p4 + p5")
# form.L <- as.formula("y ~ p1 + p4 + p5 + p6")

### Start with high income
# Basic gam from above and then gam with lower preds
gam.H <- gam(y ~ s(p1,k=35,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(p6,bs="tp") +
            s(country, bs = "re"), data = scaled.H, method = "REML")
#      
gam.H.red <- gam(y ~ s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(country, bs = "re"), data = scaled.H, method = "REML")
# Check AIC
AIC(gam.H) # -5547
AIC(gam.H.red) # -5356
summary(gam.H)
summary(gam.H.red) # slight reduction in dev explaiend and AIC, so basically the same
     
### UM 
gam.UM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
            s(country, bs = "re"), data = scaled.UM, method = "REML")
#
gam.UM.red <- gam(y ~ s(p1,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(country, bs = "re"), data = scaled.UM, method = "REML")

AIC(gam.UM) # -1414.24
AIC(gam.UM.red) # -1385.34
summary(gam.UM)
summary(gam.UM.red) # slight reduction in dev explaiend and AIC, so basically the same

### LM
gam.LM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") + s(country, bs = "re"), data = scaled.LM, method = "REML")
gam.LM.red <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(country, bs = "re"), data = scaled.LM, method = "REML")

AIC(gam.LM) # 
AIC(gam.LM.red) # clear reduction in AIC

### Anyways, doesn't look like reducing the nb of variables necessarily improves the GAMs...run the optimization tests like for RF

### 10/12/19: Examine concurvity (non linear dependencies among predictors in the GAMs)
?concurvity
# Concurvity occurs when some smooth term in a model could be approximated by one or more of the other smooth terms in the model. This is often the case when a smooth of space is included in a model, along with smooths of other covariates that also vary more or less smoothly in space. Similarly it tends to be an issue in models including a smooth of time, along with smooths of other time varying covariates.
# Concurvity can be viewed as a generalization of co-linearity, and causes similar problems of interpretation. It can also make estimates somewhat unstable


gam.H <- gam(y ~ s(p1,k=35,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,k=50,bs="tp") + s(p6,bs="tp") +
            s(country, bs = "re"), data = scaled.H, method = "REML")
            
mgcv::concurvity(gam.H, full = T)            
# This routine computes three related indices of concurvity, all bounded between 0 and 1, with 0 indicating no problem, and 1 indicating total lack of identifiability.
# Here, all values are close to 1 which is indicative of high concurvity issues !! Meaning that some smooth terms in the model can just be approximated from one or more of the other smooth terms (collinearity).

gam.UM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
            s(country, bs = "re"), data = scaled.UM, method = "REML")
mgcv::concurvity(gam.UM, full = T)  
# Same ! 

gam.LM <- gam(y ~ s(p1,bs="tp") + s(p2,bs="tp") + s(p3,bs="tp") + s(p4,bs="tp") + s(p5,bs="tp") + s(p6,bs="tp") +
            s(country, bs = "re"), data = scaled.LM, method = "REML")
mgcv::concurvity(gam.LM, full = T)  
# Same...

### Ok, calculate concurvity with fewer and uncorrelated parameters
gam.test <- gam(y ~ s(p5,bs="tp") + s(p1, bs = "re") + s(p3, bs = "re"), data = scaled.H, method = "REML")
mgcv::concurvity(gam.test, full = TRUE)  
summary(gam.test)

gam.test <- gam(y ~ s(p5,k=50,bs="tp") + s(country, bs = "re"), data = scaled.H, method = "REML")
summary(gam.test)
mgcv::concurvity(gam.test, full = TRUE)  
# Yeah, basically p5 does everything by itself here 
# And its smooth terms can be approximated from those of the countries' random effects!

### ------------------------------------------------------------------------

### Step 3) Examine changes in AIC/r2/MSE when reducing the nb of predictors
### To do so, need to create all possible formulae, from univariate to bivariate to all predictors together 
# ?combn
incomes <- c("H","UM","LM","L")
preds <- c("p1","p2","p3","p4","p5","p6")
# t(combn(preds, m = 1))
# t(combn(preds, m = 4))
# t(combn(preds, m = 6))

require("parallel")
#inc <- "H"
#i <- 1

res.gams <- mclapply(incomes, function(inc) {
    
            # Useless message
            message(paste("Running 100 GAMs for GNI-", inc, sep = ""))
            if(inc == "H") {
                dat <- scaled.H
            } else if(inc == "UM") {
                dat <- scaled.UM
            } else if(inc == "LM") {
                dat <- scaled.LM
            } else {
                dat <- scaled.L
            } # eo if else loop
            
            dat$country <- as.factor(dat$country)
            
            # Running gams for all possible formulae
            list <- do.call(c, lapply(seq_along(preds), combn, x = preds, simplify = F))
            l <- length(list)
            
            # Run a function to train and predict GAMs based on 10-fold CV
            res <- lapply(c(1:l), function(i) {
                
                        # Get corresponding element of the list - ergo the combination of predictors
                        f <- list[[i]]
                        npreds <- length(f) # n of predictors
                        form <- paste(f, collapse = '+') # formula of predictors
                        
                        message(paste(i," || Training GAMs for GNI-",inc, " based on ", form, sep = ""))
                        
                        # Create empty vectors of MSE, r2 and AIC (will be filled with k values - 100)
                        AIC.cv <- NA
                        r2.cv <- NA
                        MSE.cv <- NA
                        
                        # Use default k for inc which are not "L", otherwise use k = 2
                        if( inc != "L" ) {
                            
                            for(k in c(1:100)) {
                            
                                # Create train and test dataset to run the GAM and compute r2 and MSE from CV
                                pos <- sample(1:10, nrow(dat), replace = T) # randomly assign a number from 1-10 for each data point
                                while(length(table(pos)) != 10) { # Prevent that one number is not represented in small samples (e.g. scaled.L)
                                      pos <- sample(1:10, nrow(dat), replace = T)
                                }
                                trainGAM <- dat[pos != 10,] # head(trainGAM) 
                                testGAM <- dat[pos == 10,] # head(testGAM) 
                            
                                # Train GAMS with if else loops 
                                if( npreds == 6 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","x5","x6","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","x5","x6","country")
                                    # Force the 'countries' in testing2 to be the same as in training2 
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(x2, bs = "tp") + s(x3, bs = "tp") + 
                                            s(x4, bs = "tp") + s(x5, bs = "tp") + s(x6, bs = "tp") + 
                                            s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2$y - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                                 
                                } else if( npreds == 5 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","x5","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","x5","country")
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                                
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(x2, bs = "tp") + s(x3, bs = "tp") + 
                                            s(x4, bs = "tp") + s(x5, bs = "tp") + s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse  
                            
                                } else if( npreds == 4 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","country")
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # train GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(x2, bs = "tp") + s(x3, bs = "tp") + 
                                            s(x4, bs = "tp") + s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 3 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","country")
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                                
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(x2, bs = "tp") + s(x3, bs = "tp") + 
                                            s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 2 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","country")
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(x2, bs = "tp") + s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 1 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","country")
                                    testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # GAM
                                    gam <- gam(y ~ s(x1, bs = "tp") + s(country, bs = "re") , data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } # eo else if loop
                            
                            } # eo for loop
                        
                            # Compute average AIC/r2/MSE across the 100 CV runs
                            aic <- mean(AIC.cv, na.rm = T)
                            AIC.sd <- sd(AIC.cv, na.rm = T)
                            r2 <- mean(r2.cv, na.rm = T)
                            r2.sd <- sd(r2.cv, na.rm = T)
                            mse <- mean(MSE.cv, na.rm = T)
                            mse.sd <- sd(MSE.cv, na.rm = T)
                            
                        } else {
                            
                            for(k in c(1:100)) {
                            
                                # Create train and test dataset to run the GAM and compute r2 and MSE from CV
                                pos <- sample(1:10, nrow(dat), replace = T) # randomly assign a number from 1-10 for each data point
                                while(length(table(pos)) != 10) { # Prevent that one number is not represented in small samples (e.g. scaled.L)
                                      pos <- sample(1:10, nrow(dat), replace = T)
                                }
                                trainGAM <- dat[pos != 10,] # head(trainGAM) 
                                testGAM <- dat[pos == 10,] # head(testGAM) 
                            
                                # Train GAMS with if else loops 
                                if( npreds == 6 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","x5","x6","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","x5","x6","country")
                                    # Force the 'countries' in testing2 to be the same as in training2 
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, k=2, bs = "tp") + s(x2, k=2, bs = "tp") + s(x3, k=2, bs = "tp") + 
                                            s(x4, k=2, bs = "tp") + s(x5, k=2, bs = "tp") + s(x6, k=2, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2$y - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                                 
                                } else if( npreds == 5 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","x5","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","x5","country")
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                                
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, k=2, bs = "tp") + s(x2, k=2, bs = "tp") + s(x3, k=2, bs = "tp") + 
                                            s(x4, k=2, bs = "tp") + s(x5, k=2, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse  
                            
                                } else if( npreds == 4 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","x4","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","x4","country")
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # train GAM
                                    gam <- gam(y ~ s(x1, k=2, bs = "tp") + s(x2, k=2, bs = "tp") + s(x3, k=2, bs = "tp") + 
                                            s(x4, k=2, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 3 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","x3","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","x3","country")
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                                
                                    # Train GAM
                                    gam <- gam(y ~ s(x1, k=2, bs = "tp") + s(x2, k=2, bs = "tp") + s(x3, k=2, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 2 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","x2","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","x2","country")
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # GAM
                                    gam <- gam(y ~ s(x1, k=2, bs = "tp") + s(x2, k=2, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } else if( npreds == 1 ) {
                            
                                    training2 <- trainGAM[,c("y",f,"country")] # head(training2)
                                    colnames(training2) <- c("y","x1","country")
                                    testing2 <- testGAM[,c("y",f,"country")] # head(training2)
                                    colnames(testing2) <- c("y","x1","country")
                                    #testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                            
                                    # GAM
                                    gam <- gam(y ~ s(x1, k=1, bs = "tp") + s(country, bs = "re"), data = training2)
                                        
                                    # Extract and supply AIC
                                    AIC.cv[k] <- AIC(gam)
                                    r2.cv[k] <- round(summary(gam)$r.sq, 4)
                                    # Predict
                                    prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                    # Compute MSE from predictions
                                    mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                    # And provide to vector at the k level 
                                    MSE.cv[k] <- mse
                            
                                } # eo else if loop
                            
                            } # eo for loop
                        
                            # Compute average AIC/r2/MSE across the 100 CV runs
                            aic <- mean(AIC.cv, na.rm = T)
                            AIC.sd <- sd(AIC.cv, na.rm = T)
                            r2 <- mean(r2.cv, na.rm = T)
                            r2.sd <- sd(r2.cv, na.rm = T)
                            mse <- mean(MSE.cv, na.rm = T)
                            mse.sd <- sd(MSE.cv, na.rm = T)
                            
                        } # eo if else loop 
                        
                        # Return
                        table <- data.frame(id = i, nPV = npreds, PV = form, 
                                AIC = aic, AIC.sd = AIC.sd, 
                                R2 = r2, R2.sd = r2.sd, 
                                MSE = mse, MSE.sd = mse.sd)
                                
                        return(table)

                } # eo FUN
                
            ) # eo lapply
            
            # Rbind and return
            table <- do.call(rbind, res)
            # dim(table) ; summary(table)
            table$GNI <- inc
            rm(res); gc()
            
            # Return
            return(table)
    
    }, mc.cores = 5
     
) # eo mclapply - incomes
# Rbind
ddf <- dplyr::bind_rows(res.gams)
dim(ddf)
summary(ddf)

# 18/01/23: Save so you don't have to wait 15min everytime 
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GAM_optimization_per_GNI")
save(ddf, file = "table_gam_tests_18_01_23.Rdata")
rm(res.gams) ; gc()



### 19/01/23: Plot distribution of MSE/r2/AIC etc. facet per GNI and per npreds
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GAM_optimization_per_GNI")
ddf <- get(load("table_gam_tests_18_01_23.Rdata"))
dim(ddf)
str(ddf)
summary(ddf)

plot1 <- ggplot(aes(x = factor(nPV), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of predictors") + ylab("MSE") + theme_bw() + facet_wrap(~factor(ddf$GNI), ncol = 2, scales = "free") 

plot2 <- ggplot(aes(x = factor(nPV), y = AIC, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of predictors") + ylab("AIC") + theme_bw() + facet_wrap(~factor(ddf$GNI), ncol = 2, scales = "free") 

plot3 <- ggplot(aes(x = factor(nPV), y = R2, fill = factor(GNI)), data = ddf[ddf$R2>0.5,]) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of predictors") + ylab("R2") + theme_bw() + facet_wrap(~factor(ddf[ddf$R2>0.5,"GNI"]), ncol = 2, scales = "free") 

setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
ggsave(plot = plot1, filename = "boxplot_test_GAMs_MSE_nPV_GNI.pdf", dpi = 300, height = 7, width = 7)
ggsave(plot = plot2, filename = "boxplot_test_GAMs_AIC_nPV_GNI.pdf", dpi = 300, height = 7, width = 7)
ggsave(plot = plot3, filename = "boxplot_test_GAMs_R2_nPV_free.pdf", dpi = 300, height = 7, width = 7)

### First conclusion: GAM skill increase with n PVs, 6 being the top everywhere (R2, MSE, AIC)

### Check ranking in univariate runs
# For H
ddf[ddf$nPV == 1 & ddf$GNI == "H",]
### --> p6 > p1/p4 > p2/p3/p5 (but very weak differences overall). Much clearer for the income classes below

# For UM
ddf[ddf$nPV == 1 & ddf$GNI == "UM",]
### --> p5 > p2 > p6/p4 > p3

# For LM
ddf[ddf$nPV == 1 & ddf$GNI == "LM",]
### --> p6 > p3 > p1/p2/p4/p5

# For L
ddf[ddf$nPV == 1 & ddf$GNI == "L",]
### --> p4 > p5 > p3 > p1 > p2/p6

### From the comparison of R2/AIC/MSE between classes, it seems like a minimum of 4 PVs is needed especially for "L" countries.
### This is the same number as used in the RF, so compare the models' skills for nPV == 4
# For H
ddf[ddf$nPV == 4 & ddf$GNI == "H",]
### --> No real changes in R2, the best combinations are: p2+p3+p4+p5 (the one used in RF)/ p2+p3+p4+p6/ p2+p3+p5+p6/ p2+p4+p5+p6/ p3+p4+p5+p6

# For UM
ddf[ddf$nPV == 4 & ddf$GNI == "UM",]
### --> No real changes in R2, the best combinations are: p2+p3+p4+p5/ p3+p4+p5+p6/ p1+p4+p5+p6/ p1+p3+p5+p6/ p1+p3+p4+p5 (RF one)/ p1+p2+p3+p4 

# For LM
ddf[ddf$nPV == 4 & ddf$GNI == "LM",]
### --> all relatively similar

# For L
ddf[ddf$nPV == 4 & ddf$GNI == "L",]
### --> all ok except : p1+p2+p3+p6

### Decision: run GAMs using all combinations of 4 PVs (except p1+p2+p3+p6 in income L). That is about 15 sets of PVs for each class...
### so if you were to run 1'000 models like for RF than that would be ~67 runs models for each set...
### OR: you just take the best one (or the top 20 ones) out of those 1'000 GAMs (like for RF) and make projections our of those. 
### --> Ask Charlotte. 

### Plot correlation heatmaps (Spearman'r rank corr coeff) per GNI class (kind of did this in Script 1.1 already)
incomes <- c("H","UM","LM","L")
# inc <- "H"
for(inc in incomes) {
    
        # Useless message
        message(paste("Computing correlations for GNI-", inc, sep = ""))
        if(inc == "H") {
            dat <- scaled.H
        } else if(inc == "UM") {
            dat <- scaled.UM
        } else if(inc == "LM") {
            dat <- scaled.LM
        } else {
            dat <- scaled.L
        } # eo if else loop
        
        # Compute corr matrix
        cormat <- round(cor(dat[,c("p1","p2","p3","p4","p5","p6")], method = "spearman"), 3)
         # head(cormat)
         melted_cormat <- melt(cormat)
         # Get lower triangle of the correlation matrix
         get_lower_tri <- function(cormat) {
           cormat[upper.tri(cormat)] <- NA
           return(cormat)
         }
         # Get upper triangle of the correlation matrix
         get_upper_tri <- function(cormat) {
           cormat[lower.tri(cormat)]<- NA
           return(cormat)
         }
         # Re-order cormat
         reorder_cormat <- function(cormat){
             # Use correlation between variables as distance
             dd <- as.dist((1-cormat)/2)
             hc <- hclust(dd)
             cormat <-cormat[hc$order,hc$order]
         } # eo fun
         # Reorder the correlation matrix
         cormat <- reorder_cormat(cormat)
         upper_tri <- get_upper_tri(cormat)
         # Melt the correlation matrix
         melted_cormat <- melt(upper_tri, na.rm = T)
       
         # Create a ggheatmap
         ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
                      geom_tile(color = "white") + scale_fill_gradient2(low = "#3288bd", high = "#d53e4f", mid = "white",
                            midpoint = 0, limit = c(-1,1), space = "Lab",
                            name="Spearman\nCorrelation") +
                       theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1,
                            size = 12, hjust = 1)) + coord_fixed()
       
         ggheatmap2 <- ggheatmap + geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
                            theme(axis.title.x = element_blank(),
                              axis.title.y = element_blank(),
                              panel.grid.major = element_blank(),
                              panel.border = element_blank(),
                              panel.background = element_blank(),
                              axis.ticks = element_blank(),
                              legend.justification = c(1, 0),
                              legend.position = c(0.6, 0.7),
                              legend.direction = "horizontal")+
                              guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                                            title.position = "top", title.hjust = 0.5))
         # Save
         ggsave(plot = ggheatmap2, filename = paste("heatmap_corr_",inc,".pdf", sep=""), dpi = 300, width = 6, height = 6)
        
        ### And add simple pairwise correlation matrices with the response variable
        pdf(file = paste("pairwise_scatter_",inc,".pdf", sep = ""), height = 17, width = 18)
            pairs(~y+p1+p2+p3+p4+p5+p6, data = dat[,c("y","p1","p2","p3","p4","p5","p6")], main = "Heatmap of correlation coefficients")
        dev.off()
            
} # eo for loop 

### Results: same as in Script 1.1 of course but at least you have nicer plots now. 
### Conclusion: there are colinear PVs for each GNI class, and even a lot for the L class. Therefore, having GAMs with 6 PVs is ill advised here. Yet, 5 should be possible for the higher income classes (H and UM, maybe not LM). Again, liek for RF, for the L income class, you should include GAMs with nPVS = 3 (for the LM income class too).

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


### Now, estimate best number of basis functions, k
### In a FUN, looping through income classes and growing k values, run GAMs for each predictor set 

# k <- 3
# inc <- "L"
# l <- 2
# i <- 4

require("parallel")
incomes <- c("H","UM","LM","L")
res.inc <- lapply(incomes, function(inc) {

                # Choose range of k to explore as a function of GNI class. Has to be lower for L and LM income class (1-10; 1-20) because of the lower number of points available
                if(inc == "H") {
                    dat <- scaled.H
                    list.pred <- list.preds.H
                    l <- length(list.pred)
                    k.values <- c(2:25)
                } else if(inc == "UM") {
                    dat <- scaled.UM
                    list.pred <- list.preds.UM
                    l <- length(list.pred)
                    k.values <- c(2:25)
                } else if(inc == "LM") {
                    dat <- scaled.LM
                    list.pred <- list.preds.LM
                    l <- length(list.pred)
                    k.values <- c(2:10)
                } else {
                    dat <- scaled.L
                    list.pred <- list.preds.L
                    l <- length(list.pred)
                    k.values <- c(2:10)
                } # eo if else loop

                # Convert country to factor
                dat$country <- as.factor(dat$country)
                
                res.k <- mclapply(k.values, function(k) {
                    
                        ### Fill table in a for loop by running each set of PVs proportionally
                        res <- lapply(c(1:l), function(i) {
                                    
                                    # Get vector of pred names based on i
                                    pred <- list.pred[[i]]
                                    # pred <- c("p2","p4","p5","p6")
                                    
                                    # Useless message
                                    message(paste("Training 10 GAMs for GNI-",inc," based on k = ",k," with ",paste(pred, collapse = '+'),sep = ""))
                                    # Define the number of n times to run the PV set 
                                    N <- 10
                                    
                                    # Prepare the table to be filled with AIC, r2 MSE and DevExpl values 
                                    table2fill <- data.frame(form = rep(NA,N), AIC = NA, r2 = NA, MSE = NA, DevExpl = NA)
                                    table2fill[c(1:N),"form"] <- paste(pred, collapse = '+')
                                    
                                    for(n in c(1:N)) {
            
                                            # Split between train and test dataset to run the GAM 
                                            pos <- sample(1:10, nrow(dat), replace = T)
                                            while(length(table(pos)) != 10) {
                                                    pos <- sample(1:10, nrow(dat), replace = T)
                                            } # eo while
                                            trainGAM <- dat[pos != 10,] 
                                            testGAM <- dat[pos == 10,] 
                                            
                                            training2 <- trainGAM[,c("y",pred,"country")] # head(training2)
                                            testing2 <- testGAM[,c("y",pred,"country")] # head(training2)
                                            testing2 <- testing2[which(testing2$country %in% unique(training2$country)),]
                                            
                                            ### Adjust colnames and train GAMs depending on the number of PVs (5, 4 or 4)
                                            # If else loop
                                            if( length(pred) == 5 ) {
                                                
                                                colnames(training2) <- c("y","x1","x2","x3","x4","x5","country")
                                                colnames(testing2) <- c("y","x1","x2","x3","x4","x5","country")
                                                # Train GAM
                                                gam <- gam(y ~ s(x1,k=k,bs="tp") + s(x2,k=k,bs="tp") + s(x3,k=k,bs="tp") + 
                                                            s(x4,k=k,bs="tp") + s(x5,k=k,bs="tp") + s(country,bs="re"), data = training2)
                    
                                            } else if( length(pred) == 4 ) {
                                                
                                                colnames(training2) <- c("y","x1","x2","x3","x4","country")
                                                colnames(testing2) <- c("y","x1","x2","x3","x4","country")
                                                # Train GAM
                                                gam <- gam(y ~ s(x1,k=k,bs="tp") + s(x2,k=k,bs="tp") + s(x3,k=k,bs="tp") + 
                                                            s(x4,k=k,bs="tp") + s(country,bs="re"), data = training2)
                    
                                            } else {
                                                
                                                colnames(training2) <- c("y","x1","x2","x3","country")
                                                colnames(testing2) <- c("y","x1","x2","x3","country")
                                                # Train GAM
                                                gam <- gam(y ~ s(x1,k=k,bs="tp") + s(x2,k=k,bs="tp") + s(x3,k=k,bs="tp") + s(country,bs="re"), data = training2)
                    
                                            } # eo if else loop
                        
                                            # Predict
                                            prediction <- mgcv::predict.gam(object = gam, newdata = testing2[,c(2:length(testing2))] )
                                            # Compute MSE from predictions
                                            mse <- (sum((testing2[,"y"] - prediction)^2) / nrow(testing2))
                                            
                                            # Extract AIC/r2/MSE/DevExpl and provide to vector at the n row
                                            table2fill[n,"AIC"] <- AIC(gam)
                                            table2fill[n,"r2"] <- round(summary(gam)$r.sq,4) 
                                            table2fill[n,"MSE"] <- mse
                                            table2fill[n,"DevExpl"] <- round(summary(gam)$dev.expl,4)
                                            rm(prediction,gam,trainGAM,testGAM,testing2,training2)
                                            gc()
                
                                    } # eo for loop - n in N
                                    
                                    return(table2fill)
                                    
                                } # eo FUN
                                
                        ) # eo lapply - i in l
                        # Rbind
                        table <- dplyr::bind_rows(res)
                        table$k <- k
                        rm(res); gc()
                        
                        # Return
                        return(table)
            
                    }, mc.cores = 5 # Can't be higher because a mclapply is applied per elements of lapply (income classes)
    
                ) # mclapply - k.values
        
                # Rbind to check results 
                table <- dplyr::bind_rows(res.k)
                table$GNI <- inc
                rm(res.k); gc()
                # Return
                return(table)
        
            } # eo fun - inc
        
) # eo lapply - incomes      
  
# Rbind to check results 
table <- dplyr::bind_rows(res.inc)
dim(table); head(table)
summary(table)
table
rm(res.inc); gc()

setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/GAM_optimization_per_GNI")
save(table, file = "table_gam_tests_k_19_01_23.Rdata")

### Make plots
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
# MSE variations
plot <- ggplot(aes(x = factor(k), y = MSE, fill = factor(GNI)), data = table) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("") + ylab("MSE") + theme_bw() + facet_wrap(~factor(table$GNI), ncol = 2, scales = "free") 
            
ggsave(plot = plot, filename = "boxplot_MSE_k_GNI.pdf", dpi = 300, height = 8, width = 10)

# R2 variations
plot <- ggplot(aes(x = factor(k), y = r2, fill = factor(GNI)), data = table) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("") + ylab("R2") + theme_bw() + facet_wrap(~factor(table$GNI), ncol = 2, scales = "free") 
            
ggsave(plot = plot, filename = "boxplot_r2_k_GNI.pdf", dpi = 300, height = 8, width = 10)

# AIC variations
plot <- ggplot(aes(x = factor(k), y = AIC, fill = factor(GNI)), data = table) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("") + ylab("AIC") + theme_bw() + facet_wrap(~factor(table$GNI), ncol = 2, scales = "free") 
            
ggsave(plot = plot, filename = "boxplot_AIC_k_GNI.pdf", dpi = 300, height = 8, width = 10)

# DevExpl variations
plot <- ggplot(aes(x = factor(k), y = DevExpl, fill = factor(GNI)), data = table) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("") + ylab("AIC") + theme_bw() + facet_wrap(~factor(table$GNI), ncol = 2, scales = "free") 
            
ggsave(plot = plot, filename = "boxplot_DevExpl_k_GNI.pdf", dpi = 300, height = 8, width = 10)


### At the GNI level
plot <- ggplot(aes(x = factor(k), y = MSE), data = table[table$GNI == "H",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE") 
ggsave(plot = plot, filename = "boxplot_MSE_k_H.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = MSE), data = table[table$GNI == "UM",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE") 
ggsave(plot = plot, filename = "boxplot_MSE_k_UM.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = MSE), data = table[table$GNI == "LM",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE") 
ggsave(plot = plot, filename = "boxplot_MSE_k_LM.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = MSE), data = table[table$GNI == "L",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE") 
ggsave(plot = plot, filename = "boxplot_MSE_k_L.pdf", dpi = 300, height = 5, width = 7)


### Still unclear what the relationship is...switch to dot plot
plot <- ggplot(aes(x = factor(k), y = log(MSE)), data = table[table$GNI == "H",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE (log)") 
ggsave(plot = plot, filename = "boxplot_logMSE_k_H.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = log(MSE)), data = table[table$GNI == "UM",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE (log)") 
ggsave(plot = plot, filename = "boxplot_logMSE_k_UM.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = log(MSE)), data = table[table$GNI == "LM",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE (log)") 
ggsave(plot = plot, filename = "boxplot_logMSE_k_LM.pdf", dpi = 300, height = 5, width = 7)

plot <- ggplot(aes(x = factor(k), y = log(MSE)), data = table[table$GNI == "L",]) +
            geom_boxplot(colour = "black", fill = "white") + 
            xlab("k values") + ylab("MSE (log)") 
ggsave(plot = plot, filename = "boxplot_logMSE_k_L.pdf", dpi = 300, height = 5, width = 7)


### Summarize some mean MSE/R2/AIC per k values and class
sum.r2 <- data.frame(table %>% group_by(GNI,k) %>% summarize(R2 = mean(r2,na.rm = T), sd = sd(r2,na.rm = T)) ) # eo ddf
sum.r2

sum.AIC <- data.frame(table %>% group_by(GNI,k) %>% summarize(AIC = ztod lisa ann(AIC,na.rm = T), sd = sd(r2,na.rm = T)) ) # eo ddf
sum.AIC


sum.MSE <- data.frame(table %>% group_by(GNI,k) %>% summarize(MSE = ztod lisa ann(MSE,na.rm = T), sd = sd(r2,na.rm = T)) ) # eo ddf
sum.MSE
### Nothing really stands out...

sum.DevExpl <- data.frame(table %>% group_by(GNI,k) %>% summarize(MSE = ztod lisa ann(DevExpl,na.rm = T), sd = sd(DevExpl,na.rm = T)) ) # eo ddf
sum.DevExpl

### Conclusion: based on R2, AIC and DevExpl profiles, the optimal range seem to be: 
# k = 15-25 for H
# k = 15-25 for UM
# k = 6-10 for LM
# k = 4-8 for L (has to be lower than LM because fewer points)

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
