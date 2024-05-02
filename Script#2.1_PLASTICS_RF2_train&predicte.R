
##### 18/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
#   - Train 1'000 RF models per GNI class by using all potential PV set with 4 PVs, ntree = 200 (based on Script#1.1)
#   - Determine optimal mtry values for each income class based on the the lists of PVs 
#   - Test the effect of 'nodesize' (minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5).)
#   - Predict annual MSW per country using RF models with the determine doptimal parameter values 

### Last update: 23/05/23 (testing the effect of nodesize)

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

worldmap <- getMap(resolution = "high")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

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
# colnames(m.missing)[11] <- "year"
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
### Then, define all sets of PVs that are not collinear at the |0.7| level. Examine colinearity between PVs and target variable - overall and per GNI class

### Potential conflict between p3xp4 and p1xp6
res.cor <- round(cor(scaled.all[,c(1:7)], method = "spearman"), 3)
# On total data, p3 & p4 show a 0.895 cor coeff
### Per GNI class
# H
res.cor.H <- round(cor(scaled.H[,c(2:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear
# UM
res.cor.UM <- round(cor(scaled.UM[,c(2:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear
# LM
res.cor.LM <- round(cor(scaled.LM[,c(2:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear + p1 x p2 ; p1 and p5 are nearly there too
# L
res.cor.L <- round(cor(scaled.L[,c(2:7)], method = "spearman"), 3)
### --> many many PVs are correlated...especially p1xp5xp4xp3...

### Plot corr heatmaps (correlograms) for each income class
require("reshape2")
# Get lower triangle of the correlation matrix
get_lower_tri <- function(cormat) { cormat[base::upper.tri(cormat)] <- NA ; return(cormat) }
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat) { cormat[base::lower.tri(cormat)] <- NA ; return(cormat) }
upper_tri.H <- get_upper_tri(res.cor.H)
upper_tri.UM <- get_upper_tri(res.cor.UM)
upper_tri.LM <- get_upper_tri(res.cor.LM)
upper_tri.L <- get_upper_tri(res.cor.L)
melted_cormat.H <- melt(upper_tri.H, na.rm = T)
melted_cormat.UM <- melt(upper_tri.UM, na.rm = T)
melted_cormat.LM <- melt(upper_tri.LM, na.rm = T)
melted_cormat.L <- melt(upper_tri.L, na.rm = T)

p1 <- ggplot(data = melted_cormat.H, aes(Var2, Var1, fill = value)) + geom_tile(color = "white") + 
    geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) + 
    scale_fill_gradient2(low = "#3288bd", high = "#d53e4f", mid = "white", midpoint = 0,
        limit = c(-1,1), space = "Lab", name = "Spearman\ncorrelation\ncoefficient") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
    coord_fixed() + xlab("") + ylab("") + ggtitle("High income class")

p2 <- ggplot(data = melted_cormat.UM, aes(Var2, Var1, fill = value)) + geom_tile(color = "white") + 
    geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) + 
    scale_fill_gradient2(low = "#3288bd", high = "#d53e4f", mid = "white", midpoint = 0,
        limit = c(-1,1), space = "Lab", name = "Spearman\ncorrelation\ncoefficient") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
    coord_fixed() + xlab("") + ylab("") + ggtitle("Upper medium income class")

p3 <- ggplot(data = melted_cormat.LM, aes(Var2, Var1, fill = value)) + geom_tile(color = "white") + 
    geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) + 
    scale_fill_gradient2(low = "#3288bd", high = "#d53e4f", mid = "white", midpoint = 0,
        limit = c(-1,1), space = "Lab", name = "Spearman\ncorrelation\ncoefficient") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
    coord_fixed() + xlab("") + ylab("") + ggtitle("Lower medium income class")

p4 <- ggplot(data = melted_cormat.L, aes(Var2, Var1, fill = value)) + geom_tile(color = "white") + 
    geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) + 
    scale_fill_gradient2(low = "#3288bd", high = "#d53e4f", mid = "white", midpoint = 0,
        limit = c(-1,1), space = "Lab", name = "Spearman\ncorrelation\ncoefficient") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
    coord_fixed() + xlab("") + ylab("") + ggtitle("Low income class")
    
# Assemble in panel
require("ggpubr")
panel <- ggarrange(p1,p2,p3,p4, labels = c("a","b","c","d"), align = "hv", ncol = 2, nrow = 2)
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
ggsave(plot = panel, filename = "Supplemetary_FigureSX_corr_heatmaps.pdf", dpi = 300, width = 10, height = 10)

### Defining lists if potential PVs for model runs
# Available sets of 4 PVs are: t(combn(c("p1","p2","p3","p4","p5","p6"), m = 4))

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

### Second, use the tuneRF() fun across all PVs combinations (lists above) to estimate the mtry parameter, for each GNI class

# tuned <- tuneRF(scaled[scaled$GNI == "H",c(2:7)],scaled[scaled$GNI == "H",1], mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F) ))
# best.mtry.H[i] <- tuned[which.min(tuned[,2]),1]
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/")
# For testing 
inc <- "H"
i <- 3
n <- 3

mtry.tests <- function(inc) {
    
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
            
            message(paste("\nTraining 1000 RF for GNI-",inc,"\n", sep = ""))
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
                             pos <- sample(1:10, nrow(dat), replace = T)
                             while(length(table(pos)) != 10) {
                                     pos <- sample(1:10, nrow(dat), replace = T)
                             } # eo while
                             trainRF <- dat[pos != 10,] 
                             testRF <- dat[pos == 10,] 
                             # head(trainRF) ; head(testRF)
                             # Perform tuneRF
                             invisible(capture.output( 
                               tuned <- tuneRF(trainRF[,c(2:7)], trainRF[,"y"], mtryStart = 2, stepFactor = 0.9, improve = 1e-10, plot = F, trace = F) # eo tuneRF  
                             ))
                             # Report best mtry value
                             best.mtry <- tuned[which.min(tuned[,2]),1]
                             # And perform RF with it 
                             formula <- as.formula(paste("y~",form, sep = ""))
                             RF <- randomForest(formula, trainRF, mtry = best.mtry, ntree = 201)
                             # summary(RF)
                             # Compute RF skill
                             prediction.RF <- predict(RF, testRF[,preds])
                             pred.full <- exp((prediction.RF*(max[1] - min[1])) + min[1])
                             pred.full[which(pred.full < 0)] <- 0
                             
                             # Compute R2 and MSE of RF model based on the 10% testing set
                             measure <- exp((testRF$y * (max[1] - min[1])) + min[1])
                             r2 <- 1- sum((pred.full-measure)^2)/sum(pred.full^2)
                             # Compute mse of full model
                             mse <- (sum((testRF$y - pred.full)^2) / nrow(testRF))
                             
                             # Summarize all info in ddf
                             skillz <- data.frame(GNI = inc, n = n, formula = form, R2 = r2, MSE = mse, mtry = best.mtry) # eo ddf
                             setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/mtry_tests")
                             save(skillz, file = paste("table_skills_RF_",inc,"_",form,"_",n,"_23_01_23.Rdata", sep = "") )
                             
                         } # eo for loop - n in N
                     
                     } # eo FUN
                     , mc.cores = l
                     
            ) # eo mclapply
    
} # eo FUN

mtry.tests("H")
mtry.tests("UM")
mtry.tests("LM")
mtry.tests("L")

### Check results
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/mtry_tests") # dir()
files <- dir()[grep("_23_01_23",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
ddf <- bind_rows(res)
dim(ddf); summary(ddf) ; head(ddf)

### How many random models? 
nrow(ddf[ddf$R2 <= 0,]) # L and LM models 
nrow(ddf[ddf$R2 >= 0.9,]) # But also a lot of skillfull models 

### Subset: 
ddf2 <- ddf[ddf$R2 > 0.1,]
nrow(ddf2) # 3869 (97% of models)

### Plot distribution of MSE/r2/AIC etc. facet per GNI and per preds
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
plot1 <- ggplot(aes(x = factor(mtry), y = R2, fill = factor(GNI)), data = ddf2[ddf2$mtry %in% c(2:5),]) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("mtry") + ylab("R2") + theme_bw() + facet_wrap(~factor( ddf2[ddf2$mtry %in% c(2:5),"GNI"] ), ncol = 2, scales = "free") 

plot2 <- ggplot(aes(x = factor(mtry), y = MSE, fill = factor(GNI)), data = ddf2[ddf2$mtry %in% c(2:5),]) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("GNI class") + ylab("MSE") + theme_bw() + facet_wrap(~factor( ddf2[ddf2$mtry %in% c(2:5),"GNI"] ), ncol = 2, scales = "free") 

ggsave(plot = plot1, filename = "boxplot_RF2_mtry_tests_R2_23_05_23.pdf", dpi = 300, height = 7, width = 7)
ggsave(plot = plot2, filename = "boxplot_RF2_mtry_MSE_tests_23_05_23.pdf", dpi = 300, height = 7, width = 7)

### Look at most occuring mtry per GNI
summary(factor(ddf2[ddf2$GNI == "H","mtry"])) # should be 3:5 according to this - with 4 the most frequently chosen value
# Examine distrbution of MSE and R2 within high performing H-models
# ggplot(aes(x = factor(mtry), y = MSE, fill = factor(mtry)),
#           data = ddf[ddf$mtry %in% c(1:5) & ddf$GNI == "H" & ddf$R2 >= 0.9,]) +
#        geom_boxplot(colour = "black") + scale_fill_manual(name = "mtry", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") )
#
# ggplot(aes(x = factor(mtry), y = R2, fill = factor(mtry)),
#           data = ddf[ddf$mtry %in% c(1:5) & ddf$GNI == "H" & ddf$R2 >= 0.9,]) +
#        geom_boxplot(colour = "black") + scale_fill_manual(name = "mtry", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") )
### --> no strong differences --> keep mtry = 4

summary(factor(ddf2[ddf2$GNI == "UM","mtry"])) # mtry = 3
summary(factor(ddf2[ddf2$GNI == "LM","mtry"])) # mtry = 2
summary(factor(ddf2[ddf2$GNI == "L","mtry"])) # mtry = 3

### Use these to perform the GNI-level RF models on full data: 
# mtry = 4
# mtry = 3
# mtry = 2
# mtry = 3

### -----------------------------------------------------------------

### 23/05/23: Test the effect of nodesize on MSE and R2 (nodiesize to range between 3:15)

library("parallel")
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/nodesize_tests")

# For testing mclapply below
classes <- c("H","UM","LM","L")
nodesizesss <- seq(from = 3, to = 15, by = 1) ; nodesizesss
# For testing:
inc <- "H"
node <- 8
i <- 3
n <- 13

nodesize.tester <- function(inc) {
    
            if(inc == "H") {
                dat <- scaled.H
                list.pred <- list.preds.H
                l <- length(list.pred)
                mtry <- 4
            } else if(inc == "UM") {
                dat <- scaled.UM
                list.pred <- list.preds.UM
                l <- length(list.pred) 
                mtry <- 3
            } else if(inc == "LM") {
                dat <- scaled.LM
                list.pred <- list.preds.LM
                l <- length(list.pred)
                mtry <- 2
            } else {
                dat <- scaled.L
                list.pred <- list.preds.L
                l <- length(list.pred)
                mtry <- 3
            } # eo if else loop
            
            message(paste("\nTraining 1000 RF for GNI-",inc,"\n", sep = ""))
           
            for(node in nodesizesss) {
           
                message(paste("\nUsing nodesize = ",node, sep = ""))
           
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
                                 pos <- sample(1:10, nrow(dat), replace = T)
                                 while(length(table(pos)) != 10) {
                                     pos <- sample(1:10, nrow(dat), replace = T)
                                 } # eo while
                                 trainRF <- dat[pos != 10,] 
                                 testRF <- dat[pos == 10,]
                             
                                 # Run RF
                                 formula <- as.formula(paste("y~",form, sep = ""))
                                 RF <- randomForest(formula, trainRF, mtry = mtry, ntree = 201, nodesize = node)
                   
                                 # Compute RF skill
                                 prediction.RF <- predict(RF, testRF[,preds])
                                 pred.full <- exp((prediction.RF*(max[1] - min[1])) + min[1])
                                 pred.full[which(pred.full < 0)] <- 0
                             
                                 # Compute R2 and MSE of RF model based on the 10% testing set
                                 measure <- exp((testRF$y * (max[1] - min[1])) + min[1])
                                 r2 <- 1- sum((pred.full-measure)^2)/sum(pred.full^2)
                                 # Compute mse of full model
                                 mse <- (sum((testRF$y - pred.full)^2) / nrow(testRF))
                             
                                 # Summarize all info in ddf
                                 skillz <- data.frame(GNI = inc, n = n, formula = form, R2 = r2, MSE = mse, nodesize = node) # eo ddf
                                 setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/nodesize_tests")
                                 save(skillz, file = paste("table_skills_RF_",inc,"_",form,"_",n,"_nodesize=",node,"_23_05_23.Rdata", sep = "") )
                             
                            } # eo for loop - n in N
                     
                        }, mc.cores = l
                     
                ) # eo mclapply
            
        } # eo for loop - nodes in nodesizesss         
    
} # eo FUN

### Run funcion: 
nodesize.tester(inc = "H")
nodesize.tester(inc = "UM")
nodesize.tester(inc = "LM")
nodesize.tester(inc = "L")

### Examine outputs:
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/nodesize_tests")
# dir(); length(dir())

files <- dir()[grep("_23_05_23",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
ddf <- bind_rows(res)
dim(ddf); summary(ddf) ; head(ddf)

### How many random models? 
nrow(ddf[ddf$R2 <= 0,]) # L and LM models 
nrow(ddf[ddf$R2 >= 0.9,]) # But also a lot of skillfull models 

### Subset: 
ddf2 <- ddf[ddf$R2 > 0.1,]
nrow(ddf2) # 3869 (97% of models)

### Plot distribution of MSE/r2/AIC etc. facet per GNI and per preds
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
plot1 <- ggplot(aes(x = factor(nodesize), y = R2, fill = factor(GNI)), data = ddf2) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("nodesize") + ylab("R2") + theme_bw() + facet_wrap(~factor( ddf2[,"GNI"] ), ncol = 2, scales = "free") 

plot2 <- ggplot(aes(x = factor(nodesize), y = MSE, fill = factor(GNI)), data = ddf2) + geom_boxplot(colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("nodesize") + ylab("MSE") + theme_bw() + facet_wrap(~factor( ddf2[,"GNI"] ), ncol = 2, scales = "free") 

ggsave(plot = plot1, filename = "boxplot_RF2_nodesize_tests_R2_23_05_23.pdf", dpi = 300, height = 7, width = 7)
ggsave(plot = plot2, filename = "boxplot_RF2_nodesize_MSE_tests_23_05_23.pdf", dpi = 300, height = 7, width = 7)

### Re-scaled
plot3 <- ggplot(aes(x = factor(nodesize), y = R2, fill = factor(GNI)), data = ddf2) + geom_boxplot(colour = "black") + 
    scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
    scale_y_continuous(limits = c(0.8,1)) + xlab("nodesize") + ylab("R2") + theme_bw() +
    facet_wrap(~factor( ddf2[,"GNI"] ), ncol = 2, scales = "free") 

# plot4 <- ggplot(aes(x = factor(nodesize), y = MSE, fill = factor(GNI)), data = ddf2) + geom_boxplot(colour = "black") +
#     scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#     scale_y_continuous(limits = c(0.75,1)) + xlab("nodesize") + ylab("MSE") + theme_bw() +
#     facet_wrap(~factor( ddf2[,"GNI"] ), ncol = 2, scales = "free")

ggsave(plot = plot3, filename = "boxplot_RF2_nodesize_tests_R2_23_05_23_rescaled.pdf", dpi = 300, height = 7, width = 7)

### No real differences between nodesize values across income classes. Slught decrease in R2 values beyond nodeise values > 5...


### -----------------------------------------------------------------

setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training/")
# For testing the fun below
inc <- "L"
i <- 3
n <- 3

RF.predicter <- function(inc) {
    
                if(inc == "H") {
                    dat <- scaled.H
                    list.pred <- list.preds.H
                    l <- length(list.pred)
                    mtry.best <- 4
                } else if(inc == "UM") {
                    dat <- scaled.UM
                    list.pred <- list.preds.UM
                    l <- length(list.pred) 
                    mtry.best <- 3
                } else if(inc == "LM") {
                    dat <- scaled.LM
                    list.pred <- list.preds.LM
                    l <- length(list.pred)
                    mtry.best <- 2
                } else {
                    dat <- scaled.L
                    list.pred <- list.preds.L
                    l <- length(list.pred)
                    mtry.best <- 3
                } # eo if else loop
                
                message(paste("\nRunning 1'000 RF for GNI-",inc,"\n", sep = ""))
                
                require("parallel")
                
                res <- mclapply(c(1:l), function(i) {
                 
                            # Get vector of pred names based on i
                            preds <- list.pred[[i]]
                            form <- paste(preds, collapse = '+')
                            # Useless message
                            message( paste("Training GAMs for GNI-",inc," based on ",form,sep = "") )
                            # Define the number of n times to run the PV set 
                            # And perform RF with it 
                            formula <- as.formula(paste("y~",form, sep = ""))
                            
                            # Define the number of n times to run the PV set (total amounts to n = 1000)
                            N <- round(1000/l ,0.1) # convert to integer
                            
                            for(n in c(1:N)) {
                                
                                # Training model # n
                                RF <- randomForest(formula, dat, mtry = mtry.best, ntree = 201)
                            
                                # Compute RF skill based on all data
                                prediction.RF <- predict(RF, dat[,preds])
                                pred.full <- exp((prediction.RF*(max[1] - min[1])) + min[1])
                                pred.full[which(pred.full < 0)] <- 0
                             
                                # Compute R2 and MSE of RF model based on the 10% testing set
                                measure <- exp((dat$y * (max[1] - min[1])) + min[1])
                                r2 <- 1 - sum((pred.full-measure)^2)/sum(pred.full^2)
                                # Compute mse of full model
                                mse <- (sum((dat$y - pred.full)^2) / nrow(dat))
                             
                                # Summarize all info in ddf & save
                                skillz <- data.frame(GNI = inc, model.number = n, formula = form, R2 = r2, MSE = mse) # eo ddf
                                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training")
                                save(skillz, file = paste("table_skills_RF_",inc,"_",form,"_",n,".Rdata", sep = "") ) # saving model skills
                                save(RF, file = paste("RF.full_",inc,"_",form,"_",n,".Rdata", sep = "") ) # saving model object
                                
                                ### Make prediction and compute error to original data 
                                all.countries <- c(unique(as.character( m.missing[m.missing$GNI == inc,"country"] )), as.character(unique(dat$country)) ) 
                                #all.countries <- as.character(unique(dat$country))
                                # all.countries
                                pred <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                                pred[,1] <- unique(all.countries)
                                colnames(pred) <- c("Country", 1990:2019)
                                # Same for errors
                                error_country_perc <- as.data.frame(matrix(NA, ncol = nyears+1, nrow = length(all.countries) ) )
                                error_country_perc[,1] <- unique(all.countries)
                                colnames(error_country_perc) <- c("Country", 1990:2019)

                                # Fill error and pred data.frame with for loop. Make if else loop to account for the data in m.missing too !
                                # land <- "Switzerland"
                                for( land in all.countries ) { 
                                      
                                      message( paste("Saving predictions and errors for ", land, sep = "") )
                                      
                                      if( land %in% unique(dat$country) ) {
                                          
                                          # get the corresponding index 'j' from country.details
                                          j <- as.numeric(rownames(country.details[country.details$Country == land,]))
                                          
                                          # Create ddf containg all data for land of interest (log transformed etc. but not rescaled)
                                          dat.temp <- data.frame(y[,j], p1[,j], p2[,j], p3[,j], p4[,j], p5[,j], p6[,j], land)
                                          names(dat.temp) <- c("y","p1","p2","p3","p4","p5","p6","country")
                                          
                                          dat.temp <- dat.temp[,c(preds,"country")]
                                          scaled.temp <- as.data.frame( scale(dat.temp[,c(1:length(preds))], center = min[preds], scale = max[preds] - min[preds]) )
                                          scaled.temp[which(is.nan(scaled.temp[,2])),2] <- max[2]/min[2]
                                          scaled.temp$country <- dat.temp$country
                              
                                          # Predict y for each year for country y
                                          predict_test_RF <- exp((predict(RF, scaled.temp) * (max[1] - min[1])) + min[1])
                                          # If ever negative values, convert to zero
                                          predict_test_RF[which(predict_test_RF < 0)] <- 0
                                      
                                          # Find the position (YEARs between 1990:2015) to fill in the empty error dataset
                                          error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
                                      
                                          # Compute and supply error
                                          er <- ( (predict_test_RF[error.pos] - y_org[error.pos,j])/y_org[error.pos,j] )*100
                                          error_country_perc[error_country_perc$Country == land,c(error.pos)] <- er
                                          # And supply prediction to pred matrix
                                          pred[pred$Country == land,-1] <- predict_test_RF
                                          
                                      } else if ( land %in% unique(m.missing$country) ) {
                                          
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
                                          predict_test_RF <- exp((predict(RF, scaled.temp) * (max[1] - min[1])) + min[1])
                                          # If ever negatove values, convert to zero
                                          predict_test_RF[which(predict_test_RF < 0)] <- 0
                                          
                                          # And supply prediction to pred matrix
                                          pred[pred$Country == land,-1] <- predict_test_RF
                                          
                                      } # eo else if loop
                                      
                                  } # for loop - l in land
                                  
                                  # Check error_country & pred data.frames
                                  # error_country ; pred
                                  error_avg <- rowMeans(error_country_perc[,c(2:nyears+1)], na.rm = T)
                                  error_country_perc$mean <- error_avg
                                  # summary(error_country_perc)
                                 
                                  # Save outputs 
                                  setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions")
                                  write.table(x = error_country_perc, file = paste("table_error_perc_",inc,"_",form,"_",n,".txt", sep = ""), sep = "\t")
                                  write.table(x = pred, file = paste("table_pred_",inc,"_",form,"_",n,".txt", sep = ""), sep = "\t") 
                                  
                            } # eo for loop n in N
                              
                        } # eo FUN
                        
                        , mc.cores = l
                     
               ) # eo mclapply
               
} # eo fun - RF.predicter

RF.predicter("H")
RF.predicter("UM")
RF.predicter("LM")
RF.predicter("L")

### Check the skills of the trained RF models
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_models_training") #; dir()
# If needed, remove files from older models
# files2delete <- list.files()
# files2delete <- files2delete[!grepl("19_01_23",files2delete)]
# file.remove(files2delete)
length(dir()[grep("table_skills_RF_H",dir())]) 
length(dir()[grep("table_skills_RF_UM",dir())]) 
length(dir()[grep("table_skills_RF_LM",dir())])
length(dir()[grep("table_skills_RF_L_",dir())]) 
# Ok ~ 1000 models each
files <- dir()[grep("table_skills",dir())]
res <- lapply(files, function(f) { d <- get(load(f)); return(d) })
ddf <- bind_rows(res)
# dim(ddf); summary(ddf)

### Examine median scores 
# R2
median(ddf[ddf$GNI == "H","R2"]); IQR(ddf[ddf$GNI == "H","R2"])
median(ddf[ddf$GNI == "UM","R2"]); IQR(ddf[ddf$GNI == "UM","R2"])
median(ddf[ddf$GNI == "LM","R2"]); IQR(ddf[ddf$GNI == "LM","R2"])
median(ddf[ddf$GNI == "L","R2"]); IQR(ddf[ddf$GNI == "L","R2"])

# MSE
median(ddf[ddf$GNI == "H","MSE"]); IQR(ddf[ddf$GNI == "H","MSE"])
median(ddf[ddf$GNI == "UM","MSE"]); IQR(ddf[ddf$GNI == "UM","MSE"])
median(ddf[ddf$GNI == "LM","MSE"]); IQR(ddf[ddf$GNI == "LM","MSE"])
median(ddf[ddf$GNI == "L","MSE"]); IQR(ddf[ddf$GNI == "L","MSE"])


setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
plot <- ggplot(aes(x = factor(GNI), y = R2, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "R2", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("R2") + theme_bw() 
ggsave(plot = plot, filename = "boxplot_full.RF_R2_GNI_23_01_23.pdf", dpi = 300, height = 5, width = 4)

plot <- ggplot(aes(x = factor(GNI), y = MSE, fill = factor(GNI)), data = ddf) + geom_boxplot(colour = "black") + 
            scale_fill_manual(name = "MSE", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
            xlab("GNI classes") + ylab("MSE") + theme_bw()        
ggsave(plot = plot, filename = "boxplot_full.RF_MSE_GNI_23_01_23.pdf", dpi = 300, height = 5, width = 4)


### And check % error for each class/ country?
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions"); dir()
# If needed, remove files from older models
# files2delete <- list.files()
# files2delete <- files2delete[!grepl("19_01_23",files2delete)]
# file.remove(files2delete)

files <- dir()[grep("table_error_perc",dir())]; length(files) # 4000, 1000 per GNI
res <- lapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) })
ddf <- bind_rows(res)
# dim(ddf)
# Correct colnames quickly
colnames(ddf)[c(2:31)] <- c(1990:2019)

### To rank countries according to their mean error (all years included)
m.ddf <- melt(ddf[,c("Country","mean")], id.vars = "Country")
# dim(m.ddf)
# head(m.ddf)
# summary(m.ddf$value) # mean error ranges from -58% to +83%, but median mean error is -0.14% only !
sum.error <- data.frame(m.ddf %>% group_by(Country) %>% summarize(median = median(value)) ) # eo ddf
# summary(sum.error)
sum.error[order(sum.error$median, decreasing = T),] # missing countries have NaNs in errors - makes sense as no obs to begin with

### Compute mean % error per country and year (dcast) and save table (so more variability as the average right above)
m.ddf <- melt(ddf[,c(1:31)], id.vars = "Country")
#colnames(m.ddf) <- c("Country","Year","error")
#head(m.ddf)
# summary(m.ddf$error) # from -75% to +210%, but median error is -0.1% only ! 
med.error <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(med = median(error)) ) # eo ddf
head(med.error); summary(med.error)
# Dcast to put years as columns
d.med.error <- dcast(data = med.error, formula = Country ~ Year)
# dim(d.med.error) # 217x31 - good dimensions 
# summary(d.med.error)
write.table(d.med.error, file = "table_RF_median_errors_percentage_23_01_23.txt", sep = ";")


### Do the same but for ranges (median,IQR...)
files <- dir()[grep("table_pred_",dir())]
res <- lapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) })
ddf <- bind_rows(res)
colnames(ddf)[c(2:31)] <- c(1990:2019)
# Melt to put years as vector
m.ddf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 23'598
colnames(m.ddf)[c(2:3)] <- c("Year","MSW")
# summary(m.ddf$MSW) # 0.001095 0.085928 0.280619 0.361816 0.493844 4.127014 

ensemble.range <- data.frame(m.ddf %>% group_by(Country,Year) %>% 
        summarize(Median = median(MSW, na.rm = T), IQR = IQR(MSW, na.rm = T), 
                Q25th = quantile(MSW, na.rm = T)[2], Q75th = quantile(MSW, na.rm = T)[4],
                Mean = mean(MSW, na.rm = T), Stdev = sd(MSW, na.rm = T)) 
) # eo ddf
# Check
summary(ensemble.range)
# Save this as table for charlotte
write.table(ensemble.range, file = "table_ranges_RF_median_predictions+ranges_22_02_23.txt", sep = ";")

# And save after dcasting
ensemble <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(Median = median(MSW, na.rm = T)) ) # eo ddf
d.med.pred <- dcast(data = ensemble, formula = Country ~ Year)
# dim(d.med.pred) # good dimensions 
# summary(d.med.pred)
write.table(d.med.pred, file = "table_ranges_RF_median_predictions_23_01_23.txt", sep = ";")


### Map this error
n <- joinCountryData2Map(sum.error, joinCode = "NAME", nameJoinColumn = "Country")
# plot map
pdf(file = paste("map_ensemble_median_error_RF_23_01_23.pdf", sep = ""), width = 10, height = 6)
    mapCountryData(n, nameColumnToPlot = "median", mapTitle = "",
    catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65", 
    borderCol = "white", lwd = 0.1, addLegend = T) 
dev.off()

# ### Do the same but...per year
# colnames(ddf) <- c("Country", 1990:2019)
# for(yyy in as.character(c(1990:2019)) ) {
#
#         message(paste("Plotting RF errors for ", yyy, sep = ""))
#         er <- ddf[,c("Country",yyy)]
#         colnames(er) <- c("Country","error")
#         # Map this error
#         n <- joinCountryData2Map(er, joinCode = "NAME", nameJoinColumn = "Country")
#         # plot map
#         pdf(file = paste("map_ensemble_error_RF_",yyy,"_18_01_23.pdf", sep = ""), width = 10, height = 6)
#             mapCountryData(n, nameColumnToPlot = "error", mapTitle = "",
#             catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#             borderCol = "white", lwd = 0.1, addLegend = T)
#         dev.off()
#
# } # eo for loop


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------

### 13/12/19: Compare outputs of GAMs and RF2: predictions and errors
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/RF_predictions2")
# ens.rf <- read.table("table_ensemble_predictions_RF2_13_12_19.txt", sep = "\t", h = T)
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# ens.gam <- read.table("table_ensemble_predictions_GAM_12_12_19.txt", sep = "\t", h = T)
# colnames(ens.gam)[c(2:27)] <- str_replace_all(as.character(colnames(ens.gam)[c(2:27)]),"X","")
# colnames(ens.rf)[c(2:27)] <- str_replace_all(as.character(colnames(ens.rf)[c(2:27)]),"X","")
#
# ### A°) Errors (along the 1:1 line)
# # Compute error based on ensemble GAM to y_org
# errors.gam <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ens.gam$Country)) ) )
# errors.gam[,1] <- unique(ens.gam$Country)
# colnames(errors.gam) <- c("Country", 1990:2015)
# for(land in unique(ens.gam$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ens.gam$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
#       predictions <- ens.gam[ens.gam$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       errors.gam[errors.gam$Country == land,c(error.pos)] <- er
# } # for loop
# summary(errors.gam)
# ### ANd derive pluriannual mean of errors
# m.error.gam <- melt(errors.gam, id.vars = "Country")
# dim(m.error.gam) ; head(m.error.gam)
# summary(m.error.gam$value) # -62.8298  -4.8511  -0.4719   0.3417   4.7548 150.1499
#
# ### Same for RF pred
# errors.rf <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ens.rf$Country)) ) )
# errors.rf[,1] <- unique(ens.rf$Country)
# colnames(errors.rf) <- c("Country", 1990:2015)
# for(land in unique(errors.rf$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ens.rf$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
#       predictions <- ens.rf[ens.rf$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       errors.rf[errors.rf$Country == land,c(error.pos)] <- er
# } # for loop
# summary(errors.rf)
# ### ANd derive pluriannual mean of errors
# m.error.rf <- melt(errors.rf, id.vars = "Country")
# dim(m.error.rf) ; head(m.error.rf)
# summary(m.error.rf$value) # -73.1528  -3.0076  -0.1627   0.4969   2.8931  92.3449
#
# ### Cbind and look at the relationships along the 1:1 line
# head(m.error.rf)
# head(m.error.gam)
# m.errors <- data.frame(Country = m.error.gam$Country, Year = m.error.gam$variable, RF = m.error.rf$value, GAM = m.error.gam$value)
# summary(m.errors)
# ### Add some metadata?
# m.errors$GNI <- NA
# for(c in unique(m.errors$Country)) {
#         m.errors[m.errors$Country == c,"GNI"] <- country.details[country.details$Country == c,"GNI_classification"]
# } # for loop
#
# ### Do overall plot. try adding label to the most extreme cases
# m.errors$label <- paste(m.errors$Country, m.errors$Year, sep = "_")
# require("ggrepel")
# plot <- ggplot() + geom_point(aes(x = GAM, y = RF, fill = factor(GNI)), pch = 21, colour = "black", data = m.errors) +
#              geom_text_repel(aes(x = GAM, y = RF, label = label),
#                      data = m.errors[which(m.errors$RF > 40 | m.errors$RF < -40),], size = 1.5) +
#              geom_text_repel(aes(x = GAM, y = RF, label = label),
#                      data = m.errors[which(m.errors$GAM > 40 | m.errors$GAM < -40),], size = 1.5) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              geom_vline(xintercept = 0, linetype = "dotted", colour = "black") +
#              geom_hline(yintercept = 0, linetype = "dotted", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("GAM errors (%)") + ylab("RF errors (%)") + theme_classic()
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs")
# ggsave(plot = plot, filename = "plot_errors_GAMxRF.pdf", dpi = 300, height = 5, width = 7)
#
# summary(lm(GAM ~ RF, data = m.errors)) # Adjusted R-squared:  0.2823
# cor(na.omit(m.errors$GAM), na.omit(m.errors$RF), method = "spearman") #  0.4892
#
# ### Awesome, no do the same with MSW predictions :-)
# m.ens.rf <- melt(ens.rf, id.vars = "Country")
# m.ens.gam <- melt(ens.gam, id.vars = "Country")
# head(m.ens.rf) ; head(m.ens.gam)
# # Cbind
# m.preds <- data.frame(Country = m.ens.gam$Country, Year = m.ens.gam$variable, RF = m.ens.rf$value, GAM = m.ens.gam$value)
# # summary(m.preds)
# m.preds$GNI <- NA
# for(c in unique(m.preds$Country)) {
#         m.preds[m.preds$Country == c,"GNI"] <- country.details[country.details$Country == c,"GNI_classification"]
# } # for loop
#
# m.preds$label <- paste(m.preds$Country, m.preds$Year, sep = "_")
#
# ### ISSUES ITH THOSE GAM PREDICTIONS from: PERU? ABGOLA and LESOTHO & DJIBOUTI and Timor
# issues <- m.preds[m.preds$GAM > 5,"label"] # remove those
#
# require("ggrepel")
# plot <- ggplot() + geom_point(aes(x = (GAM), y = (RF), fill = factor(GNI)),
#                     pch = 21, colour = "black", data = m.preds[!(m.preds$label %in% issues),]) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("GAM predictions (t/pers./year)") + ylab("RF predictions (t/pers./year)") + theme_classic()
#
# ggsave(plot = plot, filename = "plot_predictions_GAMxRF.pdf", dpi = 300, height = 5, width = 7)
#
# summary(lm(GAM ~ RF, data = m.preds[!(m.preds$label %in% issues),])) # Adjusted R-squared:  0.42
# cor(na.omit(m.preds[!(m.preds$label %in% issues),]$GAM), na.omit(m.preds[!(m.preds$label %in% issues),]$RF), method = "spearman") # 0.9
#
# ### HUmm...super strong tendency of GAMs to over predict MSW for L and UM classes
# ### Compute % diff to RF predictions
# m.preds$GAM_diff <- ((m.preds$GAM)-(m.preds$RF)/(m.preds$RF))*100
# summary(m.preds$GAM_diff)
# # Rank
# m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("label","GNI","GAM_diff")]
# # Check the top countries which are mispredicted by GAMs
# unique(m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("Country")])
# m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("GNI")] # so mainly UM, LM and L

# ### Ok, need to check if these "mediocre" ensembles come from a particular GAM setup or if they are all crap
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# # Load all predictions and compute mean annual MSW per country
# files <- dir()[grep(paste("table_pred_", sep=""),dir())]
# # f <- "table_pred_UM_p3+p4+p5+p6_8.txt"
# res <- lapply(files, function(f) {
#             d <- read.table(f, sep = "\t", h = T)
#             d$GNI <- do.call(rbind,strsplit(as.character(f),"_"))[,3]
#             d$form <- do.call(rbind,strsplit(as.character(f),"_"))[,4]
#             d$n <- str_replace_all(do.call(rbind,strsplit(as.character(f),"_"))[,5],".txt","")
#             return(d)
#     }
# ) # eo lapply
# ddf <- bind_rows(res)
# dim(ddf); head(ddf)
# unique(ddf$Country) # 173
#
# ### Melt to put ALL MSW estimates as a vector and detect those that exceed 5
# m.ddf <- melt(ddf, id.vars = c("Country","GNI","form","n"))
# colnames(m.ddf)[c(5,6)] <- c("Year","MSW")
# summary(m.ddf)
# # Add an id for the GAM run based on GNI, n and form
# m.ddf$run <- paste(m.ddf$GNI, m.ddf$form, m.ddf$n, sep = "_")
# # identify those model runs that ruin the ensemble predictions: those where MSW exceed 10% of the max(y_org,na.rm=T)
# unique(m.ddf[m.ddf$MSW >= 5.16,"run"]) # 771
# bad.runs <- unique(m.ddf[m.ddf$MSW >= 5.16,"run"])
# ### And examine density distrib of formulae or n among those
# summary(factor(m.ddf[m.ddf$run %in% bad.runs,"form"]))
# summary(factor(m.ddf[m.ddf$run %in% bad.runs,"n"])) # not informative of course
#
# # Make a vector of the prediction tables that contain those bad.runs
# files2rm <- paste("table_pred_",bad.runs,".txt",sep="")
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# all.files <- dir()[grep("table_pred_",dir())]
# head(files2rm)
# head(all.files)
# files2keep <- all.files[!(all.files %in% files2rm)]
# length(files2rm) / length(all.files) # 20% of thrash
#
# ### Print the names of files2rm
# write.table(files2rm, "files2rm.txt", sep = "\t")
#
# ### Re compute GAM ensembles etc. based on those ^^
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# res <- lapply(files2keep, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) } ) # eo lapply
# # Rbind
# ddf <- bind_rows(res)
# # Melt to put years as vector
# m.ddf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
# colnames(m.ddf)[c(2:3)] <- c("Year","MSW")
# summary(m.ddf$MSW) # 0-5.15947
# # Compute mean
# ensemble.mean <- data.frame(m.ddf %>% group_by(Country,Year) %>% summarize(MSW = mean(MSW)) ) # eo ddf
# summary(ensemble.mean)
# # Dcast
# ensemble <- dcast(ensemble.mean, Country ~ Year)
# dim(ensemble) # OK
# ensemble
# colnames(ensemble)[c(2:27)] <- str_replace_all(as.character(colnames(ensemble)[c(2:27)]),"X","")
#
# ### 11/02/2020: Compute median + IQR (25th/75th percentiles) for Charlotte's talk at OSM
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# res <- lapply(files2keep, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) } ) # eo lapply
# # Rbind
# ddf <- bind_rows(res)
# # Melt to put years as vector
# m.ddf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
# colnames(m.ddf)[c(2:3)] <- c("Year","MSW")
# summary(m.ddf$MSW) # 0-5.15947
# # Compute median + IQR
# # IQR(m.ddf$MSW)
# # round(quantile(m.ddf$MSW)[2],3)
# # round(quantile(m.ddf$MSW)[4],3)
# median(m.ddf[m.ddf$Country == "Canada" & m.ddf$Year == "X2013","MSW"])
# IQR(m.ddf[m.ddf$Country == "Canada" & m.ddf$Year == "X2013","MSW"])
# round(quantile(m.ddf[m.ddf$Country == "Canada" & m.ddf$Year == "X2013","MSW"])[2],3)
# round(quantile(m.ddf[m.ddf$Country == "Canada" & m.ddf$Year == "X2013","MSW"])[4],3)
#
# ensemble.range <- data.frame(m.ddf %>% group_by(Country,Year) %>%
#         summarize(Median = median(MSW), IQR = IQR(MSW), Q2 = round(quantile(MSW)[2],3), Q4 = round(quantile(MSW)[4],3) )
# ) # eo ddf
# # Check
# summary(ensemble.range)
# # Save this as table for charlotte
# write.table(ensemble.range, file = "table_ranges_GAM_predictions_11_02_20.txt", sep = ";")
#
#
# ### Compute error of predictions to y_org
# error_ensemble <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ensemble$Country)) ) )
# error_ensemble[,1] <- unique(ensemble$Country)
# colnames(error_ensemble) <- c("Country", 1990:2015)
# # Calculate the predicted MSW for each country and get the error if measurements are available
# for(land in unique(ensemble$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ensemble$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
#       predictions <- ensemble[ensemble$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       error_ensemble[error_ensemble$Country == land,c(error.pos)] <- er
# } # for loop
# summary(error_ensemble)
#
# ### ANd derive pluriannual mean of errors
# m.error <- melt(error_ensemble, id.vars = "Country")
# dim(m.error)
# head(m.error)
# summary(m.error$value) # from -63% to +150%, but median error is -0.43% only !
# sum.error.ensembles <- data.frame(m.error %>% group_by(Country) %>% summarize(mean = mean(value, na.rm = T)) ) # eo ddf
# sum.error.ensembles[order(sum.error.ensembles$mean, decreasing = T),]
# # Ok exactly the same of course :D (you numb numb)
#
# ### 11/02/2020: Save table of errors for Charlotte
# write.table(error_ensemble, file = "table_errors_GAM_predictions_11_02_20.txt", sep = ";")
#
#
# ### Map this error
# n <- joinCountryData2Map(sum.error.ensembles, joinCode = "NAME", nameJoinColumn = "Country")
# # plot map
# pdf(file = paste("map_ensemble_error_GAM_filtered_13_12_19.pdf", sep = ""), width = 10, height = 6)
#     mapCountryData(n, nameColumnToPlot = "mean", mapTitle = "",
#     catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#     borderCol = "white", lwd = 0.1, addLegend = T)
# dev.off()
#
# ### Do the same but...per YEAR !
# for(yyy in as.character(c(1990:2015)) ) {
#
#         message(paste("Plotting errors for ", yyy, sep = ""))
#         er <- error_ensemble[,c("Country",yyy)]
#         colnames(er) <- c("Country","error")
#         # Map this error
#         n <- joinCountryData2Map(er, joinCode = "NAME", nameJoinColumn = "Country")
#         # plot map
#         pdf(file = paste("map_ensemble_error_GAM_filtered_",yyy,"_13_12_19.pdf", sep = ""), width = 10, height = 6)
#             mapCountryData(n, nameColumnToPlot = "error", mapTitle = "",
#             catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#             borderCol = "white", lwd = 0.1, addLegend = T)
#         dev.off()
#
# } # eo for loop
#
# ### Ok, save ensemble predictions of MSW and apply the same prediction fun above to RF
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# write.table(x = ensemble, "table_ensemble_predictions_GAM_filtered_13_12_19.txt", sep = "\t")
#
# ### -----------------------------------------------------------
#
# ### Re-compute ensemble across methods based on the corrected GAMs
#
# # First get the GAMs' ensemble mean
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# ens.gam <- read.table("table_ensemble_predictions_GAM_filtered_13_12_19.txt", sep = "\t", h = T)
# dim(ens.gam)
# summary(ens.gam)
# colnames(ens.gam)[c(2:27)] <- str_replace_all(as.character(colnames(ens.gam)[c(2:27)]),"X","")
# # Now get the RF ensembles
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/RF_predictions2")
# ens.rf <- read.table("table_ensemble_predictions_RF2_13_12_19.txt", sep = "\t", h = T)
# colnames(ens.rf)[c(2:27)] <- str_replace_all(as.character(colnames(ens.rf)[c(2:27)]),"X","")
#
# ### OK. Now, combine ens.gam & ens.rf to make cross-methods ensemble predictions
# dim(ens.gam); dim(ens.rf)
# head(ens.gam) ; head(ens.rf)
# # Melt
# m.ens.gam <- melt(ens.gam, id.vars = "Country")
# m.ens.rf <- melt(ens.rf, id.vars = "Country")
# head(m.ens.gam); head(m.ens.rf)
# ens.all <- data.frame(Country = m.ens.gam$Country, Year = m.ens.gam$variable, MSW = ((m.ens.gam$value+m.ens.rf$value)/2) )
# summary(ens.all)
# # And dcast
# ensembles <- dcast(ens.all, Country ~ Year)
# dim(ensembles); head(ensembles)
# summary(ensembles)
# # Save
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs")
# write.table(ensembles, file = "table_ensemble_pred_GAM+RF2_13_12_19.txt", sep = "\t")
#
# ### And derive ensemble errors to y_org for those
# error_ensemble_all <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ensembles$Country)) ) )
# error_ensemble_all[,1] <- unique(ensembles$Country)
# colnames(error_ensemble_all) <- c("Country", 1990:2015)
# for(land in unique(ensembles$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ensembles$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) )
#       predictions <- ensembles[ensembles$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       error_ensemble_all[error_ensemble_all$Country == land,c(error.pos)] <- er
# } # for loop
# summary(error_ensemble_all)
#
# ### 11/02/2020: Save table of errors for Charlotte
# write.table(error_ensemble_all, file = "table_errors_ensembles_GAM+RF_predictions_11_02_20.txt", sep = ";")
#
#
# # Derive pluriannual mean of errors
# m.error <- melt(error_ensemble_all, id.vars = "Country")
# dim(m.error)
# head(m.error)
# summary(m.error$value) # from -56% to +95%, but median error is -0.12% only !
# sum.error.ensembles <- data.frame(m.error %>% group_by(Country) %>% summarize(mean = mean(value, na.rm = T)) ) # eo ddf
# sum.error.ensembles[order(sum.error.ensembles$mean, decreasing = T),]
# ### Map this error
# n <- joinCountryData2Map(sum.error.ensembles, joinCode = "NAME", nameJoinColumn = "Country")
# # plot map
# pdf(file = paste("map_ensemble_error_GAM+RF2_13_12_19.pdf", sep = ""), width = 10, height = 6)
#     mapCountryData(n, nameColumnToPlot = "mean", mapTitle = "",
#     catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#     borderCol = "white", lwd = 0.1, addLegend = T)
# dev.off()
#
# ### Do the same per year
# for(yyy in as.character(c(1990:2015)) ) {
#         message(paste("Plotting errors for ", yyy, sep = ""))
#         er <- error_ensemble[,c("Country",yyy)]
#         colnames(er) <- c("Country","error")
#         # Map this error
#         n <- joinCountryData2Map(er, joinCode = "NAME", nameJoinColumn = "Country")
#         # plot map
#         pdf(file = paste("map_ensemble_error_GAM+RF2_",yyy,"_13_12_19.pdf", sep = ""), width = 10, height = 6)
#             mapCountryData(n, nameColumnToPlot = "error", mapTitle = "",
#             catMethod = "pretty", colourPalette = rev(brewer.pal(n=10,name="RdBu")), oceanCol = "white", missingCountryCol = "grey65",
#             borderCol = "white", lwd = 0.1, addLegend = T)
#         dev.off()
#
# } # eo for loop
#
# ### 13/12/19: From "sum.error.ensembles", compute mean errors per continent and GNI classes
# head(sum.error.ensembles)
# sum.error.ensembles$GNI <- NA
# sum.error.ensembles$Continent <- NA
# for(land in unique(sum.error.ensembles$Country)) {
#     sum.error.ensembles[sum.error.ensembles$Country == land,"GNI"] <- country.details[country.details$Country == land,"GNI_classification"]
#     sum.error.ensembles[sum.error.ensembles$Country == land,"Continent"] <- country.details[country.details$Country == land,"Continent"]
# } # eo for loop
# error.GNI <- data.frame(sum.error.ensembles %>% group_by(GNI) %>% summarize(mean = mean(mean,na.rm=T)) )
# #  GNI  mean
# #   H  0.0032
# #   L  5.2141
# #  LM -0.6715
# #  UM  0.3833
# error.cont <- data.frame(sum.error.ensembles %>% group_by(Continent) %>% summarize(mean = mean(mean,na.rm=T)) )
# #    Continent mean
# #        AF   0.8010
# #        AS   0.9447
# #       CAR -12.2692
# #        EU   0.3057
# #       NAM  -0.3913
# #        OC  -4.8061
# #       SAM   1.3306
#
#
#
# ### -----------------------------------------------------------
#
# ### Re-do the comparisons between GAMs filtered and RF
#
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/RF_predictions2")
# ens.rf <- read.table("table_ensemble_predictions_RF2_13_12_19.txt", sep = "\t", h = T)
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs/GAM_predictions")
# ens.gam <- read.table("table_ensemble_predictions_GAM_filtered_13_12_19.txt", sep = "\t", h = T)
# colnames(ens.gam)[c(2:27)] <- str_replace_all(as.character(colnames(ens.gam)[c(2:27)]),"X","")
# colnames(ens.rf)[c(2:27)] <- str_replace_all(as.character(colnames(ens.rf)[c(2:27)]),"X","")
#
# ### A°) Errors (along the 1:1 line)
# # Compute error based on ensemble GAM to y_org
# errors.gam <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ens.gam$Country)) ) )
# errors.gam[,1] <- unique(ens.gam$Country)
# colnames(errors.gam) <- c("Country", 1990:2015)
# for(land in unique(ens.gam$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ens.gam$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
#       predictions <- ens.gam[ens.gam$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       errors.gam[errors.gam$Country == land,c(error.pos)] <- er
# } # for loop
# summary(errors.gam)
# ### ANd derive pluriannual mean of errors
# m.error.gam <- melt(errors.gam, id.vars = "Country")
# dim(m.error.gam) ; head(m.error.gam)
# summary(m.error.gam$value) # -62.9422  -4.8168  -0.4420   0.5207   4.7968 150.2965
#
# ### Same for RF pred
# errors.rf <- as.data.frame(matrix(NA, ncol = 27, nrow = length(unique(ens.rf$Country)) ) )
# errors.rf[,1] <- unique(ens.rf$Country)
# colnames(errors.rf) <- c("Country", 1990:2015)
# for(land in unique(errors.rf$Country)) {
#       message( paste("Compute error of ensembles means for ", land, sep = "") )
#       country.details2 <- country.details[country.details$Country %in% ens.rf$Country,]
#       j <- as.numeric(rownames(country.details2[country.details2$Country == land,]))
#       error.pos <- names( which(!is.na(y_org[,j])) ) # names(error.pos)
#       predictions <- ens.rf[ens.rf$Country == land,error.pos]
#       er <- ( (predictions - y_org[error.pos,j])/y_org[error.pos,j] )*100
#       errors.rf[errors.rf$Country == land,c(error.pos)] <- er
# } # for loop
# summary(errors.rf)
# ### ANd derive pluriannual mean of errors
# m.error.rf <- melt(errors.rf, id.vars = "Country")
# dim(m.error.rf) ; head(m.error.rf)
# summary(m.error.rf$value) # -73.1528  -3.0076  -0.1627   0.4969   2.8931  92.3449
#
# ### 11/02/2020: Save table of errors for Charlotte
# write.table(errors.rf, file = "table_errors_RF_predictions_11_02_20.txt", sep = ";")
#
# ### From the distrbution of errors alone, hard to tell if RF outperfrom GAMs
#
# ### Cbind and look at the relationships along the 1:1 line
# head(m.error.rf)
# head(m.error.gam)
# m.errors <- data.frame(Country = m.error.gam$Country, Year = m.error.gam$variable, RF = m.error.rf$value, GAM = m.error.gam$value)
# summary(m.errors)
# ### Add some metadata?
# m.errors$GNI <- NA
# for(c in unique(m.errors$Country)) {
#         m.errors[m.errors$Country == c,"GNI"] <- country.details[country.details$Country == c,"GNI_classification"]
# } # for loop
#
# ### Do overall plot. try adding label to the most extreme cases
# m.errors$label <- paste(m.errors$Country, m.errors$Year, sep = "_")
# require("ggrepel")
# plot <- ggplot() + geom_point(aes(x = GAM, y = RF, fill = factor(GNI)), pch = 21, colour = "black", data = m.errors) +
#              geom_text_repel(aes(x = GAM, y = RF, label = label),
#                      data = m.errors[which(m.errors$RF > 40 | m.errors$RF < -40),], size = 1.5) +
#              geom_text_repel(aes(x = GAM, y = RF, label = label),
#                      data = m.errors[which(m.errors$GAM > 40 | m.errors$GAM < -40),], size = 1.5) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              geom_vline(xintercept = 0, linetype = "dotted", colour = "black") +
#              geom_hline(yintercept = 0, linetype = "dotted", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("GAM errors (%)") + ylab("RF errors (%)") + theme_classic()
# setwd("/net/kryo/work/fabioben/Inputs_plastics/outputs")
# ggsave(plot = plot, filename = "plot_errors_GAMxRF_filtered.pdf", dpi = 300, height = 5, width = 7)
#
# summary(lm(GAM ~ RF, data = m.errors)) # Adjusted R-squared:  0.26
# cor(na.omit(m.errors$GAM), na.omit(m.errors$RF), method = "spearman") # 0.4869
#
# ### Awesome, no do the same with MSW predictions :-)
# m.ens.rf <- melt(ens.rf, id.vars = "Country")
# m.ens.gam <- melt(ens.gam, id.vars = "Country")
# head(m.ens.rf) ; head(m.ens.gam)
# # Cbind
# m.preds <- data.frame(Country = m.ens.gam$Country, Year = m.ens.gam$variable, RF = m.ens.rf$value, GAM = m.ens.gam$value)
# # summary(m.preds)
# m.preds$GNI <- NA
# for(c in unique(m.preds$Country)) {
#         m.preds[m.preds$Country == c,"GNI"] <- country.details[country.details$Country == c,"GNI_classification"]
# } # for loop
#
# m.preds$label <- paste(m.preds$Country, m.preds$Year, sep = "_")
#
# require("ggrepel")
# plot <- ggplot() + geom_point(aes(x = GAM, y = RF, fill = factor(GNI)), pch = 21, colour = "black", data = m.preds) +
#             geom_text_repel(aes(x = GAM, y = RF, label = label),
#                     data = m.preds[which(m.preds$GAM > 3 & m.preds$RF < 1.5),], size = 1.5) +
#              geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
#              scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
#              xlab("GAM predictions (t/pers./year)") + ylab("RF predictions (t/pers./year)") + theme_classic()
#
# ggsave(plot = plot, filename = "plot_predictions_GAMxRF_filtered.pdf", dpi = 300, height = 5, width = 7)
#
# ### MONTENEGRO ISSUE !
#
# summary(lm(GAM ~ RF, data = m.preds)) # Adjusted R-squared:  0.53
# # W/out montenegro?
# summary(lm(GAM ~ RF, data = m.preds[-which(m.preds$Country == "Montenegro"),])) # R-squared:  0.8874 !
# ### ANd correaltion?
# cor(na.omit(m.preds$GAM), na.omit(m.preds$RF), method = "spearman") # 0.92
# cor(na.omit(m.preds[-which(m.preds$Country == "Montenegro"),]$GAM),
#     na.omit(m.preds[-which(m.preds$Country == "Montenegro"),]$RF), method = "spearman")
# # 0.923
#
# ### HUmm...super strong tendency of GAMs to over predict MSW for L and UM classes
# ### Compute % diff to RF predictions
# m.preds$GAM_diff <- ((m.preds$GAM)-(m.preds$RF)/(m.preds$RF))*100
# summary(m.preds$GAM_diff)
# # Rank
# m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("label","GNI","GAM_diff")]
# # Check the top countries which are mispredicted by GAMs
# unique(m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("Country")])
# m.preds[order(abs(m.preds$GAM_diff), decreasing = T),c("GNI")]

