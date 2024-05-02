
##### 16/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
# - Recycle parts of Kevin's neat scripts ("Transform&scale.R") to get the appropriate predictor variables (PVs) tables
# - Load PVs table, examine their values' distribution and map their spatial distribution
# - Tune some RF models per income (GNI) classes (mtry parameters, number and identity of PVs, ntrees etc.)

### Last update: 17/01/23

### ------------------------------------------------------------------------------------------------------------------------------------------------

library("tidyverse")
library("reshape2")
library("scales")
library("randomForest")
library("RColorBrewer")
library("viridis")
library("ggsci")
library("ggthemes")
library("scales")
library("wesanderson")

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

### PART I: EXAMINING DATA DISTRIBUTION AND PREPARING SCALED DATA FOR MACHINE LEARNING MODELS (GAMs, RF and NNET) 

### A°) Get data sent by Charlotte L., remove outliers, scale them etc. Basically: prepare data for the models
setwd(paste(WD,"/data/", sep = "")) #; dir()

# Load datasets, check dimensions, delete useless stuff etc.
MSW_collected_UN <- read.csv("MSW_collected_corrected_14_01_23.csv", na.strings = c("NA"), stringsAsFactors = F) # MSW = Municipal solid waste
colnames(MSW_collected_UN) <- c("Country", 1990:2019) # adjust colnames
# dim(MSW_collected_UN) ; str(MSW_collected_UN)

young_pop <- read.csv("young_pop.csv", na.strings = c("NA"), stringsAsFactors = F)
# dim(young_pop) ; str(young_pop); summary(young_pop)
colnames(young_pop) <- c("Country", 1990:2019)

share_urb_pop <- read.csv("share_urb_pop.csv", na.strings = c("NA"), stringsAsFactors = F)
# dim(share_urb_pop) ; str(share_urb_pop) ; summary(share_urb_pop)
colnames(share_urb_pop) <- c("Country", 1990:2019)

elec_acc <- read.csv("elec_acc.csv", na.strings = c("NA"), stringsAsFactors = F)
# dim(elec_acc) ; str(elec_acc) ; summary(elec_acc)
colnames(elec_acc) <- c("Country", 1990:2019)

GDP_per_capita <- read.csv("GDP.csv", na.strings = c("NA"), stringsAsFactors = F)
# dim(GDP_per_capita) ; str(GDP_per_capita) ; summary(GDP_per_capita)
colnames(GDP_per_capita) <- c("Country", 1990:2019)

energy_consumption <- read.csv("energy_consumption.csv", stringsAsFactors = F)
# dim(energy_consumption) ; str(energy_consumption) ; summary(energy_consumption)
colnames(energy_consumption) <- c("Country", 1990:2019)

greenhouse_gas_pP <- read.csv("greenhouse_gas.csv", stringsAsFactors = F)
# dim(greenhouse_gas_pP) ; str(greenhouse_gas_pP) ; summary(greenhouse_gas_pP)
colnames(greenhouse_gas_pP) <- c("Country", 1990:2019)

country.details <- read.csv("country_details.csv", stringsAsFactors = F)[,c(1,8)]
# dim(country.details) 

### Detect & remove outliers in MSW_collected_UN (target variable)
### NOTE: I thought the criterion was a bit too strict, especially for cases like Austria where I thought reliable data from the 90s was being thrown out, so I tried other criteria (see https://statsandr.com/blog/outliers-detection-in-r/#grubbss-test) but they appeared to be even strciter sometimes, like Hampel's filter. So for now I chose to stick to the current boxplot method based on the 1.5*IQR criterion. All observations outside of the following interval will be considered as potential outliers: I = [q0.25- 1.5*IQR; q0.75- 1.5*IQR]
### Z-scores and Chauvenet's criterion were not applied because tey relate on mean/sd of normally distributed data.

# Count # of no NaN values before
# sum(!is.na(MSW_collected_UN[,2:length(MSW_collected_UN)])) # 1900

removed <- 0
n <- nrow(young_pop) - 1 #; n
# i <- 14 # for testing

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
cat("Removed",removed,"outliers")
# Count # of no NaNs after
# sum(!is.na(MSW_collected_UN[,2:length(MSW_collected_UN)])) # 1823 --> 77 points out of 1900 were turned into NaN (4.05% of the data were removed)


### Extract, transform & scale data
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
# summary(y_org)

### Plot distributions of y + 'p' variables 
setwd(paste(WD,"/plots/", sep = ""))
# MSW
plot.y <- ggplot(data = melt(y), aes(x = value)) +
    geom_histogram(binwidth = .2, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Municipal solid waste (MSW) collected\nlog(tons per person)") + ylab("Count")
# Save plot
ggsave(plot = plot.y, "plot_distrb_MSW_logged.pdf", dpi = 300, width = 5, height = 5)

# p1 = elec_acc
plot.p1 <- ggplot(data = melt(p1), aes(x = value)) +
    geom_histogram(binwidth = 5, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Electrecity access (%)") + ylab("Count")
ggsave(plot = plot.p1, "plot_distrb_p1_elec_access.pdf", dpi = 300, width = 5, height = 5)

# p2 = energy_consumption
plot.p2 <- ggplot(data = melt(p2), aes(x = value)) +
    geom_histogram(binwidth = .25, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Energy consumption per person\n(log(kg of oil equivalents))") + ylab("Count")
ggsave(plot = plot.p2, "plot_distrb_p2_energy_consumpt_logged.pdf", dpi = 300, width = 5, height = 5)

# p3 = Gross GDP
plot.p3 <- ggplot(data = melt(p3), aes(x = value)) +
    geom_histogram(binwidth = .5, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Gross GDP  per capita\nlog(USD)") + ylab("Count")
ggsave(plot = plot.p3, "plot_distrb_p3_GDP_logged.pdf", dpi = 300, width = 5, height = 5)


# p4 = GHG emissions
plot.p4 <- ggplot(data = melt(p4), aes(x = value)) +
    geom_histogram(binwidth = .5, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Total GHG emissions\nlog(CO2 equivalents)") + ylab("Count")
ggsave(plot = plot.p4, "plot_distrb_p4_GHG_logged.pdf", dpi = 300, width = 5, height = 5)


# p5 = urban pop
plot.p5 <- ggplot(data = melt(p5), aes(x = value)) +
    geom_histogram(binwidth = 5, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Urban population (%)") + ylab("Count")
ggsave(plot = plot.p5, "plot_distrb_p5_urb_pop.pdf", dpi = 300, width = 5, height = 5)

# p6 = young pop
plot.p6 <- ggplot(data = melt(p6), aes(x = value*100)) +
    geom_histogram(binwidth = 1, colour = "black", fill = "white") +
    geom_vline(aes(xintercept = mean(value*100, na.rm = T)), color = "red", linetype = "dashed", size = 1) +
    xlab("Population under 14 yo (%)") + ylab("Count")
ggsave(plot = plot.p6, "plot_distrb_p6_young_pop.pdf", dpi = 300, width = 5, height = 5)


### Create dataset with only complete data
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
# dim(te); summary(te) # Ok gut gut

# Scale data for RF
max <- c(max(y_complete),max(p1),max(p2),max(p3),max(p4),max(p5),max(p6))
min <- c(min(y_complete),min(p1),min(p2),min(p3),min(p4),min(p5),min(p6))
# min
# max
scaled <- as.data.frame(scale(te, center = min, scale = max - min)) # scale...rather "range"
scaled$country <- countries_complete
scaled$year <- year_complete
scaled$GNI <- GNI_complete

# List of countries present for now: 
# country.details[order(unique(country.details$Country)),"Country"]
# MSW_collected_UN[order(unique(MSW_collected_UN$Country)),"Country"]
# unique(scaled$country)[order(unique(scaled$country))]
# setdiff(unique(MSW_collected_UN$Country), unique(scaled$country) )
# summary(as.factor(scaled$country))

# To check countries individually: 
# scaled[scaled$country == "Samoa",]

### Check completion of PVs table
# dim(scaled) ; summary(scaled) ; unique(scaled$GNI)
# Merge L and LM GNI categories
# summary(factor(scaled$GNI)) # L == 35 points, 153 = 169, UM = 294, H = 882
#scaled[scaled$GNI == "LM","GNI"] <- rep("L", 169)
# Rename UM 
#scaled[scaled$GNI == "UM","GNI"] <- rep("M", 294)
# summary(factor(scaled$GNI))
# Tally n obs per country and then per GNI
# tally1 <- data.frame(scaled %>% group_by(GNI) %>% summarize(n = n()) ) ; tally1
tally2 <- data.frame(scaled %>% group_by(country,GNI) %>% summarize(n = n()) ) ; tally2
# Check
# tally1
tally2[order(tally2$n, decreasing = T),] # clearly shows that H countries provide more data (and probably more accurate)
# M/L countries are mixed though

### Create datasets for every income classes
scaled.H <- scaled[scaled$GNI == "H",]
scaled.UM <- scaled[scaled$GNI == "UM",]
scaled.LM <- scaled[scaled$GNI == "LM",]
scaled.L <- scaled[scaled$GNI == "L",]
# dim(scaled.H); dim(scaled.UM); dim(scaled.LM); dim(scaled.L)

### Create matrix to fill with errors of predict for each country (for later)
error_country <- matrix(NA, nrow = dim(country.details)[1], ncol = nyears)
rownames(error_country) <- country.details[,1]
colnames(error_country) <- 1990:2019
error_country <- as.data.frame(error_country)
# dim(error_country)
# head(error_country)

### Save datasets above. Since I have not permission to write/save stuff on Kevin's session, moving to my own
setwd(paste(WD,"/data/complete_data/", sep = "")) ; dir()

write.table(x = scaled.H, file = "data_scaled_H.txt", row.names = F, sep = "\t")
write.table(x = scaled.UM, file = "data_scaled_UM.txt", row.names = F, sep = "\t")
write.table(x = scaled.LM, file = "data_scaled_LM.txt", row.names = F, sep = "\t")
write.table(x = scaled.L, file = "data_scaled_L.txt", row.names = F, sep = "\t")
save(error_country, file = "table_errors_country.Rdata") # save this one R object for now
save(min, file = "vector_min_values_PVs.Rdata") 
save(max, file = "vector_max_values_PVs.Rdata") 

### Plot some histograms to assess distribution (facet per GNI class)

setwd(paste(WD,"/plots/", sep = ""))

# Choose palettes at https://thefeeney.netlify.com/post/color-palettes-in-r/  #TeamZissou
colnames(scaled)  # summary(scaled) ; levels(factor(scaled$GNI))
plot <- ggplot(scaled, aes(x=y, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Collected MSW") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_MSW.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p1, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Access to electricity (%)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_elec_access.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p2, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Energy consumption (log)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_log_energ_consum.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p3, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("GDP per capita (log)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_log_GDP.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p4, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("GHG emissions (log)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_log_GHG.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p5, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Urban population (%)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_urban_pop.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p5, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Urban population (%)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_urban_pop.pdf", dpi = 300, height = 6, width = 4)

plot <- ggplot(scaled, aes(x=p6, fill = factor(GNI))) + geom_histogram(colour = "black") +
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        facet_grid(factor(GNI) ~ ., scales = "free_y") +
        xlab("Young population (%)") + ylab("Count")
ggsave(plot = plot, filename = "plot_distrib_young_pop.pdf", dpi = 300, height = 6, width = 4)


### ----------------------------------------------------------------

### 16/01/23: Examine distribution of the initial 4 GNI classes in parameter space based on PCA
### Aims to assess how much L and LM overlap to see if it's ok to merge them.
require("vegan")
require("FactoMineR")

# Start from 'scaled'
# colnames(scaled); summary(factor(scaled$GNI))

# Perform PCA with the 6 predictors (scale.uni = T)
pca <- PCA(X = scaled[,c(2:7)], ncp = 4, graph = F, scale.unit = TRUE)
# summary(pca) # Keep first 3 axes
pc1 <- "53.4"; pc2 <- "24.5" ; pc3 <- "10.1"
# head(pca$ind$coord)
scaled$PC1 <- pca$ind$coord[,1]
scaled$PC2 <- pca$ind$coord[,2]
scaled$PC3 <- pca$ind$coord[,3]
# summary(scaled)

### Draw 2 kinds of plots: Violin plots of the distribution of PCs per GNI & GNI2 + dot plot in PC space
# levels(factor(scaled$GNI)) ; levels(factor(scaled$GNI2))
plot1 <- ggplot() + geom_point(aes(x = PC1, y = PC2, fill = factor(GNI)), data = scaled, pch = 21, colour = "black", alpha = 0.7) + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + 
            xlab(paste("PC1 (",pc1,"%)", sep = "")) + ylab(paste("PC2 (",pc2,"%)", sep = "")) +
            geom_hline(yintercept = 0, linetype = "dashed") + geom_vline(xintercept = 0, linetype = "dashed") + 
            theme_bw()
setwd(paste(WD,"/plots/", sep = ""))
ggsave(plot = plot1, filename = "plot_GNI_PC1+2.pdf", height = 5, width = 7, dpi = 300)


### Violin plots per PCs -> first mekt to have PC as factor
m.scaled <- melt(scaled[,c(8:13)], id.vars = c("country","year","GNI"))
# summary(m.scaled)
violin1 <- ggplot(aes(x = factor(GNI), y = value, fill = factor(GNI)), data = m.scaled) + geom_violin(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + theme_bw() + 
            ylab("Distribution") + xlab("GNI") + facet_wrap(factor(variable)~.)
ggsave(plot = violin1, filename = "violin_GNI_PC1+2+3.pdf", height = 5, width = 8, dpi = 300)


### Interesting...check distrib of scaled PVs per GNI 
m.scaled2 <- melt(scaled[,c(1:10)], id.vars = c("country","year","GNI"))
# summary(m.scaled2)
violinPVs <- ggplot(aes(x = factor(GNI), y = value, fill = factor(GNI)), data = m.scaled2) + geom_violin(colour = "black") + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + theme_bw() + 
            ylab("Distribution") + xlab("GNI") + facet_wrap(factor(m.scaled2$variable), ncol = 3, scales = "free")
#
ggsave(plot = violinPVs, filename = "violin_GNI_PVs.pdf", height = 8, width = 10, dpi = 300)


### So, we're trying to assess the relationships between y and basically PC1 (which summarizes the variance of all PVs) and whether LM and L can be merged. Plot these variables per GNI

plot1 <- ggplot(aes(x = PC1, y = y, fill = factor(GNI)), data = scaled) + geom_point(colour = "black", pch = 21) + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + theme_bw() + 
            ylab("MSW collected (log)") + xlab("PC1") + facet_wrap(factor(scaled$GNI), ncol = 2, scales = "free")
ggsave(plot = plot1, filename = "plot_y~PC1_GNI_freescales.pdf", height = 6.5, width = 8, dpi = 300)

plot2 <- ggplot(aes(x = PC2, y = y, fill = factor(GNI)), data = scaled) + geom_point(colour = "black", pch = 21) + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + theme_bw() + 
            ylab("MSW collected (log)") + xlab("PC2") + facet_wrap(factor(scaled$GNI), ncol = 2, scales = "free")
ggsave(plot = plot2, filename = "plot_y~PC2_GNI_freescales.pdf", height = 6.5, width = 8, dpi = 300)

plot3 <- ggplot(aes(x = PC3, y = y, fill = factor(GNI)), data = scaled) + geom_point(colour = "black", pch = 21) + 
            scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5")) + theme_bw() + 
            ylab("MSW collected (log)") + xlab("PC3") + facet_wrap(factor(scaled$GNI), ncol = 2, scales = "free")
ggsave(plot = plot3, filename = "plot_y~PC3_GNI_freescales.pdf", height = 6.5, width = 8, dpi = 300)


### Need to identify this weird group of points with relatively high Y at low PC1 values in L income class
# y > 0.6 & PC1 < -0.7
# scaled[which(scaled$y > 0.6 & scaled$PC1 < -6 & scaled$GNI == "L"),] # All the data from Niger 


### Examine colinearity between PVs and target variable - overall and per GNI class
library("GGally")
ggpairs(data = scaled[,c(1:7)]) # quick plot to examine overall patterns
### Potential conflict between p3xp4 and p1xp6
res.cor <- round(cor(scaled[,c(1:7)], method = "spearman"), 3)
# On total data, p3 & p4 show a 0.895 cor coeff
# And per GNI class? 
# H
res.cor <- round(cor(scaled.H[,c(1:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear
# UM
res.cor <- round(cor(scaled.UM[,c(1:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear
# LM
res.cor <- round(cor(scaled.LM[,c(1:7)], method = "spearman"), 3)
### --> p3 x p4 signif collinear + p1 x p2 ; p1 and p5 are nearly there too
# L
res.cor <- round(cor(scaled.L[,c(1:7)], method = "spearman"), 3)
### --> many many PVs are correlated...especially p1xp5xp4xp3...


### ----------------------------------------------------------------

### PART II: PRELIMINARY RF OPTIMIZATION AND PARAMS SELECTION (mtry param, plus playing around with the PV to assess change in MSE and r2 etc.)

### 16/01/20: To simply re-load the results from above:
setwd(paste(WD,"/data/complete_data/", sep = ""))
scaled.H <- read.table("data_scaled_H.txt", h = T, sep = "\t")
scaled.UM <- read.table("data_scaled_UM.txt", h = T, sep = "\t")
scaled.LM <- read.table("data_scaled_LM.txt", h = T, sep = "\t")
scaled.L <- read.table("data_scaled_L.txt", h = T, sep = "\t")
error_country <- get(load("table_errors_country.Rdata"))
min <- get(load("vector_min_values_PVs.Rdata"))
max <- get(load("vector_max_values_PVs.Rdata"))

### B°) Try tuning some RF model
require("randomForest")
# ?randomForest
# So, first, need to get an approximation for the 'mtry' parameter (at this stage, it was == 3 for Kevin)
# mtry = Number of variables randomly sampled as candidates at each split.  Note that the default values are different for
#        classification (sqrt(p) where p is number of variables in ‘x’) and regression (p/3)
# For that use the tuneRF function
# ?tuneRF
# Starting with the default value of mtry, search for the optimal value (with respect to Out-of-Bag error estimate) of mtry for randomForest.

setwd(paste(WD,"/plots/tuneRF_mtry/", sep = ""))

pb <- txtProgressBar(min = 0, max = 100, style = 3)
best.mtry <- NA # empty vector
for(i in 1:100) {
   pos <- sample(1:10, nrow(te), replace=T)
   invisible(capture.output(tuned <- tuneRF(scaled[,c(2:7)],scaled[,1], mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
   best.mtry[i] <- tuned[which.min(tuned[,2]),1]
   setTxtProgressBar(pb, i)
}
pdf(file = "best.mtry.initial_total_dataset.pdf", width = 5, height = 4)
       barplot(table(best.mtry), xlab = "mtry value", ylab = 'Number of results')
dev.off()

# mtry = 3 is the most frequent choice for total dataset. Do the same for each GNI class

### High income
best.mtry.H <- NA # empty vector
pb <- txtProgressBar(min = 0, max = 100, style = 3)
for(i in 1:100){
  pos <- sample(1:10, nrow(te), replace=T)
  invisible(capture.output(tuned <- tuneRF(scaled[scaled$GNI == "H",c(2:7)],scaled[scaled$GNI == "H",1], 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F) ))
  best.mtry.H[i] <- tuned[which.min(tuned[,2]),1]
  setTxtProgressBar(pb, i)
}
pdf(file = "best.mtry.initial_H.pdf", width = 5, height = 4)
      barplot(table(best.mtry.H), xlab = "mtry value", ylab = 'Number of results')
dev.off()


### U-Medium income
best.mtry.UM <- NA # empty vector
pb <- txtProgressBar(min = 0, max = 100, style = 3)
for(i in 1:100){
  pos <- sample(1:10, nrow(te), replace=T)
  invisible(capture.output(tuned <- tuneRF(scaled[scaled$GNI == "UM",c(2:7)],scaled[scaled$GNI == "UM",1], 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.UM[i] <- tuned[which.min(tuned[,2]),1]
  setTxtProgressBar(pb, i)
}
pdf(file = "best.mtry.initial_UM.pdf", width = 5, height = 4)
      barplot(table(best.mtry.UM), xlab = "mtry value", ylab = 'Number of results')
dev.off()


### LM income
best.mtry.LM <- NA # empty vector
pb <- txtProgressBar(min = 0, max = 100, style = 3)
for(i in 1:100){
  pos <- sample(1:10, nrow(te), replace=T)
  invisible(capture.output(tuned <- tuneRF(scaled[scaled$GNI == "LM",c(2:7)],scaled[scaled$GNI == "LM",1], 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.LM[i] <- tuned[which.min(tuned[,2]),1]
  setTxtProgressBar(pb, i)
}
pdf(file = "best.mtry.initial_LM.pdf", width = 5, height = 4)
      barplot(table(best.mtry.LM), xlab = "mtry value", ylab = 'Number of results')
dev.off()


### Low income
best.mtry.L <- NA # empty vector
pb <- txtProgressBar(min = 0, max = 100, style = 3)
for(i in 1:100) {
  pos <- sample(1:10, nrow(te), replace=T)
  invisible(capture.output(tuned <- tuneRF(scaled[scaled$GNI == "L",c(2:7)], scaled[scaled$GNI == "L",1], 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.L[i] <- tuned[which.min(tuned[,2]),1]
  setTxtProgressBar(pb, i)
}
pdf(file = "best.mtry.initial_L.pdf", width = 5, height = 4)
      barplot(table(best.mtry.L), xlab = "mtry value", ylab = 'Number of results')
dev.off()

### Conclusion of mtry tests
# H : mtry == 4
# UM : mtry == 3
# LM : mtry == 2
# L : mtry == 2


### ----------------------------------------------------------------

### 16/01/2023

# Besides using a simple architecture (mtry params above), using a minimal number of PV can help to prevent overfitting. We ran several tests to select which PVs are most important in the model creation. We use the following techniques to assess the importance of each PV: removal, random permutation and univariate runs. Following we describe these procedures:
# Removal: We run both MLM while sequentially removing one parameter at a time. By removing one parameter from the training data we assess the importance of this variable in the model creation. For each removed PV, we trained ten models on the same randomly sampled 90% of the data. This was repeated nine more times using newly sampled training data to avoid a bias in the training data. Thus, a total of 100 models were created. We then remove the parameter that has the least effect on the model performance (MSE). This process was repeated with the remaining PVs until only one PV was left. 
# Random permutation: Analogue to the removal process we run both models while randomly permuting one parameter at a time. This means that for one PV, the values were randomly replaced with values from within the same PV. If the PV is important in the model creation this is expected to create larger errors than for less important PVs. For each randomly permuted PV, we trained ten models on the same randomly sampled 90% of the data. This was repeated nine more times using newly sampled training data to avoid a bias in the training data. We then remove the parameter that has the least effect on the model performance (MSE). This process was repeated with the remaining PVs until only one PV was left.
# Univariate: We ran the model using only one PV at a time and calculated the MSE. Here, we expect that PVs that are important for the model creation have lower MSE values.

### Implement wrapper functions Kevin defined for variables and parameters selection
run_model <- function(scaled.temp, income, folder.addition, formula, run_mode) {
    
          pb <- txtProgressBar(min = 0, max = 100, style = 3) # verbose

          MSE.RF.old <- 999
          MSE.RF <- NA
          
          # Choose mtryvalue according to income factor and tests above
          if(income == "H") {
              mtryvalue <- 4
          } else if(income == "UM") {
              mtryvalue <- 3
          } else if(income == "LM") {
              mtryvalue <- 2
          } else if(income == "L") {
              mtryvalue <- 2
          } # eo else if loop
          
          # for k values in 1:100
          for(k in 1:100) {
              
                  # Create train and test dataset
                  pos <- sample(1:10, nrow(scaled.temp), replace=T)
                  while( length(table(pos)) != 10 ) { # Prevent that one number is not represented in small samples
                          # Split the data into 10 groups, randomly (9/10 training, 1/10 testing)
                          pos <- sample(1:10, nrow(scaled.temp), replace=T)
                  } # eo while loop
                  
                  # Filter out the group that will be used for testing ("unseen data")
                  trainRF <- scaled.temp[pos!=10,]
      
                  # Create a RF model
                  RF <- randomForest(formula, trainRF, mtry = mtryvalue)
      
                  MSE.RF.cv <- NA
                  
                  # split in 1/10 
                  for(i in 1:10) {
                          # Create train and test dataset
                          testRF <- scaled.temp[pos==i, ]  
                          # Prediction using neural network
                          temp <- data.frame(testRF[,-c(1)])
                          if(dim(temp)[2]==1){ 
                                  colnames(temp) <- colnames(testRF)[2] 
                          } # eo if loop
                          predict_testRF <- predict(RF, temp)
                          # Unscale for CV
                          predict_testRF.cv <- exp((predict_testRF * (max[1] - min[1])) + min[1])
                          testRF.cv <- exp((testRF$y * (max[1] - min[1])) + min[1])
                          # Prevent negative values
                          predict_testRF[which(predict_testRF < 0)] <- 0
                          # Calculate Mean Square Error (MSE)
                          MSE.RF.cv[i] = (sum((testRF.cv - predict_testRF.cv)^2) / nrow(testRF))
                  } # eo for i 1:10 for cross validation 
      
                  # Compute mean MSE
                  MSE.RF[k] <- mean(MSE.RF.cv)
      
                  # Save the RF if the error of unseen data isn't the largest
                  if( which.max(MSE.RF.cv) != 10) {
                      
                      if(MSE.RF[k] < MSE.RF.old) {
                          
                            temp <- data.frame(scaled.temp[,-c(1)])
                          
                            if( dim(temp)[2] == 1 ) {
                                colnames(temp) <- colnames(testRF)[2] 
                            } # eo if loop
                            
                            # Compute r2
                            pred <- predict(RF, temp)
                            pred <- exp((pred * (max[1] - min[1])) + min[1])
                            pred[which(pred<0)] <- 0
                            measure <- exp((scaled.temp$y * (max[1] - min[1])) + min[1])
                            r2 <- summary(lm(pred ~ measure))$r.squared
                            ### Save RF model for predicting later
                            if( run_mode == "remove") {
                                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/remove/")
                            } else if(run_mode == "shuffle") {
                                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/shuffle/")
                            } else {
                                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/univariate/")
                            } # eo if else loop
      
                            frmla <- str_replace_all(as.character(formula), " ", "")[3]
                            save(RF, file = paste("data_MSE_RF_optim_best_",run_mode,"_p-",l,"_",frmla,"_GNI-",income,".Rdata", sep = "") )
                            
                            # Save CV plot as .pdf 
                            pdf( paste("plots_MSE_optim_RF_best_model_",run_mode,"_p-",l,"_",frmla,"_GNI-",income,".pdf", sep = "") , width = 10, height = 5)
                                    par(mfrow=c(1,2))
                                    plot(measure, pred, main = paste0("MSE: ", MSE.RF[k]), 
                                        col = ifelse(pos==10, "red", "black"), xlab = "Measured", ylab = "Predicted from RF")
                                    #title(main = "Cross-validation", xlab = "Measured", ylab = "Predicted from RF") 
                                    abline(0,1)
                                    mtext(paste("r^2:",r2))
                                    boxplot(MSE.RF.cv, main = "MSE values from CV")
                                    points(rep(1,10) ,MSE.RF.cv, col = c(rep("grey",9),"red"))
                            dev.off()
          
                            MSE.RF.old <- MSE.RF[k]
                            
                    } # eo if else loop
                        
                 } # eo if else loop
                    
                 setTxtProgressBar(pb,k)
                  
        } # eo for loop
  
        close(pb)
        return(MSE.RF)
  
} # eo run_model FUN


# Function to run the parameter selection automatically --> wrapper for run_model
# Uses a string containing all variables, the data, the income class and the run mode (removal, shuffle or univariate)
# For testing :
 # p <- params.nr
 # dat <- scaled.L
 # income <- "L"
 # run_mode <- "shuffle"
run_model_wrapper <- function(p, dat, income, run_mode) {
    
              cat("Starting with parameters:", params.name, "\nTime: ", Sys.time() )
              MSE <- matrix(NA,100,length(p))
  
              # Choose mtryvalue according to income factor and tests above
              if(income == "H") {
                  mtryvalue <- 4
              } else if(income == "UM") {
                  mtryvalue <- 3
              } else if(income == "LM") {
                  mtryvalue <- 2
              } else if(income == "L") {
                  mtryvalue <- 2
              } # eo else if loop
              
              cnt <- 1
              
              l <- length(p) # to change the names of the files accordingly
              
              # Loop through each remaining parameter
              # i <- 1 # for testing
              for(i in p) {
                  
                      if( run_mode == "remove" ) {
                          message(paste("\nNow running without: ", params.original[i], sep = "" ))
                          formula <- as.formula(paste("y ~", paste(params.name[-cnt], collapse = " + " )) ) # formula
                          MSE[,cnt] <- run_model(scaled.temp = dat[,-c(i+1,params_out+1,8:10) ] , 
                                          paste0(income,"_",paste0(params.original[params_out],collapse="_")),
                                          formula = formula,
                                          run_mode = "remove")
                      } # eo if loop - run_mode == "remove"
    
                      # Shuffle all parameters if mode = shuffle
                      if( run_mode == "shuffle" ) {
                          message(paste("\nNow shuffling: ", params.original[i], sep = "") )
                          dat.temp <- dat
                          dat.temp[,i+1] <- sample(dat[,i+1]) # Shuffle parameter p 
                          formula <- as.formula(paste("y ~",paste(params.name,collapse=" + ")))
                          MSE[,cnt] <- run_model(dat.temp[,-c(params_out+1,8:10)],
                                          paste0(income, "_", paste0(params.original[params_out], collapse="_")),
                                          formula = formula,
                                          run_mode = "shuffle")
                      } # eo if loop - run_mode == "shuffle"
    
                      # Remove all other parameters if univariate
                      if( run_mode == "univariate" ) {
                          message(paste("\nNow running with only: ", params.original[i], sep = "") )
                          formula <- as.formula(paste("y ~",params.name[i]))
                          MSE[,cnt] <- run_model(dat[,c(1,i+1)],
                                          paste0(income, "_", paste0(params.original[params_out], collapse="_")),
                                          formula = formula,
                                          run_mode = "univariate")
                      } # eo if loop - run_mode == "univariate"
    
                      cnt <- cnt+1 # moving to next col of the MSE matrix 
                      
               } # eo for loop
  
            # Save MSE matrix
            if( run_mode == "remove") {
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/remove/")
            } else if(run_mode == "shuffle") {
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/shuffle/")
            } else {
                setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/univariate/")
            } # eo if else loop
      
            #setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/")
            write.csv(x = MSE, file = paste("data_MSE_RF_optim_",run_mode,"_p-",l,"_GNI-",income,".csv", sep = "") )
  
            # Create a plot that shows the MSE of each model.
            # --> model with the least change in MSE is the one where removing the variable has the least effect -> unimportant PV
            mean <- colMeans(MSE)
            mini <- apply(MSE,2,min)
            sd <- apply(MSE,2,sd)
            # Plot
            pdf( paste("plot_MSE_RF_optim_",run_mode,"_p-",l,"_GNI-",income,".pdf", sep = ""),
                    width = 11, height = 7)
                    par(xpd = T,mfrow = c(1,2))
                    barCenters <- barplot(mean[order(mean)], names = params.name[order(mean)])
                    barplot(mini[order(mean)], add = T, col = "lightgrey",names="")
                    arrows(barCenters, (mean-sd)[order(mean)], barCenters, (mean+sd)[order(mean)],
                        lwd = 1.5, angle = 90, code = 3, length = 0.05)
                    boxplot(MSE[,order(mean)], names = params.name[order(mean)])
            dev.off()
  
            # Remove the parameter that had the least influence on MSE
            remove <- which.min(colMeans(MSE))
            message( paste("\nRemoved parameter: ", params.name[remove], ". Now going to re-run with the remaining parameters...\n") )
  
            # Update the parameter list for the next run
            if( is.null(params_out) ) { 
                params_out <<- remove
            } else { 
                params_out <<- c(params_out, params.nr[remove] )
            } # eo if else loop
            
            params.name <<- params.name[-remove]
            params.nr <<- params.nr[-remove]
  
} # eo FUN - run_model_wrapper


### Now, run the functions for optimizing the RF models

### Option 1) for parameter removal  ----------------------------------------------------------------
run_mode <- "remove"
mtryvalue <- 4
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- c(1:6)
params_out <- NULL
cat("RUNNING PARAMETER REMOVAL FOR H \n      ")
while( length(params.nr) > 1) { 
    message(paste("In while loop because length(params.nr) == ",length(params.nr), sep = "")) 
    l <- length(params.nr)
    run_model_wrapper(p = params.nr, dat = scaled.H, income = "H", run_mode = "remove")
} # eo while

mtryvalue <- 3
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- c(1:6)
params_out <- NULL
cat("RUNNING PARAMETER REMOVAL FOR UM \n      ")
while(length(params.nr) > 1) {
    message(paste("In while loop because length(params.nr) == ",length(params.nr), sep = "")) 
    l <- length(params.nr)
    run_model_wrapper(p = params.nr, dat = scaled.UM, income = "UM", run_mode = "remove")
}

mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- c(1:6)
params_out <- NULL
cat("RUNNING PARAMETER REMOVAL FOR LM \n      ")
while(length(params.nr) > 1) { 
    message(paste("In while loop because length(params.nr) == ",length(params.nr), sep = "")) 
    l <- length(params.nr)
    run_model_wrapper(p = params.nr, dat = scaled.LM, income = "LM", run_mode = "remove")
}

mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- c(1:6)
params_out <- NULL
cat("RUNNING PARAMETER REMOVAL FOR L \n      ")
while(length(params.nr) > 1) { 
    message(paste("In while loop because length(params.nr) == ",length(params.nr), sep = "")) 
    l <- length(params.nr)
    run_model_wrapper(p = params.nr, dat = scaled.L, income = "L", run_mode = "remove")
}


### Option 2) for parameter shuffling  ----------------------------------------------------------------
run_mode <- "shuffle"

# High income
mtryvalue <- 4
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING PARAMETER RESHUFFLING FOR H \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.H, income = "H", run_mode = "shuffle")

# Medium income
mtryvalue <- 3
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING PARAMETER RESHUFFLING FOR UM \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.UM, income = "UM", run_mode = "shuffle")

# MLow income
mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING PARAMETER RESHUFFLING FOR LM \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.LM, income = "LM", run_mode = "shuffle")

# Low income
mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING PARAMETER RESHUFFLING FOR L \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.L, income = "L", run_mode = "shuffle")


### Option 3) for univariate  ----------------------------------------------------------------
# High income
run_mode <- "univariate"
mtryvalue <- 4
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING UNIVARIATE PARAMETER FOR H \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.H, income = "H", run_mode = "univariate")

# Medium income
mtryvalue <- 3
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING UNIVARIATE PARAMETER FOR UM \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.UM, income = "UM", run_mode = "univariate")

# Low Medium income
mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING UNIVARIATE PARAMETER FOR LM \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.LM, income = "LM", run_mode = "univariate")

# Low Medium income
mtryvalue <- 2
params.original <- c("p1","p2","p3","p4","p5","p6")
params.name <- params.original
params.nr <- 1:6
params_out <- NULL
cat("RUNNING UNIVARIATE PARAMETER FOR L \n      ")
l <- length(params.nr)
run_model_wrapper(params.nr, dat = scaled.L, income = "L", run_mode = "univariate")


### ----------------------------------------------------------------

### 03/12/19: PV selection : Check the changes in RF accuracy and choose the variables to keep
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/"); dir()

# For testing the FUN below
income <- "H"
run_mode <- "remove"

variable.selection <- function(income, run_mode) {
      
        if( run_mode == "remove") {
                 setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/remove/")
        } else if(run_mode == "shuffle") {
                 setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/shuffle/")
        } else {
                 setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/univariate/")
        } # eo if else loop
         
          MSE <- NA
          MSE.diff <- NA
          var.names <- NA
          
          files <- dir()[grep(paste(income,".csv",sep=""),dir())]
          files <- files[grep(run_mode,files)]
          # for testing for loop: 
          # f <- "data_RF_optim_shuffle_p-6_GNI-H.csv"
          # f <- files[1]
          for(f in files) {
              
                  # Extract the names of the predictors according to the .pdf or the .Rdata that matched the .csv
                  # First, get the matching "p-"
                  chars <- do.call(rbind,strsplit(f, split = "_"))[,5]
                  files2 <- dir()[grep(paste("_",chars,"_",sep=""),dir())]
                  files2 <- files2[grep(".Rdata",files2)]
                  files2 <- files2[grep(paste("-",income,"_.", sep = ""),files2)]
                  names <- do.call(rbind,strsplit(files2, split = "_"))[,c(7)]
                  names2 <- data.frame( do.call(rbind,strsplit(names, split = "\\+")) )
                  colnames <- lapply(c(1:length(names2)), function(c) {
                                  n <- unique(names2[,c(c)])
                                  return(n)
                          } # eo FUN
                  ) # eo lapply
                  # Bind in vector and remove duplicates
                  colnames <- unlist(colnames)[!duplicated(unlist(colnames))]
                  
                  dat <- read.csv(f, h = T) # str(dat) # only numerics, ok
                  dat <- dat[,-1] # remove 1st col because indices only
                  
                  if( length(colnames) == length(dat) ) {
                      colnames(dat) <- colnames
                  } else {
                      message(paste("Vector of colnames and dimension of NSE matrix do not match !", sep = ""))
                  }
                  
                  ### Make plot of distribution of MSE change per variable from here:
                  # - melt 
                  melt.dat <- melt(dat)
                  colnames(melt.dat) <- c("variable","MSE")
                  #quartz()
                  plot <- ggplot(aes(x = factor(variable), y = MSE, fill = factor(variable)), data = melt.dat) + 
                              geom_boxplot(colour = "black") + scale_fill_manual(name = "PV", values = economist_pal()(6) ) +
                              xlab("") + scale_y_continuous(name = "MSE") +
                              ggtitle( paste("MSE variations through ",run_mode," for GNI-",income," | ",chars, sep = "") )
                  #
                  ggsave(plot = plot, filename = paste("boxplot_MSE_",run_mode,"_GNI-",income,"_",chars,".pdf", sep = ""), dpi = 300, height = 6, width = 6)
              
              } # eo for loop
              
        #setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/")
        
} # eo FUN - variable.selection

variable.selection(income = "H", run_mode = "remove") 
variable.selection(income = "UM", run_mode = "remove") 
variable.selection(income = "LM", run_mode = "remove") 
variable.selection(income = "L", run_mode = "remove") 


### And to examine variations in the RF %Var explained from the RF objects themselves ?
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/RF_optimization_per_GNI/remove/") #; dir()
files <- dir()[grep("Rdata",dir())]#; files
# For testing:
# f <- files[3]
res <- lapply(files, function(f) { 
                message(paste(f, sep = ""))
                dat <- get(load(f)) 
                rsq <- dat$rsq[500]
                expl <- round(rsq*100,2)
                # Extract metadata from file name
                chars <- do.call(rbind,strsplit(f, split = "_"))[,6]
                income <- do.call(rbind,strsplit(f, split = "_"))[,8]
                formula <- do.call(rbind,strsplit(f, split = "_"))[,7]
                run <- do.call(rbind,strsplit(f, split = "_"))[,5]
                ddf <- data.frame(run = run, income = income, formula = formula, expl = expl)
                return(ddf)
        } # eo FUN     
) # eo lapply
table <- do.call(rbind, res)
table[order(table$expl),]

### 17/01/23: After examining all the ranks based on the MSE and the % explained variance, here are the rankings of the PVs per GNI class

### As a reminder, here are the top 4 rankings for the 2019-20 analyses: 
# H -> p2/p3/p4/p5
# UM -> p1/p3/p4/p5
# LM -> p1/p2/p4/p5
# L -> p1/p4/p5/p6
 
### Updated rankings 
# H : p2 > p5/p4 > p3 > p6 > p1 (would remove p1 & p6)
# UM : p5/p4 > rest (would retain p5/p4 for tests)
# LM : p5 > p2 > rest (would retain p5/p2 for tests) 
# L : p2 > p4/p5 > rest (would retain p2/p4/p5)


### Finally estimate best 'mtry' value again based on the tests above, with tuneRF(), like before
# High income
best.mtry.H <- NA # empty vector
N <- nrow(scaled.H) + nrow(scaled.UM) + nrow(scaled.LM) + nrow(scaled.L)
for(i in 1:100){
  pos <- sample(1:10, N, replace = T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.H[,c("p2","p3","p4","p5")], y = scaled.H$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.H[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.H)
### --> mtry = 3

# U Medium income
best.mtry.UM <- NA # empty vector
for(i in 1:100){
  pos <- sample(1:10, N, replace=T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.UM[,c("p4","p5")], y = scaled.UM$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.UM[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.UM)
### --> mtry = 2

# Low Medium income
best.mtry.LM <- NA # empty vector
for(i in 1:100){
  pos <- sample(1:10, N, replace=T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.LM[,c("p2","p5")], y = scaled.LM$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.LM[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.LM)
### --> mtry = 2

# Low income
best.mtry.L <- NA # empty vector
for(i in 1:100){
  pos <- sample(1:10, N, replace=T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.L[,c("p2","p4","p5")], y = scaled.L$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.L[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.L)
### --> mtry = 3

### Ok so final mtry estimates are: 
# H = 3
# UM = 2
# LM = 2
# L = 3


### 17/01/23: Using the mtry values and the class-specific predictors sets, evaluate the impact of the ntree parameter on the RF performance
### (MSE and %Var expl)
classes <- c("H","UM","LM","L")
ntreees <- seq(from = 10, to = 500, by = 10) ; ntreees
require("parallel")
# For testing:
cl <- "L"
res.classes <- mclapply(classes, function(cl) {
    
            # Useless message
            message(paste("Running RF for GNI-", cl, sep = ""))
            message(paste("", sep = ""))
            # Second lapply for various ntrees values
            # for testing: 
            # nt <- 50
           
            res.ntrees <- lapply(ntreees, function(nt) {
                                
                            # Choose predictors & mtry according to cl
                            if(cl == "H") {
                                mtryvalue <- 3
                                preds <- c("p2","p3","p4","p5")
                                scaled.temp <- scaled.H
                            } else if(cl == "UM") {
                                mtryvalue <- 2
                                preds <- c("p4","p5")
                                scaled.temp <- scaled.UM
                            } else if(cl == "LM") {
                                mtryvalue <- 2
                                preds <- c("p2","p5")
                                scaled.temp <- scaled.LM
                            } else if(cl == "L") {
                                mtryvalue <- 3
                                preds <- c("p2","p4","p5")
                                scaled.temp <- scaled.L
                            } # eo else if loop
                            
                            MSE.RF.old <- 999
                            MSE.RF <- NA
                            # And same for adjusted R2
                            r2.RF <- NA
                            
                            message(paste("Running 100 RF models for GNI-", cl, " for ntree = ", nt, sep = ""))
                            
                            # for k values in 1:100
                            for(k in 1:100) {
              
                                    # Create train and test dataset
                                    pos <- sample(1:10, nrow(scaled.temp), replace = T )
                                    while( length(table(pos)) != 10 ) { # Prevent that one number is not represented in small samples
                                            # Split the data into 10 groups, randomly (9/10 training, 1/10 testing)
                                            pos <- sample(1:10, nrow(scaled.temp), replace = T )
                                    } # eo while loop
                  
                                    # Filter out the group that will be used for testing ("unseen data")
                                    trainRF <- scaled.temp[pos!=10,]
      
                                    # Create a RF model
                                    RF <- randomForest(x = trainRF[,preds], y = trainRF[,"y"], data = trainRF, mtry = mtryvalue, ntree = nt)
                                    # Return % explained variance
                                    expl <- RF$rsq[nt]
                                    r2.RF[k] <- expl
                                    
                                    MSE.RF.cv <- NA
                                    # split in 1/10 
                                    for(i in 1:10) {
                                            # Create train and test dataset
                                            testRF <- scaled.temp[pos == i,]  
                                            # Prediction using neural network
                                            temp <- data.frame(testRF[,-c(1)])
                                            if(dim(temp)[2]==1){ 
                                                    colnames(temp) <- colnames(testRF)[2] 
                                            } # eo if loop
                                            predict_testRF <- predict(RF, temp)
                                            # Unscale for CV
                                            predict_testRF.cv <- exp((predict_testRF * (max[1] - min[1])) + min[1])
                                            testRF.cv <- exp((testRF$y * (max[1] - min[1])) + min[1])
                                            # Prevent negative values
                                            predict_testRF[which(predict_testRF < 0)] <- 0
                                            # Calculate Mean Square Error (MSE)
                                            MSE.RF.cv[i] = (sum((testRF.cv - predict_testRF.cv)^2) / nrow(testRF))
                                    } # eo for i 1:/10 for cross validation 
      
                                    # Compute mean MSE
                                    MSE.RF[k] <- mean(MSE.RF.cv)
      
                                    # Save the RF if the error of unseen data isn't the largest
                                    if( which.max(MSE.RF.cv) != 10) {
                      
                                        if(MSE.RF[k] < MSE.RF.old) {
                          
                                              temp <- data.frame(scaled.temp[,-c(1)])
                                              if( dim(temp)[2] == 1 ) {
                                                  colnames(temp) <- colnames(testRF)[2] 
                                              } # eo if loop
                                              # Compute r2
                                              pred <- predict(RF, temp)
                                              pred <- exp((pred * (max[1] - min[1])) + min[1])
                                              pred[which(pred<0)] <- 0
                                              measure <- exp((scaled.temp$y * (max[1] - min[1])) + min[1])
                                              r2 <- summary(lm(pred ~ measure))$r.squared
                                              MSE.RF.old <- MSE.RF[k]
                            
                                      } # eo if else loop
                        
                                   } # eo if else loop
                  
                          } # eo for loop

                          # return(MSE.RF)
                          mean.mse <- mean(MSE.RF, na.rm = T)
                          mean.r2 <- round(mean(r2.RF, na.rm = T),3)
                          # return table
                          tibble <- data.frame(ntree = nt, MSE = mean.mse, r2 = mean.r2)
                          return(tibble)
                
                } # eo FUN
                
            ) # eo lapply
            # Rbind, provide 'cl' and return
            res <- do.call(rbind, res.ntrees)
            res$GNI <- cl
            rm(res.ntrees) ; gc()
            
            # Return
            return(res)
    
    } , mc.cores = 4
    
) # mclapply
# Rbind
table <- do.call(rbind, res.classes)
dim(table); str(table); summary(table)

setwd(paste(WD,"/outputs/RF_optimization_per_GNI/", sep = ""))
save(x = table, file = "table_RF_tests_ntrees_17_01_23.RData")

### Make profile of distrbution of MSE and then r2 (% explained variance) by facetting per each GNI class
setwd(paste(WD,"/plots/", sep = ""))

# 1st, MSE (like in Kevin's thesis)
plot1 <- ggplot(data = table) + geom_line(aes(x = ntree, y = MSE )) + 
        geom_point(aes(x = ntree, y = MSE, fill = factor(GNI)), pch = 21, colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
        xlab("Number of trees in the RF models") + ylab("Average MSE") + 
        theme_bw() + facet_wrap(~factor(GNI), ncol = 2, scales = "free_y") 
ggsave(plot = plot1, filename = "plot_MSExntree_17_01_23.pdf", dpi = 300, height = 5, width = 7)

# 2nd, % explained var
plot2 <- ggplot(data = table) + geom_line(aes(x = ntree, y = r2)) + 
        geom_point(aes(x = ntree, y = r2, fill = factor(GNI)), pch = 21, colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of trees in the RF models") + ylab("Average % of explained variance") + 
        theme_bw() + facet_wrap(~factor(GNI), ncol = 2, scales = "free_y") 

ggsave(plot = plot2, filename = "plot_explVarxntree_17_01_23.pdf", dpi = 300, height = 5, width = 7)

### ntree = 100-200 seems to be good enough. However, UM and LM models do not really converge towards an optimal ntree value ever, likely because not enough PVs (n = 2) and maybe too low mtry value?

### --> Try again by adding one or two PVs for the UM and LM models based on the 'univariate' tests (remove & shuffle do not provide clear enough trends in rankings). 
# U Medium income: add p3 and p6
best.mtry.UM <- NA 
for(i in 1:100){
  pos <- sample(1:10, N, replace=T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.UM[,c("p3","p4","p5","p6")], y = scaled.UM$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.UM[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.UM)
### --> mtry = 2

# Low Medium income: add p1 and p4
best.mtry.LM <- NA 
for(i in 1:100){
  pos <- sample(1:10, N, replace=T)
  invisible(capture.output(tuned <- tuneRF(x = scaled.LM[,c("p1","p2","p4","p5")], y = scaled.LM$y, 
              mtryStart = 2, stepFactor = 0.9, improve = 0.0000000001, plot = F, trace = F)))
  best.mtry.LM[i] <- tuned[which.min(tuned[,2]),1]
}
table(best.mtry.LM)
### --> new mtry = 3

### Re-run code above
res.classes <- mclapply(classes, function(cl) {
    
            # Useless message
            message(paste("Running RF for GNI-", cl, sep = ""))
            message(paste("", sep = ""))
            # Second lapply for various ntrees values
            # for testing: 
            # nt <- 50
           
            res.ntrees <- lapply(ntreees, function(nt) {
                                
                            # Choose predictors & mtry according to cl
                            if(cl == "H") {
                                mtryvalue <- 3
                                preds <- c("p2","p3","p4","p5")
                                scaled.temp <- scaled.H
                            } else if(cl == "UM") {
                                mtryvalue <- 2
                                preds <- c("p3","p4","p5","p6")
                                scaled.temp <- scaled.UM
                            } else if(cl == "LM") {
                                mtryvalue <- 3
                                preds <- c("p1","p2","p4","p5")
                                scaled.temp <- scaled.LM
                            } else if(cl == "L") {
                                mtryvalue <- 3
                                preds <- c("p2","p4","p5")
                                scaled.temp <- scaled.L
                            } # eo else if loop
                            
                            MSE.RF.old <- 999
                            MSE.RF <- NA
                            # And same for adjusted R2
                            r2.RF <- NA
                            
                            message(paste("Running 100 RF models for GNI-", cl, " for ntree = ", nt, sep = ""))
                            
                            # for k values in 1:100
                            for(k in 1:100) {
              
                                    # Create train and test dataset
                                    pos <- sample(1:10, nrow(scaled.temp), replace = T )
                                    while( length(table(pos)) != 10 ) { # Prevent that one number is not represented in small samples
                                            # Split the data into 10 groups, randomly (9/10 training, 1/10 testing)
                                            pos <- sample(1:10, nrow(scaled.temp), replace = T )
                                    } # eo while loop
                  
                                    # Filter out the group that will be used for testing ("unseen data")
                                    trainRF <- scaled.temp[pos!=10,]
      
                                    # Create a RF model
                                    RF <- randomForest(x = trainRF[,preds], y = trainRF[,"y"], data = trainRF, mtry = mtryvalue, ntree = nt)
                                    # Return % explained variance
                                    expl <- RF$rsq[nt]
                                    r2.RF[k] <- expl
                                    
                                    MSE.RF.cv <- NA
                                    # split in 1/10 
                                    for(i in 1:10) {
                                            # Create train and test dataset
                                            testRF <- scaled.temp[pos == i,]  
                                            # Prediction using neural network
                                            temp <- data.frame(testRF[,-c(1)])
                                            if(dim(temp)[2]==1){ 
                                                    colnames(temp) <- colnames(testRF)[2] 
                                            } # eo if loop
                                            predict_testRF <- predict(RF, temp)
                                            # Unscale for CV
                                            predict_testRF.cv <- exp((predict_testRF * (max[1] - min[1])) + min[1])
                                            testRF.cv <- exp((testRF$y * (max[1] - min[1])) + min[1])
                                            # Prevent negative values
                                            predict_testRF[which(predict_testRF < 0)] <- 0
                                            # Calculate Mean Square Error (MSE)
                                            MSE.RF.cv[i] = (sum((testRF.cv - predict_testRF.cv)^2) / nrow(testRF))
                                    } # eo for i 1:/10 for cross validation 
      
                                    # Compute mean MSE
                                    MSE.RF[k] <- mean(MSE.RF.cv)
      
                                    # Save the RF if the error of unseen data isn't the largest
                                    if( which.max(MSE.RF.cv) != 10) {
                      
                                        if(MSE.RF[k] < MSE.RF.old) {
                          
                                              temp <- data.frame(scaled.temp[,-c(1)])
                                              if( dim(temp)[2] == 1 ) {
                                                  colnames(temp) <- colnames(testRF)[2] 
                                              } # eo if loop
                                              # Compute r2
                                              pred <- predict(RF, temp)
                                              pred <- exp((pred * (max[1] - min[1])) + min[1])
                                              pred[which(pred<0)] <- 0
                                              measure <- exp((scaled.temp$y * (max[1] - min[1])) + min[1])
                                              r2 <- summary(lm(pred ~ measure))$r.squared
                                              MSE.RF.old <- MSE.RF[k]
                            
                                      } # eo if else loop
                        
                                   } # eo if else loop
                  
                          } # eo for loop

                          # return(MSE.RF)
                          mean.mse <- mean(MSE.RF, na.rm = T)
                          mean.r2 <- round(mean(r2.RF, na.rm = T),3)
                          # return table
                          tibble <- data.frame(ntree = nt, MSE = mean.mse, r2 = mean.r2)
                          return(tibble)
                
                } # eo FUN
                
            ) # eo lapply
            # Rbind, provide 'cl' and return
            res <- do.call(rbind, res.ntrees)
            res$GNI <- cl
            rm(res.ntrees) ; gc()
            
            # Return
            return(res)
    
    } , mc.cores = 4
    
) # mclapply
# Rbind
table <- do.call(rbind, res.classes)
dim(table); str(table); summary(table)

setwd(paste(WD,"/outputs/RF_optimization_per_GNI/", sep = ""))
save(x = table, file = "table_RF_tests_ntrees_v2_17_01_23.RData")

### Make profile of distrbution of MSE and then r2 (% explained variance) by facetting per each GNI class
setwd(paste(WD,"/plots/", sep = ""))

# 1st, MSE (like in Kevin's thesis)
plot1 <- ggplot(data = table) + geom_line(aes(x = ntree, y = MSE )) + 
        geom_point(aes(x = ntree, y = MSE, fill = factor(GNI)), pch = 21, colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
        xlab("Number of trees in the RF models") + ylab("Average MSE") + 
        theme_bw() + facet_wrap(~factor(GNI), ncol = 2, scales = "free_y") 
ggsave(plot = plot1, filename = "plot_MSExntree_v2_17_01_23.pdf", dpi = 300, height = 5, width = 7)

# 2nd, % explained var
plot2 <- ggplot(data = table) + geom_line(aes(x = ntree, y = r2)) + 
        geom_point(aes(x = ntree, y = r2, fill = factor(GNI)), pch = 21, colour = "black") + 
        scale_fill_manual(name = "GNI", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) + 
        xlab("Number of trees in the RF models") + ylab("Average % of explained variance") + 
        theme_bw() + facet_wrap(~factor(GNI), ncol = 2, scales = "free_y") 

ggsave(plot = plot2, filename = "plot_explVarxntree_v2_17_01_23.pdf", dpi = 300, height = 5, width = 7)


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
