
##### 21/02/23: R script to identify those countries x years where at least one of the six PV goes outside the training range of the models (RF and GAM mostly), per GNI class.

##### Present script aims to: 
# - Load the data, load the missing countries details and provide their PV values
# - Simply identify (with a MESS for instance, or with boxplots) the countries x years whose PVs range outside the observed training range

### Last update: 21/02/23

### ------------------------------------------------------------------------------------------------------------------------------------------------

library("tidyverse")
library("reshape2")
library("scales")
library("RColorBrewer")
library("viridis")
library("ggsci")
library("ggthemes")
library("ggrepel")
library("scales")
library("wesanderson")
library("modEvA") # ?MESS

# Define main working dir
setwd("/net/kryo/work/fabioben/Inputs_plastics/")
WD <- getwd() 

### ------------------------------------------------------------------------------------------------------------------------------------------------

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
#head(missing)
m.missing <- melt(missing, id.vars = c("country","GNI","index","y","p1","p2","p3","p4","p5","p6"))
m.missing <- m.missing[,-c(length(m.missing))]
# Change last colname
colnames(m.missing)[11] <- "Year"
colnames(m.missing)[1] <- "Country"

### Fill estimates of PVs in a for loop
for(c in missing.countries) {
     i <- missing[missing$country == c,"index"]
     m.missing[m.missing$Country == c,"y"] <- y[,i] # should always be NA
     m.missing[m.missing$Country == c,"p1"] <- p1[,i]
     m.missing[m.missing$Country == c,"p2"] <- p2[,i]
     m.missing[m.missing$Country == c,"p3"] <- p3[,i]
     m.missing[m.missing$Country == c,"p4"] <- p4[,i]
     m.missing[m.missing$Country == c,"p5"] <- p5[,i]
     m.missing[m.missing$Country == c,"p6"] <- p6[,i]
} # eo for loop
# Check
# dim(m.missing)
# head(m.missing)
# Re-locate 'Year'
m.missing <- m.missing %>% relocate(Year, .after = Country)

### Get the full countries dataset to compare
te$Country <- countries_complete
te$Year <- year_complete
te$GNI <- GNI_complete
m.full <- te

### Check Botswana for Charlotte: 
# m.full[m.full$Country == "Botswana",] # rows 321:323
# summary(m.full[m.full$GNI == "UM" & m.full$Country != "Botswana",c("p1","p2","p3","p4","p5","p6")])
# require("FactoMineR")
# pca <- PCA(m.full[m.full$GNI == "UM",c(1:7)], scale.unit = T, graph = F)
# plot.PCA(pca, axes = c(1,2), choix = c("ind"))
# plot.PCA(pca, axes = c(2,3), choix = c("ind"))
# summary(pca)

### First, compare ranges of PVs between m.full vs. m.missing
#summary(m.full[,c("p1","p2","p3","p4","p5","p6")])
#summary(m.missing[,c("p1","p2","p3","p4","p5","p6")])

### Bind in same ddf
m.full$Status <- "Full"
m.missing$Status <- "Missing"
names <- colnames(m.full)[c(8,9,10,2:7,11)] # ; names
ddf <- rbind(m.full[,names], m.missing[,names])
#dim(ddf);summary(ddf)

### Plot distribs between Status, and facet per GNI
p1 <- ggplot(data = ddf, aes(x = factor(Status), y = p1, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("Electricity access (%)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y") 

p2 <- ggplot(data = ddf, aes(x = factor(Status), y = p2, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("Energy consumption per capita (log)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y")

p3 <- ggplot(data = ddf, aes(x = factor(Status), y = p3, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("GDP per capita (log)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y")

p4 <- ggplot(data = ddf, aes(x = factor(Status), y = p4, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("GHG emission per capita (log)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y")

p5 <- ggplot(data = ddf, aes(x = factor(Status), y = p5, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("Fraction of urban population (%)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y")

p6 <- ggplot(data = ddf, aes(x = factor(Status), y = p6, fill = factor(Status))) + 
    geom_violin(color = "black") + geom_boxplot(color = "black", fill = "white", width = .2) + 
    scale_fill_manual(name = "", values = c("#3288bd","#d53e4f")) + ylab("Fraction of young population (%)") + xlab("") +
    theme_bw() + facet_wrap(.~ factor(GNI), scales = "free_y")

# Save plots
setwd(paste(WD,"/plots/", sep = "")) ; dir()
ggsave(plot = p1, filename = "boxplots_full.counrries_vs_missing.countries_p1_21_02_23.pdf", dpi = 300, width = 6, height = 6)
ggsave(plot = p2, filename = "boxplots_full.counrries_vs_missing.countries_p2_21_02_23.pdf", dpi = 300, width = 6, height = 6)
ggsave(plot = p3, filename = "boxplots_full.counrries_vs_missing.countries_p3_21_02_23.pdf", dpi = 300, width = 6, height = 6)
ggsave(plot = p4, filename = "boxplots_full.counrries_vs_missing.countries_p4_21_02_23.pdf", dpi = 300, width = 6, height = 6)
ggsave(plot = p5, filename = "boxplots_full.counrries_vs_missing.countries_p5_21_02_23.pdf", dpi = 300, width = 6, height = 6)
ggsave(plot = p6, filename = "boxplots_full.counrries_vs_missing.countries_p6_21_02_23.pdf", dpi = 300, width = 6, height = 6)


### Define ranges of obs values per GNI from 'm.full' (4*6*2 = 48 values) and return countries x years in 'm.missing' that are outisde those ranges
library("parallel")
# c <- "UM"
res <- mclapply(c("H","UM","LM","L"), function(c) {
    
        message(paste("Doing ",c, sep = ""))
        # Define min and max of each PV per GNI
        subset <- m.full[m.full$GNI == c,]
        ranges <- lapply(c("p1","p2","p3","p4","p5","p6"), function(p) {
                # Get min and max values
                mins <- min(subset[,p])
                maxs <- max(subset[,p])
                return(data.frame(PV = p, min = mins, max = maxs))
            } # eo FUN
        ) # eo lapply
        # Rbind
        range <- dplyr::bind_rows(ranges)
        rm(ranges); gc()
        range$GNI <- c
        
        return(range)
    
    }, mc.cores = 4
    
) # eo mclapply - c per GNI
# Rbind
ranges <- do.call(rbind,res)
#ranges

### Use 'ranges' to identify the countries x years in m.missing that go outside the obs ranges of PV
m.missing$ID <- paste(m.missing$Country, m.missing$Year, sep = "_")
# c <- "H"
# p <- "p5"
res <- mclapply(c("H","UM","LM","L"), function(c) {
    
        message(paste("Doing ",c, sep = ""))
        message(paste(" ", sep = ""))
        # Define min and max of each PV per GNI
        subset <- m.missing[m.missing$GNI == c,]
        
        res.2 <- lapply(c("p1","p2","p3","p4","p5","p6"), function(p) {
                    message(paste(p, sep = ""))
                    mi <- ranges[ranges$GNI == c & ranges$PV == p,"min"]
                    ma <- ranges[ranges$GNI == c & ranges$PV == p,"max"]
                    out <- subset[which(subset[,p] < mi | subset[,p] > ma),"ID"]
                    country <- unique(subset[which(subset[,p] < mi | subset[,p] > ma),"Country"])
                    if( length(out) > 0 ) {
                        return(data.frame(PV = p, ID = out))
                    } else {
                        return(data.frame(PV = p, ID = NA))
                    }
                    
            } # eo FUN
        ) # eo lapply
        # Rbind
        range.out <- dplyr::bind_rows(res.2)
        range.out$GNI <- c
        rm(res.2); gc()
        
        return(range.out)
    
    }, mc.cores = 4
    
) # eo mclapply - c per GNI
# Rbind
out <- do.call(rbind,res)
# dim(out)
# length(unique(out$ID)) / length(unique(m.missing$ID)) # 35.8% of the missing entries have some PV outside of the training range

# Extract country from this 
out$Country <- do.call(rbind, strsplit(out$ID, split = "_", fixed = T))[,1]
# head(out)
# tail(out)
#unique(out$Country)
#out[is.na(out$Country),]

out.table <- na.omit(out)
#unique(out.table$Country) # 55 countries

tally <- data.frame( out.table %>% group_by(Country) %>% summarize(n = n(), rate = n/(6*30)) )
tally <- tally[order(tally$rate, decreasing = T),]

### Save these 2 tables for Charlotte
setwd(WD)
write.csv(out.table, file = "table_out_per_missing.counties_21_02_23.csv", sep = ";")
write.csv(tally, file = "table_rate_out_per_missing.counties_21_02_23.csv", sep = ";")

### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------