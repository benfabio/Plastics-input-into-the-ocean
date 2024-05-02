
##### 23/01/23: R script to examine Kevin Lang's legacy (data, codes etc.), © Fabio Benedetti, UP group, IPB, D-USYS, ETH Zürich. 
##### Goal of the project is to model and predict waste production per country and per year (1990-2019) based on societal predictors 
##### such as % urban pop, GDP, access to electricty, etc. using GAMS, Random Forest (RF), and Neural Networks (NNET) if robust enough.
##### Ultimately, you need to provide to C. Laufkötter a long table (countries x year) containing the annual waste production estimates.
##### She will then use those to predict the total amount of plastics that enters the ocean using a waste treatment flow model. 

##### Present script aims to: 
# - Compare the ensemble predictions from GAM and RF models (1000 models per model type and GNI class) + early NNET models (less than 1000) + full GBM models (1000) + GLMs
# - Compute final ensemble predictions & errors based on RF and GAM predictions

### Last update: 09/02/23

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
library("ggrepel")

worldmap <- getMap(resolution = "high")

### ------------------------------------------------------------------------------------------------------------------------------------------------

### A°) Compare predictions and error from GAMs and RF

### A.1. Retrieve ensemble predictions and errors
### Retrieve RF ensemble predictions and errors
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions"); dir()[grep("median",dir())]
pred.rf <- read.table("table_ranges_RF_median_predictions_23_01_23.txt", h = T, sep = ";") 
error.rf <- read.table("table_RF_median_errors_percentage_23_01_23.txt", h = T, sep = ";")

### Retrieve GAM ensemble predictions and errors
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_predictions"); dir()[grep("median",dir())]
pred.gam <- read.table("table_ensemble_predictions_GAM_23_01_23.txt", h = T, sep = "\t")
error.gam <- read.table("table_GAM_median_errors_percentage_23_01_23.txt", h = T, sep = ";")

### Retrieve NNET ensemble predictions and errors
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/NNET_predictions"); dir()[grep("median",dir())]
pred.nnet <- read.table("table_ensemble_predictions_NNET_24_01_23.txt", h = T, sep = "\t")
error.nnet <- read.table("table_NNET_median_errors_percentage_24_01_23.txt", h = T, sep = ";")

### Retrieve GBM ensemble predictions and errors
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GBM_predictions"); dir()[grep("ensemble",dir())]
pred.gbm <- read.table("table_ensemble_predictions_GBM_30_01_23.txt", h = T, sep = "\t")
error.gbm <- read.table("table_GBM_median_errors_percentage_30_01_23.txt", h = T, sep = ";")


# Check dimensions & colnames
# dim(pred.rf); dim(pred.gam); dim(pred.nnet); dim(pred.gbm)
# dim(error.rf); dim(error.gam); dim(error.nnet); dim(error.gbm)
# OK

colnames(pred.rf)[c(2:31)] <- as.character(c(1990:2019))
colnames(pred.gam)[c(2:31)] <- as.character(c(1990:2019))
colnames(pred.nnet)[c(2:31)] <- as.character(c(1990:2019))
colnames(pred.gbm)[c(2:31)] <- as.character(c(1990:2019))
colnames(error.rf)[c(2:31)] <- as.character(c(1990:2019))
colnames(error.gam)[c(2:31)] <- as.character(c(1990:2019))
colnames(error.nnet)[c(2:31)] <- as.character(c(1990:2019))
colnames(error.gbm)[c(2:31)] <- as.character(c(1990:2019))

# Merge all after melting to long format
m.pred.rf <- melt(pred.rf)
m.error.rf <- melt(error.rf)
colnames(m.pred.rf)[c(2,3)] <- c("Year","MSW_RF")
colnames(m.error.rf)[c(2,3)] <- c("Year","Error_RF")

m.pred.gam <- melt(pred.gam)
m.error.gam <- melt(error.gam)
colnames(m.pred.gam)[c(2,3)] <- c("Year","MSW_GAM")
colnames(m.error.gam)[c(2,3)] <- c("Year","Error_GAM")

m.pred.nnet <- melt(pred.nnet)
m.error.nnet <- melt(error.nnet)
colnames(m.pred.nnet)[c(2,3)] <- c("Year","MSW_NNET")
colnames(m.error.nnet)[c(2,3)] <- c("Year","Error_NNET")

m.pred.gbm <- melt(pred.gbm)
m.error.gbm <- melt(error.gbm)
colnames(m.pred.gbm)[c(2,3)] <- c("Year","MSW_GBM")
colnames(m.error.gbm)[c(2,3)] <- c("Year","Error_GBM")

# Bind in one ddf
m.pred <- m.pred.rf
m.pred$MSW_GAM <- m.pred.gam$MSW_GAM
m.pred$MSW_NNET <- m.pred.nnet$MSW_NNET
m.pred$MSW_GBM <- m.pred.gbm$MSW_GBM

m.error <- m.error.rf
m.error$Error_GAM <- m.error.gam$Error_GAM
m.error$Error_NNET <- m.error.nnet$Error_NNET
m.error$Error_GBM <- m.error.gbm$Error_GBM
# str(m.pred)
# str(m.error)

### Add an ID
m.pred$ID <- paste(m.pred$Country, m.pred$Year, sep = "_")
m.error$ID <- paste(m.error$Country, m.error$Year, sep = "_")

### A.3. Compare outputs (biplots)
# Compare predictions
ggplot(data = m.pred) + geom_point(aes(x = MSW_RF, y = MSW_GAM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF predictions") + ylab("MSW GAM predictions") +
    theme_bw()

ggplot(data = m.pred) + geom_point(aes(x = MSW_RF, y = MSW_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF predictions") + ylab("MSW NNET predictions") +
    theme_bw()

ggplot(data = m.pred) + geom_point(aes(x = MSW_GAM, y = MSW_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW NNET predictions") + xlab("MSW GAM predictions") +
    theme_bw()

ggplot(data = m.pred) + geom_point(aes(x = MSW_GAM, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW GBM predictions") + xlab("MSW GAM predictions") +
    theme_bw()

ggplot(data = m.pred) + geom_point(aes(x = MSW_RF, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW GBM predictions") + xlab("MSW RF predictions") +
    theme_bw()

ggplot(data = m.pred) + geom_point(aes(x = MSW_NNET, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW GBM predictions") + xlab("MSW NNET predictions") +
    theme_bw()
     
# Check ranges
# summary(m.pred)

# Issue with some countries
m.pred[m.pred$MSW_GAM > 4,] # 
m.pred[m.pred$MSW_NNET > 4,]
#unique(m.pred[m.pred$MSW_NNET > 4,"Country"])
countries2remove <- unique(m.pred[m.pred$MSW_GAM > 4,"Country"])
# [1] "Botswana"                  "South Sudan"              
# [3] "Tuvalu"                    "Korea, Dem. People's Rep."
# [5] "Guinea-Bissau"   
# Re-plot
p1 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_RF, y = MSW_GAM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF predictions") + ylab("MSW GAM predictions") +
    geom_text_repel(aes(x = MSW_RF, y = MSW_GAM, label = ID),
    data = m.pred[m.pred$MSW_GAM > 2 & m.pred$MSW_RF < 2 & !(m.pred$Country %in% countries2remove),]) +
    theme_bw() + ggtitle("RF x GAM predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")

p2 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_RF, y = MSW_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF predictions") + ylab("MSW NNET predictions") +
    geom_text_repel(aes(x = MSW_RF, y = MSW_NNET, label = ID),
    data = m.pred[m.pred$MSW_RF > 2,]) +
    theme_bw() + ggtitle("RF x NNET predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")

p3 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_GAM, y = MSW_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW NNET predictions") + xlab("MSW GAM predictions") +
    geom_text_repel(aes(x = MSW_GAM, y = MSW_NNET, label = ID),
    data = m.pred[m.pred$MSW_GAM > 2 & !(m.pred$Country %in% countries2remove),]) +
    theme_bw() + ggtitle("GAM x NNET predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")

p4 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_RF, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF predictions") + ylab("MSW GBM predictions") +
    geom_text_repel(aes(x = MSW_RF, y = MSW_GBM, label = ID),
    data = m.pred[m.pred$MSW_RF > 2 & !(m.pred$Country %in% countries2remove),]) +
    theme_bw() + ggtitle("RF x GBM predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")

p5 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_NNET, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW NNET predictions") + ylab("MSW GBM predictions") +
    geom_text_repel(aes(x = MSW_NNET, y = MSW_GBM, label = ID),
    data = m.pred[!(m.pred$Country %in% countries2remove) & m.pred$MSW_NNET > 1.5,]) +
    theme_bw() + ggtitle("NNET x GBM predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")

p6 <- ggplot(data = m.pred[-which(m.pred$Country %in% countries2remove),]) +
    geom_point(aes(x = MSW_GAM, y = MSW_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    ylab("MSW NNET predictions") + xlab("MSW GAM predictions") +
    geom_text_repel(aes(x = MSW_GAM, y = MSW_GBM, label = ID),
    data = m.pred[m.pred$MSW_GAM > 2 & !(m.pred$Country %in% countries2remove),]) +
    theme_bw() + ggtitle("GAM x GBM predictions\n(minus Botswana, North Korea, Tuvalu, South Sudan & Guinea-Bissau)")


setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
ggsave(plot = p1, filename = "plot_compare_preds_RFxGAM_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p2, filename = "plot_compare_preds_RFxNNET_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p3, filename = "plot_compare_preds_GAMxNNET_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p4, filename = "plot_compare_preds_RFxGBM_30_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p5, filename = "plot_compare_preds_NNETxGNM_30_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p6, filename = "plot_compare_preds_GAMxGBM_30_01_23.pdf", dpi = 300, height = 10, width = 10)


### Compare errors
# Check ranges
summary(m.error)

### Plot distrbution of errors per model
m.m.error <- melt(m.error)
# head(m.m.error)
p1 <- ggplot(aes(x = factor(variable), y = abs(value), fill = factor(variable)), data = m.m.error) + geom_boxplot(colour = "black") +
             scale_fill_manual(name = "Model types", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
             xlab("") + ylab("Error (%)") + theme_bw() + scale_x_discrete(labels = NULL, breaks = NULL)

p2 <- ggplot(aes(x = factor(variable), y = abs(value), fill = factor(variable)), data = na.omit(m.m.error[m.m.error$value < 50,])) +
         geom_boxplot(colour = "black") + scale_fill_manual(name = "Model types", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5") ) +
         xlab("") + ylab("Error (%)") + theme_bw() + scale_x_discrete(labels = NULL, breaks = NULL) 

# data.frame( m.m.error %>% group_by(variable) %>% summarize(med = median(value, na.rm = T), iqr = IQR(value, na.rm = T)) )
library("ggpubr")
panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "h", common.legend = T)
ggsave(plot = panel, filename = "boxplot_compare_errors_models_30_01_23.pdf", dpi = 300, height = 6, width = 6)

### Compare range of errors between model types
# str(m.error)
ggplot(data = m.error) + geom_point(aes(x = Error_RF, y = Error_GAM)) +
     geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
     xlab("MSW RF errors") + ylab("MSW GAM errors") +
     theme_bw()

ggplot(data = m.error) + geom_point(aes(x = Error_RF, y = Error_NNET)) +
     geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
     xlab("MSW RF errors") + ylab("MSW NNET errors") +
     theme_bw()

ggplot(data = m.error) + geom_point(aes(x = Error_GAM, y = Error_NNET)) +
     geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
     ylab("MSW NNET errors") + xlab("MSW GAM errors") +
     theme_bw()

### Add labels with ggrepel
p1 <- ggplot(data = m.error) + geom_point(aes(x = Error_RF, y = Error_GAM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF errors") + ylab("MSW GAM errors") +
    geom_text_repel(aes(x = Error_RF, y = Error_GAM, label = ID),
    data = m.error[m.error$Error_RF > 25 | m.error$Error_GAM > 25,]) +
    theme_bw()

p2 <- ggplot(data = m.error) + geom_point(aes(x = Error_RF, y = Error_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF errors") + ylab("MSW NNET errors") +
    geom_text_repel(aes(x = Error_RF, y = Error_NNET, label = ID),
    data = m.error[m.error$Error_RF > 25 | m.error$Error_NNET > 50,]) +
    theme_bw()

p3 <- ggplot(data = m.error) + geom_point(aes(x = Error_GAM, y = Error_NNET)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW GAM errors") + ylab("MSW NNET errors") +
    geom_text_repel(aes(x = Error_GAM, y = Error_NNET, label = ID),
    data = m.error[m.error$Error_GAM > 25 | m.error$Error_NNET > 50,]) +
    theme_bw()

p4 <- ggplot(data = m.error) + geom_point(aes(x = Error_RF, y = Error_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW RF errors") + ylab("MSW GBM errors") +
    geom_text_repel(aes(x = Error_RF, y = Error_GBM, label = ID),
    data = m.error[m.error$Error_RF > 25 | m.error$Error_GBM > 25,]) +
    theme_bw()

p5 <- ggplot(data = m.error) + geom_point(aes(x = Error_NNET, y = Error_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW NNET errors") + ylab("MSW GBM errors") +
    geom_text_repel(aes(x = Error_NNET, y = Error_GBM, label = ID),
    data = m.error[m.error$Error_GBM > 50 | m.error$Error_NNET > 50,]) +
    theme_bw()

p6 <- ggplot(data = m.error) + geom_point(aes(x = Error_GAM, y = Error_GBM)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    xlab("MSW GAM errors") + ylab("MSW GBM errors") +
    geom_text_repel(aes(x = Error_GAM, y = Error_GBM, label = ID),
    data = m.error[m.error$Error_GAM > 25 | m.error$Error_GBM > 50,]) +
    theme_bw()

ggsave(plot = p1, filename = "plot_compare_errors_RFxGAM_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p2, filename = "plot_compare_errors_RFxNNET_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p3, filename = "plot_compare_errors_GAMxNNET_25_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p4, filename = "plot_compare_errors_RFxGBM_30_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p5, filename = "plot_compare_errors_NNETxGBM_30_01_23.pdf", dpi = 300, height = 10, width = 10)
ggsave(plot = p3, filename = "plot_compare_errors_GAMxGBM_30_01_23.pdf", dpi = 300, height = 10, width = 10)


### 30/01/23: Quickly compare to GLM predictions and errors (no need to print plots for now)
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GLM_predictions"); dir()[grep("ensemble",dir())]
pred.glm <- read.table("table_ensemble_predictions_GLM_30_01_23.txt", h = T, sep = "\t")
error.glm <- read.table("table_GLM_median_errors_percentage_30_01_23.txt", h = T, sep = ";")
colnames(pred.glm)[c(2:31)] <- as.character(c(1990:2019))
colnames(error.glm)[c(2:31)] <- as.character(c(1990:2019))
m.pred.glm <- melt(pred.glm)
m.error.glm <- melt(error.glm)
colnames(m.pred.glm)[c(2,3)] <- c("Year","MSW_GLM")
colnames(m.error.glm)[c(2,3)] <- c("Year","Error_GLM")
m.pred$MSW_GLM <- m.pred.glm$MSW_GLM
m.error$Error_GLM <- m.error.glm$Error_GLM
# summary(m.pred)
# summary(m.error)

### Re-plot distrbution of errors per model
m.m.error <- melt(m.error)
# head(m.m.error)
p1 <- ggplot(aes(x = factor(variable), y = abs(value), fill = factor(variable)), data = m.m.error) + geom_boxplot(colour = "black") +
             scale_fill_manual(name = "Model types", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5","#E1AF00") ) +
             xlab("") + ylab("Error (%)") + theme_bw() + scale_x_discrete(labels = NULL, breaks = NULL)

p2 <- ggplot(aes(x = factor(variable), y = abs(value), fill = factor(variable)), data = na.omit(m.m.error[m.m.error$value < 50,])) +
         geom_boxplot(colour = "black") + scale_fill_manual(name = "Model types", values = c("#3B9AB2","#F21A00","#EBCC2A","#78B7C5","#E1AF00") ) +
         xlab("") + ylab("Error (%)") + theme_bw() + scale_x_discrete(labels = NULL, breaks = NULL) 

library("ggpubr")
panel <- ggarrange(p1, p2, ncol = 2, nrow = 1, align = "h", common.legend = T)
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/plots")
ggsave(plot = panel, filename = "boxplot_compare_errors_models+GLM_30_01_23.pdf", dpi = 300, height = 4, width = 7)



### ------------------------------------------------------------------------------------------------------------------------------------------------

### 09/02/23
library("parallel")

### B°) Derive mean/median ensemble predictions - merge GAM and RF outputs

setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/RF_predictions")
# dir()[grep("table_pred_",dir())] ; length(dir()[grep("table_pred_",dir())]) # ~1000 models per GNI
files <- dir()[grep(paste("table_pred_", sep = ""),dir())] # length(files)
res <- mclapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) }, mc.cores = 25 ) # eo lapply
ddf <- bind_rows(res)
# dim(ddf); head(ddf) 
colnames(ddf)[c(2:length(ddf))] <- as.character(c(1990:2019))
m.ddf.rf <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
colnames(m.ddf.rf)[c(2:3)] <- c("Year","MSW")
# head(m.ddf)

### Retrieve GAM ensemble predictions and errors
setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/models/GAM_predictions")
# dir()[grep("table_pred_",dir())] ; length(dir()[grep("table_pred_",dir())]) # ~1000 models per GNI
files <- dir()[grep(paste("table_pred_", sep = ""),dir())] # length(files)
res <- mclapply(files, function(f) { d <- read.table(f, sep = "\t", h = T); return(d) }, mc.cores = 25 ) # eo lapply
ddf <- bind_rows(res)
colnames(ddf)[c(2:length(ddf))] <- as.character(c(1990:2019))
m.ddf.gam <- melt(ddf, id.vars = "Country") # dim(m.ddf) # 2'806'232
colnames(m.ddf.gam)[c(2:3)] <- c("Year","MSW")

### Rbind both and compute median/mean/sdev/IQR 
m.ddf <- rbind(m.ddf.rf,m.ddf.gam)
# dim(m.ddf)
summary(m.ddf)

ensemble <- data.frame(m.ddf %>%
            group_by(Country,Year) %>%
            summarize(MSW_median = median(MSW, na.rm = T), MSW_mean = mean(MSW, na.rm = T),
                IQR = IQR(MSW, na.rm = T), STDEV = sd(MSW, na.rm = T))
) # eo ddf

summary(ensemble)
# unique(ensemble$MSW_mean)

setwd("/net/kryo/work/fabioben/Inputs_plastics/2023/outputs/")
write.table(x = ensemble, "table_ensemble_predictions_GAM+RF_09_02_23.txt", sep = "\t")


### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------------------------------