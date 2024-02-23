# Here we will analyze the disease cohort dataset to obtain the disease signature. The dataset consists of 161 microarray samples (74 normal brain tissues and 86 Alzheimer's disease brain tissues).
# The analysis is conducted using the Geo2 pipeline with slite modifications in terms of the design matrix and the linear model used to fit the data
# The differential gene expression was obtained in T-scores given that the signatures obtained from the camp were in Z-scores which are relatively similar metrics.
################################################################
#   Differential expression analysis with limma
library(GEOquery)
library(limma)
library(dplyr)
library(tibble)

# load series and platform data from GEO

gset <- getGEO("GSE5281", GSEMatrix =TRUE, AnnotGPL=TRUE)
if (length(gset) > 1) idx <- grep("GSE5281", attr(gset, "names")) else idx <- 1
gset <- gset[[idx]]

# make proper column names to match toptable 
fvarLabels(gset) <- make.names(fvarLabels(gset))

# group membership for all samples
gsms <- paste0(rep(c("0", "1"), c(74, 87)), collapse = "")

sml <- strsplit(gsms, split="")[[1]]
sel <- which(sml != 'x')
sml <- sml[sel]
gset <- gset[,sel]

# log2 transformation
ex <- exprs(gset)
qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm=T))
LogC <- (qx[5] > 100) ||
  (qx[6]-qx[1] > 50 && qx[2] > 0)
if (LogC) { ex[which(ex <= 0)] <- NaN
exprs(gset) <- log2(ex) }




# assign samples to groups and set up design matrix
gs <- factor(sml)
groups <- make.names(c("0","1"))
levels(gs) <- groups
gset$group <- gs
design <- model.matrix(~ 0 + group, gset)

colnames(design) <- levels(gs)

gset <- gset[complete.cases(exprs(gset)), ] # skip missing values

fit <- lmFit(gset, design)  # fit linear model

# set up contrasts of interest and recalculate model coefficients
cts <- paste(groups, c(tail(groups, -1), head(groups, 1)), sep="-")
cont.matrix <- makeContrasts(contrasts=cts, levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)

# compute statistics and table of top significant genes
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, coef = 1, adjust="fdr", sort.by="t", number = 54000)

tT <- subset(tT, select=c("ID","adj.P.Val","P.Value","t","Gene.symbol"))


ex_2 = as.data.frame(ex)
dim(ex_2)

ex_2 <- ex_2[complete.cases(ex_2), ]

# Separate the control and disease samples
control_samples <- ex_2[, 1:74]
disease_samples <- ex_2[, 75:161]


# add gene annotations
annotations <- read.table("Downloads/gene_ids",header=TRUE, sep="\t")

ex_2 <- ex_2 %>% 
  rownames_to_column(var = 'ID') %>%
  inner_join(., annotations, by= 'ID')


# merging tT
rownames(tT) <- NULL
colnames(annotations)[colnames(annotations) == 'id'] <- 'ID'
tT <- tT[,1:5]

tT <- tT %>% 
  rownames_to_column(var = 'id') %>%
  inner_join(., annotations, by= 'id')


ex_fil<- ex_2[,c("Gene.ID","z_scores","p_values","adjusted_p_values")]


ex_fil <- ex_fil[ex_fil$adjusted_p_value<0.05,]

col_name <- "Gene.ID"  
ex_fil[, col_name][ex_fil[, col_name] == ""] <- NA
ex_fil <- ex_fil[complete.cases(ex_fil), ]

bing <- read.table("project/n_test",header=TRUE, sep="\t")

# Merge the data frames based on the shared column

colnames(bing)[colnames(bing) == 'id'] <- 'Gene.ID'
merged_df <- merge(tT, annotations, by = "ID", all.x = TRUE)
merged_df <- merged_df[complete.cases(merged_df), ]
ex_fil <- merge(ex_fil, bing, by = "Gene.ID", all.x = TRUE)
ex_fil <- ex_fil[complete.cases(ex_fil), ]

unique_ids <- unique(ex_fil$Gene.ID)


sig <- merged_df[,c(1,5)]

# Group by 'IDs' and calculate the average score for each group
ex_fil_z <- ex_fil[,1:2]

#ex_fil_z$Gene.ID <-  as.numeric(ex_fil_z$Gene.ID)
sig_merged <- sig %>%
  group_by(Gene.ID) %>%
  summarise(t = mean(t)) %>%
  ungroup()



# save signature dataframe
write.csv(sig_merged, file = "t_based_signature.csv")

