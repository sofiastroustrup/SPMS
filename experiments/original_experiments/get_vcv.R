#! /usr/bin/Rscript
library("ape")
library("here")
setwd(here())

# parse arguments 
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)

# do what we want 
tree = read.tree(paste(args[1], '.nw', sep=''))
vcv_ = vcv(tree)
write.table(vcv_, file=paste(args[1],'_vcv.csv', sep=''), row.names=F, col.names=F)
