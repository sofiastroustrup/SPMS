# Do ancestral reconstruction using phytools 
library(tidyverse)
library(mvMORPH)
library(ape)
library(here)
library(phytools)

# load simulated data set 
simdata <- as.data.frame(read_csv(here("experiments/comparison/seed=10_sigma=0.5_alpha=0.05_dt=0.05/procrustes_aligned_rotated.csv"), col_names=FALSE))
tree <- read.tree(here("experiments/data/chazot_subtree_rounded.nw"))
tree$tip.label <- c("A", "B", "C", "D", "E") # set names of tips to match simulated data
rownames(simdata) <- tree$tip.label

traits<-mvSIM(tree,nsim=1, model="BMM", param=list(sigma=list(matrix(c(2,1,1,1.5),2,2),
         matrix(c(4,1,1,4),2,2)), names_traits=c("head.size","mouth.size")))

# fit brownian motion model to the simulated data 
BM.fit <- mvBM(tree, simdata, model="BM1", method="pic")
