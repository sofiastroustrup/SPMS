# Do ancestral reconstruction using phytools 
library(tidyverse)
library(geomorph)
library(ape)
setwd("/Users/lkn315/Desktop")

# load simulated data set 
simdata <- read.csv("flat_true_tree.csv", header=FALSE)
leaf_idx <- c(3, 4, 5, 7, 8)+1 # we add one because R is 1 indexed 
leaves <- slice(simdata, leaf_idx)

tree <- read.tree("../data/chazot_subtree.nw")

# procrustes align simulated data 
gpa_simdata <- gpagen(leaves, print.progress = T,Proj=F) # # GPA-alignment

# do ancestral reconstruction
anc_recon <- fastAnc(tree, gpa_simdata$coords, model="BM")
