# Do ancestral reconstruction using phytools 
library(tidyverse)
library(geomorph)
library(ape)
library(here)
library(phytools)

# load simulated data set 
simdata <- read_csv(here("SPMS/experiments/comparison/simdata/procrustes_aligned_rotated.csv"), col_names=FALSE)%>% t()
tree <- read.tree(here("SPMS/experiments/data/chazot_subtree.nw"))
#tree <- read.newick(here("SPMS/experiments/data/chazot_subtree.nw")) 
tree$tip.label <- c("A", "B", "C", "D", "E") # set names of tips to match simulated data
colnames(simdata) <- tree$tip.label
print(simdata)

# do ancestral reconstruction
# fastAnc uses felsensteins pruning algorithm and reroots the tree to get MLE
# sigma^2 is not needed to estimate the mean
anc_recon <- fastAnc(tree, simdata[1,], model="BM")


n_traits <- nrow(simdata)
n_tips <- length(tree$tip.label)
n_nodes <- tree$Nnode

# Create empty matrix to store results
# Columns: internal nodes, Rows: traits
anc_matrix <- matrix(NA, nrow=n_traits, ncol=n_nodes)
colnames(anc_matrix) <- (n_tips + 1):(n_tips + n_nodes)  # Node IDs
rownames(anc_matrix) <- paste0("trait", 1:n_traits)      # Trait names

# Loop through all traits in simdata
for (i in 1:n_traits) {
  # Extract trait vector and ensure it has names
  trait_vector <- simdata[i,]
  names(trait_vector) <- tree$tip.label
  
  # Perform ancestral state reconstruction
  anc_result <- fastAnc(tree, trait_vector, model="BM")
  
  # Store the results in the matrix
  anc_matrix[i,] <- anc_result
}
write.csv(anc_matrix, file=here("SPMS/experiments/comparison/simdata/anc_recon.csv"))

# plot different ancestral states 
plot(anc_matrix[,2][seq(1, 40, 2)], anc_matrix[,1][seq(2, 40, 2)])

# plot tree to see which nodes numbers refer to 
plot(tree)
nodelabels()  # 
