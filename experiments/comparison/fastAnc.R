# Do ancestral reconstruction using phytools 
library(tidyverse)
library(geomorph)
library(ape)
library(here)
library(phytools)

# load simulated data set 
prepath = ""#"Library/CloudStorage/OneDrive-UniversityofCopenhagen/SPMS"
folder = "full_tree_sigma=0.6_alpha=0.01_dt=0.05"
sim_seed = "seed=288650342"
simdata <- read_csv(here("experiments", "comparison", folder, sim_seed, "procrustes_aligned_rotated45.csv"), col_names=FALSE)%>% t()
tree <- read.tree(here("experiments", "data", "chazot_full_tree_rounded.nw")) #read.tree(here(paste0(prepath, "/experiments/data/chazot_subtree_rounded.nw")))
colnames(simdata) <- tree$tip.label
print(simdata)

# do ancestral reconstruction
# fastAnc uses felsensteins pruning algorithm and reroots the tree to get MLE
# sigma^2 is not needed to estimate the mean
#anc_recon <- fastAnc(tree, simdata[1,], model="BM", CI=TRUE)
n_traits <- nrow(simdata)
n_tips <- length(tree$tip.label)
n_nodes <- tree$Nnode

# Create empty matrix to store results
# Columns: internal nodes, Rows: traits
CI <- list()
anc_matrix <- matrix(NA, nrow=n_traits, ncol=n_nodes)
colnames(anc_matrix) <- (n_tips + 1):(n_tips + n_nodes)  # Node IDs
rownames(anc_matrix) <- paste0("trait", 1:n_traits)      # Trait names
print(anc_matrix)

#create output folder 
output_folder <- here("experiments/comparison", folder, sim_seed, "fastAnc")
print(output_folder)
dir.create(output_folder, showWarnings = TRUE, recursive = TRUE)

# Loop through all traits in simdata
for (i in 1:n_traits) {
  # Extract trait vector and ensure it has names
  trait_vector <- simdata[i,]
  names(trait_vector) <- tree$tip.label
  
  # Perform ancestral state reconstruction
  anc_result <- fastAnc(tree, trait_vector, model="BM", CI=TRUE)
  write.csv(anc_result$CI95, col.names=FALSE,file=here("experiments", "comparison", folder, sim_seed, paste0("fastAnc/95%_conf_trait",i ,".csv")))

  # Store the results in the matrix
  anc_matrix[i,] <- anc_result$ace
  #CI[[i]] <- anc_result$CI  # Store confidence intervals for each trait
}

# export reconstructed states as csv 
write.csv(anc_matrix, file=paste0(output_folder, "/fastAnc_recon.csv"))



########################################
# Plot all ancestral states side by side
########################################

# Create output directory if it doesn't exist
output_folder <- here(paste0("experiments/comparison/", folder, "/plots"))
dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)

# Save the grid of ancestral shape plots to PDF
pdf(file.path(output_folder, "ancestral_shapes_grid.pdf"), 
    width = 10, height = 8)  # Adjust width and height as needed
# First, determine how many nodes we have
n_nodes <- ncol(anc_matrix)
node_ids <- colnames(anc_matrix)
n_landmarks <- nrow(anc_matrix) / 2  # Assuming alternating x,y coordinates

# Set up a plotting grid based on number of nodes
par(mfrow=c(2, ceiling(n_nodes/2)), mar=c(4, 4, 2, 1))

# Plot each node's shape reconstruction
for (i in 1:n_nodes) {
  # Extract x and y coordinates for this node
  x_coords <- anc_matrix[seq(1, nrow(anc_matrix), 2), i]
  y_coords <- anc_matrix[seq(2, nrow(anc_matrix), 2), i]
  
  # Create the plot
  plot(x_coords, y_coords, 
       main=paste("Node", node_ids[i]), 
       xlab="X", ylab="Y",
       pch=19, col="blue", 
       asp=1)  # asp=1 ensures proper shape visualization
  
  # Connect points to show the shape outline
  lines(c(x_coords, x_coords[1]), c(y_coords, y_coords[1]), 
        col="darkblue", lwd=1.5)
}
dev.off()

# Reset the plotting layout
par(mfrow=c(1,1))
pdf(file.path(output_folder, "ancestral_shapes.pdf"), 
    width = 10, height = 8)
# Create a single plot with all shapes overlaid
# Use a different color for each node
colors <- rainbow(n_nodes)

# Create an empty plot with appropriate boundaries
all_x <- as.vector(anc_matrix[seq(1, nrow(anc_matrix), 2),])
all_y <- as.vector(anc_matrix[seq(2, nrow(anc_matrix), 2),])
plot(NULL, xlim=range(all_x), ylim=range(all_y),
     main="All Ancestral Shapes", 
     xlab="X coordinate", ylab="Y coordinate", 
     asp=1)

# Add each node's shape with a different color

for (i in 1:n_nodes) {
  x_coords <- anc_matrix[seq(1, nrow(anc_matrix), 2), i]
  y_coords <- anc_matrix[seq(2, nrow(anc_matrix), 2), i]
  
  # Add shape with distinct color
  lines(c(x_coords, x_coords[1]), c(y_coords, y_coords[1]), 
        col=colors[i], lwd=2)
  points(x_coords, y_coords, pch=19, col=colors[i], cex=0.8)
}
# Add a legend to identify each node
legend("topright", legend=paste("Node", node_ids), 
       col=colors, lwd=2, pch=19, cex=0.8)
dev.off()
# Also add a reference plot of the tree with node numbers
plot(tree, main="Phylogenetic Tree with Node IDs")
nodelabels(bg="white")
tiplabels(bg="lightblue")


