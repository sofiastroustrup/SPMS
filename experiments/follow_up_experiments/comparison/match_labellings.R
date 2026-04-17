library(tidyverse)
library(ape)
library(here)
library(phytools)

# Read both trees
tree_phytools <- tree <- read.tree(here("experiments", "follow_up_experiments", "data", "chazot_subtree_rounded.nw")) 
tree_preorder <- tree <- read.tree(here("experiments", "follow_up_experiments", "data", "chazot_subtree_levelorder.nw")) 

# plot tree with phytools node numbers
png(filename = here("experiments", "follow_up_experiments", "comparison","_test", "tree_with_phytools_indices.png"), width = 2800, height = 2200)
plot(tree_phytools, no.margin=TRUE,edge.width=2,type="fan",cex=0.9)
nodelabels(frame="none",adj=c(1.1,-0.4))
dev.off()
print("Tree with phytools node indices plotted and saved as tree_with_phytools_indices.png")

png(filename = here("experiments", "follow_up_experiments", "comparison","_test", "tree_with_levelorder_indices.png"), width = 2800, height = 2200)
plot(tree_preorder, no.margin=TRUE,edge.width=2,type="fan",cex=0.9)
nodelabels(tree_preorder$node.label, 
           frame = "none", 
           adj = c(1.1, -0.4),
           cex = 0.7,
           col = "red")
dev.off()
print("Tree with levelorder node indices plotted and saved as tree_with_levelorder_indices.png")


# Check dimensions
n_tips <- length(tree_phytools$tip.label)
n_nodes <- tree_phytools$Nnode

print(paste("Number of tips:", n_tips))
print(paste("Number of internal nodes:", n_nodes))
print(paste("phytools node numbering: ", n_tips + 1, "to", n_tips + n_nodes))

# Create mapping between preorder labels and phytools indices
node_mapping <- tibble(
  phytools_index = (n_tips + 1):(n_tips + n_nodes),
  preorder_label = tree_preorder$node.label
)

print(node_mapping,n=30)

# Export mapping
write_csv(node_mapping, here("experiments", "follow_up_experiments", "comparison", "_test", "node_label_mapping.csv"))
print("Mapping exported to node_label_mapping.csv")

