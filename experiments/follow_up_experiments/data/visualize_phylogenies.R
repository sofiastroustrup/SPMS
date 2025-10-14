library(ape)
library(phytools)
library(here)

# function for visualizing root edge
root.to.singleton<-function(tree){
  if(!inherits(tree,"phylo"))
    stop("tree should be object of class \"phylo\".")
  if(!is.null(tree$root.edge)){
    tree$edge[tree$edge>Ntip(tree)]<-
      tree$edge[tree$edge>Ntip(tree)]+1
    if(attr(tree,"order")%in%c("postorder","pruningwise")){
      tree$edge<-rbind(tree$edge,c(1,2)+Ntip(tree))
      tree$edge.length<-c(tree$edge.length,tree$root.edge)
    } else {
      tree$edge<-rbind(c(1,2)+Ntip(tree),tree$edge)
      tree$edge.length<-c(tree$root.edge,tree$edge.length)
    }
    tree$root.edge<-NULL
    tree$Nnode<-tree$Nnode+1
    if(!is.null(tree$node.label)) 
      tree$node.label<-c("",tree$node.label)
  }
  tree
}

# visualize tree for rank test (simulation and inference) height = 7,6
tree <- read.tree(here("experiments", "data", "chazot_subtree.nw"))
stree<-root.to.singleton(tree) # needed to visualized root edge
pdf(file = here("experiments", "data","plots", "sim_phylo_ranktest.pdf"), width = 9, height = 9)
plot(stree, show.node.label = FALSE, show.tip.label=FALSE, root.edge = TRUE, offset=3,  align.tip.label=TRUE,  type='cladogram') #,
edgelabels(round(stree$edge.length,1), bg="white", col="black", font=5, cex=2.5, offset=1)
dev.off()

# visualize trees for root experiment (simulations)
mixed <- read.tree("experiments/data/chazot_subtree.nw")
pdf(file = here("experiments", "data","plots", "sim_phylo_root_est_mixed.pdf"), width = 9, height = 9)
plot(mixed, show.node.label = FALSE, show.tip.label=FALSE, offset=3,  align.tip.label=TRUE, direction='rightwards', type='cladogram') #,
edgelabels(round(mixed$edge.length,1), bg="white", col="black", font=5, offset=1, cex=2.5)
dev.off()

asym_tree <- read.tree("experiments/data/asymmetric_sim.nw")
pdf(file = here("experiments", "data","plots", "sim_phylo_root_est_asym.pdf"), width = 9, height = 9)
plot(asym_tree, show.node.label = FALSE, show.tip.label=FALSE, root.edge = TRUE, offset=3,  align.tip.label=TRUE, direction='rightwards', type="cladogram") #type='cladogram',
edgelabels(round(asym_tree$edge.length,1), bg="white", col="black", font=5, offset=1, cex=2.5)
dev.off()

sym_tree <- read.tree("experiments/data/symmetric_sim.nw")
pdf(file = here("experiments", "data","plots", "sim_phylo_root_est_sym.pdf"), width = 9, height = 9)
plot(sym_tree, show.node.label = FALSE, show.tip.label=FALSE, root.edge = TRUE, offset=3,  align.tip.label=TRUE, direction='rightwards', type="cladogram") #type='cladogram',
edgelabels(round(sym_tree$edge.length,1), bg="white", col="black", font=5, offset=1, cex=2.5)
dev.off()

# visualize trees for root experiment (inference)
mixed_inf_ <- read.tree("experiments/data/topologies_for_plotting/chazot_subtree_rb=2.nw")
mixed_inf<-root.to.singleton(mixed_inf_) # needed to visualized root edge
pdf(file = here("experiments", "data","plots", "inf_phylo_root_est_mixed.pdf"), width = 9, height = 9)
plot(mixed_inf, show.node.label = FALSE, show.tip.label=FALSE, offset=3,  align.tip.label=TRUE, direction='rightwards', type='cladogram') #,
edgelabels(round(mixed_inf$edge.length,1), bg="white", col="black", font=5, offset=1, cex=1.5)
dev.off()

asym_inf_ <- read.tree("experiments/data/topologies_for_plotting/asymmetric_sim_rb=2.nw")
asym_inf<-root.to.singleton(asym_inf_) # needed to visualized root edge
pdf(file = here("experiments", "data","plots", "inf_phylo_root_est_asym.pdf"), width = 9, height = 9)
plot(asym_inf, show.node.label = FALSE, show.tip.label=FALSE, root.edge = TRUE,  align.tip.label=TRUE, direction='rightwards', type="cladogram") #type='cladogram',
edgelabels(round(asym_inf$edge.length,1), bg="white", col="black", font=5, offset=1, cex=1.5)
dev.off()

sym_inf_ <- read.tree("experiments/data/topologies_for_plotting/symmetric_sim_rb=2.nw")
sym_inf<-root.to.singleton(sym_inf_) # needed to visualized root edge
pdf(file = here("experiments", "data","plots", "inf_phylo_root_est_sym.pdf"), width = 9, height = 9)
plot(sym_inf, show.node.label = FALSE, show.tip.label=FALSE, root.edge = TRUE, offset=3,  align.tip.label=TRUE, direction='rightwards', type="cladogram") #type='cladogram',
edgelabels(round(sym_inf$edge.length,1), bg="white", col="black", font=5, offset=1, cex=1.5)
dev.off()

#nodelabels(text="x",node=1+Ntip(stree),
#           frame="none",adj=c(2,0.2), col="darkgreen")

#nodelabels(text=0:(stree$Nnode-2),node=1:(stree$Nnode+Ntip(tree)),
#           frame="none",adj=c(0.7,-0.4))

# visualize tree for simulation study, inference
plot(stree, show.node.label = FALSE, root.edge = FALSE,type="cladogram", offset=3)
edgelabels(round(tree$edge.length,1), bg="white", col="grey", font=8, offset=0)
nodelabels(text=c(1,2,6),node=2:stree$Nnode+Ntip(stree),
           frame="none",adj=c(0.7,-0.4))
nodelabels(text="x_0",node=1+Ntip(stree),
           frame="none",adj=c(2,0.2), col="darkgreen")
#edgelabels(round(tree$edge.length,1), bg="white", col="grey", font=2, offset=3)

# visualize tree for simulation 
plot(stree, show.node.label = FALSE, root.edge=TRUE, offset=3, type='cladogram')
edgelabels(round(tree$edge.length,1), bg="white", col="grey", font=8, offset=0)
nodelabels(text=c(0,1,2,6),node=1:tree$Nnode+Ntip(stree),
           frame="none",adj=c(0.7,-0.4))



# full tree ---------------------------------------------------------------
ftree <- read.tree("_chazot/data/levelorder_chazot_full_tree.nw")
ctree <- root.to.singleton(ftree)

# visualize tree for simulation 
plot(ctree, show.node.label = FALSE, root.edge=TRUE, offset=3, type='phylogram')
#edgelabels(round(ctree$edge.length,1), bg="white", col="grey", font=8, offset=0)
nodelabels(text=c(0,1,2,6),node=1:ftree$Nnode+Ntip(ctree),
frame="none",adj=c(0.7,-0.4))

