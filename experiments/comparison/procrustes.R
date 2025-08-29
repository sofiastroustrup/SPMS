library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)
n = 20 
d = 2
n_species = 5

folder ="exp_2_sigma=0.7_alpha=0.025_dt=0.05/seed=905103247"
sim <- read_csv(here(paste0("experiments/comparison/", folder, "/leaves.csv")), col_names=FALSE)
print(sim)

# reshape simulated data to fit with geomorph:  n_landmarks x n_dim x n_specimens
sim_mat <- as.matrix(sim)
reshaped_sim <- sapply(lapply(1:n_species, function(ii) {
  matrix(sim[ii,], byrow = TRUE, nrow = n, ncol = d)
}), identity, simplify = "array")
print(reshaped_sim)

# test
test <- array(unlist(reshaped_sim), dim=c(n,d,n_species))

# Procrustes align data and export
proc <- gpagen(test, Proj=FALSE, ProcD = FALSE)
print(proc)
#proc_rotated <- fixed.angle(proc, angle=45)#rotate.coords(proc, type="flipY")
pdf(here(paste0("experiments/comparison/", folder, "/procrustes_plot.pdf")),
    width = 10, height = 8)
plot(proc)
dev.off()
# export procrustes aligned and rotated data 
proc_final <- array_reshape(aperm(proc$coords, c(3, 1, 2)), c(n_species, n*d))
write.table(proc_final, file=here(paste0("experiments/comparison/", folder, "/procrustes_aligned.csv")), row.names=FALSE, col.names=FALSE, sep=",")

