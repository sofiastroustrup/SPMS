library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)
n = 20 
d = 2
n_species = 5

folder ="exp_2_sigma=0.7_alpha=0.05_dt=0.05/seed=658822120"
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

# rotate proc_final 45 degrees and export
# Function to rotate coordinates clockwise
rotate_clockwise <- function(coords, angle_degrees = 45) {
  # Convert angle to radians (negative for clockwise rotation)
  angle_rad <- -angle_degrees * pi / 180
  
  # Create rotation matrix
  rot_matrix <- matrix(
    c(cos(angle_rad), sin(angle_rad),
      -sin(angle_rad), cos(angle_rad)),
    nrow = 2, byrow = TRUE
  )
  
  # Apply rotation to the coordinates
  rotated_coords <- coords
  
  # Loop through each specimen
  for (i in 1:nrow(coords)) {
    # Reshape to get landmarks as x,y pairs
    coords_xy <- matrix(coords[i,], ncol = 2, byrow = TRUE)
    
    # Apply rotation
    rotated_xy <- coords_xy %*% rot_matrix
    
    # Put back into the same format
    rotated_coords[i,] <- as.vector(t(rotated_xy))
  }
  
  return(rotated_coords)
}

# Rotate the proc_final coordinates 45 degrees clockwise
proc_final_rotated <- rotate_clockwise(proc_final, angle_degrees = 45)

# Export the rotated data
write.table(proc_final_rotated, 
            file=here(paste0("experiments/comparison/", folder, "/procrustes_aligned_rotated45.csv")), 
            row.names=FALSE, col.names=FALSE, sep=",")

# Visualize original vs rotated coordinates
pdf(here(paste0("experiments/comparison/", folder, "/procrustes_rotated_comparison.pdf")),
    width = 12, height = 6)

par(mfrow=c(1,2))

# Plot original Procrustes coordinates
plot(NULL, xlim=c(-0.3, 0.3), ylim=c(-0.3, 0.3), 
     main="Original Procrustes Alignment", xlab="x", ylab="y", asp=1)
for (i in 1:nrow(proc_final)) {
  coords <- matrix(proc_final[i,], ncol=2, byrow=TRUE)
  lines(coords, type="o", pch=20, cex=0.8, col=rainbow(nrow(proc_final))[i])
}
grid()

# Plot rotated Procrustes coordinates
plot(NULL, xlim=c(-0.3, 0.3), ylim=c(-0.3, 0.3), 
     main="Rotated 45° Clockwise", xlab="x", ylab="y", asp=1)
for (i in 1:nrow(proc_final_rotated)) {
  coords <- matrix(proc_final_rotated[i,], ncol=2, byrow=TRUE)
  lines(coords, type="o", pch=20, cex=0.8, col=rainbow(nrow(proc_final_rotated))[i])
}
grid()

dev.off()
