library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)
n = 20 
d = 2

folder ="seed=2557803684_sigma=0.7_alpha=0.025_dt=0.05"#"seed=4098652401_sigma=0.3_alpha=0.1_dt=0.01" #"seed=1259603298_sigma=0.5_alpha=0.1_dt=0.05"
sim <- read_csv(here(paste0("experiments/comparison/", folder, "/leaves.csv")), col_names=TRUE)
print(dim(sim))

# reshape simulated data to fit with geomorph 
sim_mat <- as.matrix(sim[,-1])
reshaped_sim <- sapply(lapply(1:5, function(ii) {
  matrix(sim[ii,-1], byrow = TRUE, nrow = n, ncol = d)
}), identity, simplify = "array")
print(reshaped_sim)

# test 
test <- array(unlist(reshaped_sim), dim=c(n,d,5))

# Procrustes align data and export 
proc <- gpagen(test, Proj=FALSE)

#proc_rotated <- fixed.angle(proc, angle=45)#rotate.coords(proc, type="flipY")
pdf(here(paste0("experiments/comparison/", folder, "/procrustes_plot.pdf")),
    width = 10, height = 8)
plot(proc)
dev.off()
# export procrustes aligned and rotated data 
proc_final <- array_reshape(aperm(proc$coords, c(3, 1, 2)), c(5, n*d))
write.table(proc_final, file=here(paste0("experiments/comparison/", folder, "/procrustes_aligned.csv")), row.names=FALSE, col.names=FALSE, sep=",")


# Rotate landmarks by 45 degrees
'''rotate_landmarks <- function(coords, angle_degrees) {
  # Convert degrees to radians
  angle_rad <- angle_degrees * pi / 180
  
  # Create rotation matrix
  rot_matrix <- matrix(c(cos(angle_rad), -sin(angle_rad), 
                         sin(angle_rad), cos(angle_rad)), 
                       nrow=2, byrow=TRUE)
  
  # Get number of specimens
  n_specimens <- dim(coords$coords)[3]
  
  # Create array to store rotated coordinates
  rotated_coords <- coords
  
  # Apply rotation to each specimen
  for(i in 1:n_specimens) {
    # Extract landmarks for specimen i
    specimen_coords <- coords$coords[,,i]
    
    # Rotate landmarks using matrix multiplication
    rotated_coords$coords[,,i] <- specimen_coords %*% rot_matrix
  }
  
  return(rotated_coords)
}

# Apply 45 degree rotation
proc_45deg <- rotate_landmarks(proc, 45)

# Plot original and rotated landmarks
par(mfrow=c(1,2))
plot(proc)
plot(proc_45deg)

# export procrustes aligned and rotated data 
proc_final <- array_reshape(aperm(proc_45deg$coords, c(3, 1, 2)), c(5, 40))
#write.table(proc_final, file=here(paste0("experiments/comparison/", folder, "/procrustes_aligned_rotated.csv")), row.names=FALSE, col.names=FALSE, sep=",")
'''
