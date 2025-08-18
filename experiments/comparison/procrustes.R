library(geomorph)
library(tidyverse)
library(here)
library(stringr)

sim <- read_csv(here("SPMS/experiments/comparison/simdata/leaves.csv"), col_names=TRUE)
print(sim)

# reshape simulated data to fit with geomorph 
sim_mat <- as.matrix(sim[,-1])
reshaped_sim <- sapply(lapply(1:5, function(ii) {
  matrix(sim[ii,-1], byrow = TRUE, nrow = 20, ncol = 2)
}), identity, simplify = "array")
print(reshaped_sim)

# test 
test <- array(unlist(reshaped_sim), dim=c(20,2,5))

# Procrustes align data and export 
proc <- gpagen(test, Proj=FALSE)

proc_rotated <- fixed.angle(proc, angle=45)#rotate.coords(proc, type="flipY")
plot(proc_rotated)


# Rotate landmarks by 45 degrees
rotate_landmarks <- function(coords, angle_degrees) {
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
write.table(proc_final, file=here("SPMS/experiments/comparison/simdata/procrustes_aligned_rotated.csv"), row.names=FALSE, col.names=FALSE, sep=",")


# use geomorph for ancestral state reconstruction 

# load phylogenetic tree
# set names of dimenions