library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)
library(fs)  # For directory operations

# Parameters for the data
n = 20 
d = 2
p = 5

# Get list of all simulation folders in the comparison directory
base_dir <- here("experiments/comparison/exp_1_sigma=0.5_alpha=0.025_dt=0.05")
sim_folders <- dir_ls(base_dir, type = "directory", regexp = "seed=")

# Loop over all simulation folders
for (folder_path in sim_folders) {
  # Extract just the folder name from the path
  folder <- basename(folder_path)
  
  # Print current folder being processed
  cat("\nProcessing folder:", folder, "\n")
  
  # Check if leaves.csv exists in this folder
  leaves_path <- here(paste0(base_dir, "/",folder, "/leaves.csv"))
  print(leaves_path)
  if (!file.exists(leaves_path)) {
    cat("  Skipping: leaves.csv not found\n")
    next
  }
  
  # Read the simulated data
  tryCatch({
    sim <- read_csv(leaves_path, col_names=FALSE)
    print(sim)
    cat("  Data dimensions:", dim(sim)[1], "x", dim(sim)[2], "\n")
    
    # Reshape simulated data to fit with geomorph 
    #sim_mat <- as.matrix(sim)#as.matrix(sim[,-1])
    reshaped_sim <- sapply(lapply(1:dim(sim)[1], function(ii) {
      matrix(sim[ii,], byrow = TRUE, nrow = n, ncol = d)
    }), identity, simplify = "array")
    
    # Convert to proper array for gpagen
    landmark_array <- array(unlist(reshaped_sim), dim=c(n, d, nrow(sim)))
    
    # Procrustes align data
    cat("  Performing Procrustes alignment...\n")
    proc <- gpagen(landmark_array, Proj=FALSE)
    
    # Save the plot
    plot_path <- here(paste0(base_dir, "/", folder, "/procrustes_plot.pdf"))
    pdf(plot_path, width = 10, height = 8)
    plot(proc)
    dev.off()
    cat("  Saved plot to:", plot_path, "\n")
    
    # Export procrustes aligned data
    proc_final <- array_reshape(aperm(proc$coords, c(3, 1, 2)), c(nrow(sim), n*d))
    output_path <- here(paste0(base_dir, "/", folder, "/procrustes_aligned.csv"))
    write.table(proc_final, file=output_path, row.names=FALSE, col.names=FALSE, sep=",")
    cat("  Saved aligned data to:", output_path, "\n")
    
    # Optional: Also create a rotated version
    # Uncomment if you want this functionality
    # rotate_landmarks <- function(coords, angle_degrees) {
    #   angle_rad <- angle_degrees * pi / 180
    #   rot_matrix <- matrix(c(cos(angle_rad), -sin(angle_rad), 
    #                        sin(angle_rad), cos(angle_rad)), 
    #                      nrow=2, byrow=TRUE)
    #   n_specimens <- dim(coords$coords)[3]
    #   rotated_coords <- coords
    #   for(i in 1:n_specimens) {
    #     specimen_coords <- coords$coords[,,i]
    #     rotated_coords$coords[,,i] <- specimen_coords %*% rot_matrix
    #   }
    #   return(rotated_coords)
    # }
    # 
    # proc_45deg <- rotate_landmarks(proc, 45)
    # proc_final_rotated <- array_reshape(aperm(proc_45deg$coords, c(3, 1, 2)), c(nrow(sim), n*d))
    # rotated_output_path <- here(paste0("experiments/comparison/", folder, "/procrustes_aligned_rotated.csv"))
    # write.table(proc_final_rotated, file=rotated_output_path, row.names=FALSE, col.names=FALSE, sep=",")
    # cat("  Saved rotated aligned data to:", rotated_output_path, "\n")
    
  }, error = function(e) {
    cat("  ERROR processing folder:", folder, "\n")
    cat("  Error message:", e$message, "\n")
  })
}

cat("\nAll folders processed.\n")