library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)

d = 2
folder ="sigma=0.5_alpha=0.005_dt=0.05"
#n_species = 5
#n = 20 
#folder ="exp_2_sigma=0.7_alpha=0.025_dt=0.05/seed=3713700383"
sim_folders = list.files(path = here("experiments", "procrustes",folder), full.names = TRUE)
print(sim_folders)
for (i in 1:length(sim_folders)) {
  current_folder <- sim_folders[i]
  print(paste("Processing folder:", current_folder))
  
  # Extract seed from folder name
  seed_name <- basename(current_folder)
  
  # Read data from this folder
  sim <- read_csv(file.path(current_folder, "leaves.csv"), col_names=FALSE)
  
  # Get dimensions
  n_species <- nrow(sim)
  n <- ncol(sim) / d  # Number of landmarks = total columns / dimensions
  
  # Process data for this folder
  sim_mat <- as.matrix(sim)
  reshaped_sim <- sapply(lapply(1:n_species, function(ii) {
    matrix(sim[ii,], byrow = TRUE, nrow = n, ncol = d)
  }), identity, simplify = "array")
  
  reshaped_data <- array(unlist(reshaped_sim), dim=c(n, d, n_species))
  
  # Procrustes alignment
  proc <- gpagen(reshaped_data, Proj=FALSE, ProcD = FALSE)

  output_dir <- file.path(current_folder)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save plot
  pdf(file.path(output_dir, "procrustes_plot.pdf"), width = 10, height = 8)
  plot(proc)
  dev.off()
  
  # Export aligned data
  proc_final <- array_reshape(aperm(proc$coords, c(3, 1, 2)), c(n_species, n*d))
  write.table(proc_final, 
              file=file.path(output_dir, "procrustes_aligned.csv"), 
              row.names=FALSE, col.names=FALSE, sep=",")
  
  print(paste("Completed processing folder:", current_folder))
}






