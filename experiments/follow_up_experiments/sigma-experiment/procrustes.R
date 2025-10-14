library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(reticulate)

d = 2
folder = here("experiments", "sigma-experiment","sigma=0.1") #"sigma=0.1"
plotfolder = paste0(folder, "/procrustes_plots")

#n_species = 5
#n = 20 
#folder ="exp_2_sigma=0.7_alpha=0.025_dt=0.05/seed=3713700383"
sim_files = list.files(path = folder, pattern = "^simdata.*\\.csv$", full.names = TRUE)
z_obs_array <- numeric(length(sim_files)) # Preallocate numeric vector for Z.obs
for (i in 1:length(sim_files)) {
  current_file <- sim_files[i]
  print(paste("Processing file:", current_file))
  
  # Extract seed from folder name
  #seed_name <- basename(current_file)
  
  # Read data from this folder
  sim <- read_csv(file.path(current_file), col_names=FALSE)

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


  # compute integration measure
  integration_measure <- integration.Vrel(proc$coords)
  print(integration_measure$Z.obs)
  z_obs_array[i] <- integration_measure$Z.obs


  output_dir <- file.path(folder)
  dir.create(plotfolder, showWarnings = FALSE, recursive = TRUE)
  
  # Export aligned data
  proc_final <- matrix(aperm(proc$coords, c(3, 1, 2)), nrow = n_species, ncol = n * d)
  ds_part <- str_extract(basename(current_file), "ds=\\d+")
  output_filename <- paste0("procrustes_aligned_", ds_part, ".csv")


  # Save plot
  pdf(file.path(plotfolder, paste0("procrustes_aligned_", ds_part, ".pdf")), width = 10, height = 8)
  plot(proc)
  dev.off()
  
  write.table(proc_final, 
                file = file.path(output_dir, output_filename), 
                row.names = FALSE, col.names = FALSE, sep = ",")

  print(paste("Completed processing file:", current_file))
}






