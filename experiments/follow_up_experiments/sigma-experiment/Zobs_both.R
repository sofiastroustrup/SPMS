library(geomorph)
library(readr)
library(dplyr)
library(here)
library(stringr)
library(ggplot2)

d = 2
parent_folder = here("experiments", "sigma-experiment")
sigma_folders = list.dirs(parent_folder, recursive = FALSE, full.names = TRUE)

integration_measures_list_procrustes <- list()
integration_measures_list_noalign <- list()

for (sigma_folder in sigma_folders) {
  sigma_value <- str_extract(basename(sigma_folder), "\\d+\\.\\d+")
  sim_files = list.files(path = sigma_folder, pattern = "^simdata.*\\.csv$", full.names = TRUE)
  z_obs_array_procrustes <- numeric(length(sim_files))
  z_obs_array_noalign <- numeric(length(sim_files))
  
  for (i in seq_along(sim_files)) {
    current_file <- sim_files[i]
    sim <- read_csv(current_file, col_names = FALSE)
    n_species <- nrow(sim)
    n <- ncol(sim) / d
    sim_mat <- as.matrix(sim)
    
    # Procrustes-aligned integration measure
    reshaped_sim <- sapply(lapply(1:n_species, function(ii) {
      matrix(sim[ii,], byrow = TRUE, nrow = n, ncol = d)
    }), identity, simplify = "array")
    reshaped_data <- array(unlist(reshaped_sim), dim = c(n, d, n_species))
    proc <- gpagen(reshaped_data, Proj = FALSE, ProcD = FALSE)
    integration_measure_procrustes <- integration.Vrel(proc$coords)
    z_obs_array_procrustes[i] <- integration_measure_procrustes$Z.obs

    # Non-aligned integration measure
    integration_measure_noalign <- integration.Vrel(sim_mat)
    z_obs_array_noalign[i] <- integration_measure_noalign$Z.obs
  }
  
  integration_measures_list_procrustes[[sigma_value]] <- z_obs_array_procrustes
  integration_measures_list_noalign[[sigma_value]] <- z_obs_array_noalign
}

# Prepare data for plotting
integration_df_procrustes <- stack(integration_measures_list_procrustes)
colnames(integration_df_procrustes) <- c("Z.obs", "sigma")
integration_df_procrustes$method <- "Procrustes aligned"

integration_df_noalign <- stack(integration_measures_list_noalign)
colnames(integration_df_noalign) <- c("Z.obs", "sigma")
integration_df_noalign$method <- "No alignment"

integration_df <- rbind(integration_df_procrustes, integration_df_noalign)
#integration_df$sigma <- as.numeric(as.character(integration_df$sigma))

# Plot
pdf(file = here("experiments", "sigma-experiment", "integration_vs_sigma_comparison.pdf"), width = 20, height = 10)
ggplot(integration_df, aes(x = sigma, y = Z.obs)) +
  geom_jitter(width = 0.05, alpha = 0.6, size = 2) +
  labs(
    x = expression(sigma),
    y = "Z-score",
    #title = expression("Integration measure vs. " * sigma)
  ) +
  facet_wrap(~method, nrow = 1) +
  theme_bw(base_size = 12) +
  theme(
    strip.text = element_text(size = 28),
    plot.title = element_text(hjust = 0.5, size = 30),
    axis.title.x = element_text(size = 28),
    axis.title.y = element_text(size = 18),
    axis.text.x = element_text(size = 20, angle = 45, vjust = 1, hjust = 1),
    axis.text.y = element_text(size = 16),
    legend.position = "none"
    #legend.title = element_blank(),
    #legend.text = element_text(size = 14)
  )   #scale_x_continuous(labels = function(x) sprintf("%.2f", x))
dev.off()
