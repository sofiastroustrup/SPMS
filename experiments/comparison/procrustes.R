library(geomorph)
library(tidyverse)


sim <- read_csv("simdata/leaves.csv", col_names=TRUE)
print(sim)

# reshape to match geomorph: n_landmarks x dim x species 


# Procrustes align data and export 