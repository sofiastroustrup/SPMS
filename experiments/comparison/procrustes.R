library(geomorph)
library(here)


sim <- read.table(here("experiments/comparison/simdata/leaves.csv"), header=TRUE)
print(sim)

# reshape to match geomorph: n_landmarks x dim x species 


# Procrustes align data and export 