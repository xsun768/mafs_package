#!/usr/bin/env Rscript

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]
weight <- args[3]
n_cores <- as.integer(args[4])

# Load library
library(Ball)

# Read data
data <- read.csv(input_file)
X <- as.matrix(data[, -ncol(data)])
y <- data[, ncol(data)]

# Compute BCOR
n_features <- ncol(X)
scores <- numeric(n_features)

for (i in 1:n_features) {
  tryCatch({
    result <- Ball::bcor.test(X[, i], y)
    scores[i] <- result$statistic
  }, error = function(e) {
    scores[i] <- 0
  })
}

# Save results
output <- data.frame(
  feature_index = 0:(n_features - 1),
  chisquare = scores,
  constant = scores,
  probability = scores
)

write.csv(output, output_file, row.names = FALSE)
