
# BCOR Feature Selection using Ball package
# Simplified version without doParallel

suppressPackageStartupMessages({
    library(Ball)
})

save_bcor_sis_results <- function(input_file, output_file, method="standard", weight="chisquare") {
    # Load data
    data <- read.csv(input_file, header = FALSE)
    X <- as.matrix(data[, -ncol(data)])
    y <- data[, ncol(data)]
    nsis <- ncol(X)
    
    cat(sprintf("Data loaded: %d samples, %d features\n", nrow(X), nsis), file=stderr())
    cat("Processing sequentially (no parallel)\n", file=stderr())
    
    # Run bcorsis on all features at once
    cat("Computing Ball correlation...\n", file=stderr())
    bcor_result <- bcorsis(X, y, method = method, d = nsis, weight = weight)
    
    # Extract statistics
    results <- bcor_result$complete.info$statistic
    colnames(results) <- c("constant", "probability", "chisquare")
    
    cat(sprintf("Results dimension: %d x %d\n", nrow(results), ncol(results)), file=stderr())
    cat(sprintf("Score range (%s): [%.6f, %.6f]\n", weight, 
                min(results[, weight]), max(results[, weight])), file=stderr())
    
    # Write results to file
    write.csv(results, file = output_file, row.names = FALSE)
    
    return(results)
}

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    stop("Usage: Rscript bcorsis.R <input_file> <output_file> [method] [weight]")
}

input_file <- args[1]
output_file <- args[2]
method <- if(length(args) > 2) args[3] else "standard"
weight <- if(length(args) > 3) args[4] else "chisquare"

# Print processing information
cat("BCOR Feature Selection (Sequential)\n", file=stderr())
cat("====================================\n", file=stderr())
cat(sprintf("Input file: %s\n", input_file), file=stderr())
cat(sprintf("Output file: %s\n", output_file), file=stderr())
cat("====================================\n", file=stderr())

# Execute main function
result <- save_bcor_sis_results(input_file, output_file, method, weight)

cat("Processing complete!\n", file=stderr())