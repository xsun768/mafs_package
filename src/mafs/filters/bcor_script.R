library(Ball)

save_bcor_sis_results <- function(input_file, output_file, method="standard", 
                                   weight="chisquare") {
    data <- read.csv(input_file)
    X <- as.matrix(data[, -ncol(data)])
    y <- data[, ncol(data)]
    
    bcor_result <- bcorsis(X, y, method = method, d = ncol(X), weight = weight)
    
    complete_statistics <- bcor_result$complete.info$statistic
    colnames(complete_statistics) <- c("constant", "probability", "chisquare")
    
    write.csv(complete_statistics, file = output_file, row.names = FALSE)
    
    return(complete_statistics)
}

args = commandArgs(trailingOnly = TRUE)
input_file = args[1]
output_file = args[2]
method = if(length(args) > 2) args[3] else "standard"
weight = if(length(args) > 3) args[4] else "chisquare"

result <- save_bcor_sis_results(input_file, output_file, method, weight)
print("BCOR processing complete")
