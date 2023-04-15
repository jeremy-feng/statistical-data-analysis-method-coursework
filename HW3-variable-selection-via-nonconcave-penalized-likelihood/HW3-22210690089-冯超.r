library(ggplot2)
library(lattice)
library(data.table)
library(progress)
library(MASS) # 多元正态分布
library(caret) # 交叉验证
library(ncvreg) # SCAD, LASSO
library(Matrix)
library(glmnet) # Ridge
library(nnGarrote)

# 生成真实的自变量相关系数矩阵
rho <- matrix(0, 8, 8)
for (i in 1:8) {
    for (j in 1:8) {
        rho[i, j] <- 0.5^abs(i - j)
    }
}
# 生成真实的 beta
beta <- c(3, 1.5, 0, 0, 2, 0, 0, 0)


# 生成自变量和因变量的数据框
generate_data <- function(n, rho, beta, sigma) {
    # 生成多元正态分布的数据
    x <- mvrnorm(n, rep(0, 8), rho)
    # 生成因变量
    y <- x %*% beta + rnorm(n, 0, sigma)
    return(list(x = x, y = y))
}

# 计算 model error
cal_model_error <- function(beta, beta_hat, x) {
    expected_y <- x %*% beta
    expected_y_hat <- x %*% beta_hat
    return(mean((expected_y - expected_y_hat)^2))
}

# SCAD 交叉验证，寻找最优的 gamma 和 lambda
get_best_param_for_scad <- function(
    x, y, gamma_seq, lambda_seq, k = 5) {
    # 生成 k 折交叉验证的验证集索引列表
    validation_index_list <- createFolds(y, k = k, list = TRUE)
    # 对每个 gamma 和 lambda 的网格组合，计算 k 个验证集的 model error 的均值
    mean_me_list <- list()
    for (gamma in gamma_seq) {
        for (lambda in lambda_seq) {
            me_list <- c()
            for (i in 1:k) {
                # 生成训练集和验证集
                train_index <- validation_index_list[-i]
                validation_index <- validation_index_list[[i]]
                x_train <- x[unlist(train_index), ]
                y_train <- y[unlist(train_index)]
                x_validation <- x[unlist(validation_index), ]
                # 用训练集拟合模型
                fit <- ncvfit(x_train, y_train,
                    penalty = "SCAD", gamma = gamma,
                    lambda = lambda
                )
                # 计算验证集上的 model error
                beta_hat <- fit$beta
                me <- cal_model_error(beta, beta_hat, x_validation)
                me_list <- c(me_list, me)
            }
            # 计算 k 个验证集的 model error 的均值
            mean_me_list[[paste(gamma, lambda, sep = ",")]] <- mean(me_list)
        }
    }
    # 找到最优的 gamma 和 lambda
    best_param <- names(which.min(mean_me_list))
    best_param <- unlist(strsplit(best_param, ","))
    best_param <- as.numeric(best_param)
    return(best_param)
}

# LASSO 和 Ridge 交叉验证，寻找最优的 lambda
get_best_param <- function(
    x, y, penalty, lambda_seq, k = 5) {
    # 生成 k 折交叉验证的验证集索引列表
    validation_index_list <- createFolds(y, k = k, list = TRUE)
    # 对每个 lambda 的网格组合，计算 k 个验证集的 model error 的均值
    mean_me_list <- list()
    for (lambda in lambda_seq) {
        me_list <- c()
        for (i in 1:k) {
            # 生成训练集和验证集
            train_index <- validation_index_list[-i]
            validation_index <- validation_index_list[[i]]
            x_train <- x[unlist(train_index), ]
            y_train <- y[unlist(train_index)]
            x_validation <- x[unlist(validation_index), ]
            if (penalty == "lasso") {
                # 用训练集拟合模型
                fit <- ncvfit(x_train, y_train,
                    penalty = "lasso",
                    lambda = lambda,
                )
            } else if (penalty == "ridge") {
                # 用训练集拟合模型
                fit <- glmnet(x_train, y_train,
                    alpha = 0,
                    lambda = lambda,
                    standardize = FALSE,
                    intercept = FALSE,
                )
            } else {
                stop("penalty must be 'lasso' or 'ridge'.")
            }
            # 计算验证集上的 model error
            beta_hat <- fit$beta
            me <- cal_model_error(beta, beta_hat, x_validation)
            me_list <- c(me_list, me)
        }
        # 计算 k 个验证集的 model error 的均值
        mean_me_list[as.character(lambda)] <- mean(me_list)
    }
    # 找到最优的 lambda
    best_param <- names(which.min(mean_me_list))
    best_param <- as.numeric(best_param)
    return(best_param)
}

# 计算 relative model errors
# No. of correct zero coefficients
# 和 No. of incorrect zero coefficients
cal_metrics <- function(
    me_ols,
    me_with_penalty,
    beta,
    beta_hat_with_penalty) {
    rme <- me_with_penalty / me_ols
    correct_zero <- sum(beta_hat_with_penalty == 0 & beta == 0)
    incorrect_zero <- sum(beta_hat_with_penalty == 0 & beta != 0)
    return(list(
        rme = rme, correct_zero = correct_zero,
        incorrect_zero = incorrect_zero
    ))
}

get_result <- function(x, y, penalty, me_ols) {
    # 估计 beta
    if (penalty == "scad_1") {
        # ==========用 SCAD 1 估计 beta==========
        # 先找到最优参数 gamma 和 lambda
        best_param_for_scad_1 <- get_best_param_for_scad(x,
            y,
            gamma_seq = seq(2.1, 4, 0.1),
            lambda_seq = c(0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 3, 5, 10), k = 5
        )
        # 将最优参数 gamma 和 lambda 代入模型，得到估计的 beta
        beta_hat_with_penalty <- ncvfit(x, y,
            penalty = "SCAD", gamma = best_param_for_scad_1[1],
            lambda = best_param_for_scad_1[2]
        )$beta
    } else if (penalty == "scad_2") {
        # 先找到最优参数 lambda，将 gamma 固定为 3.7
        best_param_for_scad_2 <- get_best_param_for_scad(x, y,
            gamma_seq = seq(3.7, 3.7),
            lambda_seq = c(0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 3, 5, 10), k = 5
        )
        # 将最优参数 gamma 和 lambda 代入模型，得到估计的 beta
        beta_hat_with_penalty <- ncvfit(x, y,
            penalty = "SCAD", gamma = best_param_for_scad_2[1],
            lambda = best_param_for_scad_2[2]
        )$beta
    } else if (penalty == "lasso") {
        # 先找到最优参数 lambda
        best_param_for_lasso <- get_best_param(x, y,
            penalty = "lasso",
            lambda_seq = c(0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 3, 5, 10), k = 5
        )
        # 将最优参数 lambda 代入模型，得到估计的 beta
        beta_hat_with_penalty <- ncvfit(x, y,
            penalty = "lasso", lambda = best_param_for_lasso
        )$beta
    } else if (penalty == "ridge") {
        # 先找到最优参数 lambda
        best_param_for_ridge <- get_best_param(x, y,
            penalty = "ridge",
            lambda_seq = c(0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 3, 5, 10), k = 5
        )
        # 将最优参数 lambda 代入模型，得到估计的 beta
        beta_hat_with_penalty <- glmnet(x, y,
            alpha = 0,
            lambda = best_param_for_ridge,
            standardize = FALSE,
            intercept = FALSE,
        )$beta
    } else if (penalty == "garrote") {
        # 先找到最优参数 lambda
        nn_garrote <- cv.nnGarrote(x, y, verbose = FALSE)
        best_param_for_garrote <- nn_garrote$optimal.lambda.nng
        # 将最优参数 lambda 代入模型，得到估计的 beta
        beta_hat_with_penalty <- nnGarrote(x, y,
            lambda.nng = best_param_for_garrote
        )$betas[2:9]
    }
    # 计算 model error
    me_with_penalty <- cal_model_error(beta, beta_hat_with_penalty, x)
    # 计算 rme, correct zero, incorrect zero
    metrics <- cal_metrics(
        me_ols, me_with_penalty, beta, beta_hat_with_penalty
    )
    rme <- metrics$rme
    correct_zero <- metrics$correct_zero
    incorrect_zero <- metrics$incorrect_zero
    return(list(
        rme = rme,
        correct_zero = correct_zero,
        incorrect_zero = incorrect_zero
    ))
}

init_results <- function(iters) {
    # 创建一个空的 data table，用于存储结果
    results <- data.table(matrix(NA_real_, nrow = iters, ncol = 15))
    col_names <- c(
        "rme_scad_1",
        "rme_scad_2",
        "rme_lasso",
        "rme_ridge",
        "rme_garrote",
        "correct_zero_scad_1",
        "correct_zero_scad_2",
        "correct_zero_lasso",
        "correct_zero_ridge",
        "correct_zero_garrote",
        "incorrect_zero_scad_1",
        "incorrect_zero_scad_2",
        "incorrect_zero_lasso",
        "incorrect_zero_ridge",
        "incorrect_zero_garrote"
    )
    setnames(results, col_names)
    return(results)
}

get_results <- function(n, sigma, iters = 100) {
    # 初始化 results
    results <- init_results(iters)
    pb <- progress_bar$new(total = iters, clear = FALSE)
    for (iter in 1:iters) {
        # 显示进度
        pb$tick()
        # 生成数据
        simulated_data <- generate_data(n, rho, beta, sigma)
        x <- simulated_data$x
        y <- simulated_data$y
        # ==========OLS==========
        beta_hat_ols <- lm(y ~ x - 1)$coefficients
        # 计算 ols model error
        me_ols <- cal_model_error(beta, beta_hat_ols, x)
        # ==========其他模型==========
        for (penalty in c("scad_1", "scad_2", "lasso", "ridge", "garrote")) {
            # 获取结果，包括 rme, correct_zero, incorrect_zero
            result <- get_result(x, y, penalty, me_ols)
            names(result) <- c(
                paste0("rme_", penalty),
                paste0("correct_zero_", penalty),
                paste0("incorrect_zero_", penalty)
            )
            # 将结果添加到 results 中
            results[iter, names(result) := result]
        }
    }
    return(results)
}

init_all_results <- function() {
    # 创建一个空的 data table，用于存储所有 n 和 sigma 下的结果
    all_results <- data.table(
        n = integer(15),
        sigma = integer(15),
        penalty = character(15),
        MRME = numeric(15),
        Avg_No_of_correct_zero = numeric(15),
        Avg_No_of_incorrect_zero = numeric(15)
    )
    return(all_results)
}

# 生成模拟结果
all_results <- init_all_results()
for (i_n_and_sigma in list(c(1, 40, 3), c(6, 40, 1), c(11, 60, 1))) {
    i <- i_n_and_sigma[1]
    n_value <- i_n_and_sigma[2]
    sigma_value <- i_n_and_sigma[3]
    results <- get_results(n_value, sigma_value, iters = 100)
    # 计算 rme 的中位数，以及 correct zero 和 incorrect zero 的平均值。即前 5 列求中位数，后 10 列求平均值
    rme_median <- results[, lapply(.SD, median), .SDcols = 1:5]
    # 将 rme 的中位数保留两位小数
    set(rme_median,
        j = names(rme_median),
        value = round(rme_median * 100, digits = 2)
    )
    correct_zero_mean <- results[, lapply(.SD, mean), .SDcols = 6:10]
    incorrect_zero_mean <- results[, lapply(.SD, mean), .SDcols = 11:15]
    # 将结果添加到 all_results 中
    all_results[i:(i + 4), ] <- list(
        n = n_value,
        sigma = sigma_value,
        penalty = c("SCAD_1", "SCAD_2", "LASSO", "RIDGE", "GARROTE"),
        MRME = unlist(rme_median),
        Avg_No_of_correct_zero = unlist(correct_zero_mean),
        Avg_No_of_incorrect_zero = unlist(incorrect_zero_mean)
    )
}

# 将 all_results 导出为 csv 文件
fwrite(all_results, "all_results.csv")
