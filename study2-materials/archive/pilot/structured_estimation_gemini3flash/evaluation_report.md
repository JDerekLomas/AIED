# Structured IRT Estimation: Evaluation Report

Generated: 2026-01-30 12:56


## IRT Difficulty (b) (`b_2pl`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| RF | B_ratings_mean_std | 0.350 | 0.305 | 0.624 | 0.526 |
| RF | D_all_28 | 0.310 | 0.165 | 0.593 | 0.504 |
| ElasticNet | A_ratings_mean | 0.305 | 0.369 | 0.492 | 0.402 |
| ElasticNet | C_ratings_plus_direct | 0.288 | 0.370 | 0.490 | 0.398 |
| Direct_LLM | direct_prediction | 0.273 | 0.008 | 0.490 | 0.434 |
| Lasso | A_ratings_mean | 0.266 | 0.280 | 0.478 | 0.385 |
| Lasso | C_ratings_plus_direct | 0.260 | 0.286 | 0.479 | 0.386 |
| Ridge | A_ratings_mean | 0.259 | 0.356 | 0.536 | 0.429 |
| RF | A_ratings_mean | 0.237 | 0.113 | 0.636 | 0.552 |
| ElasticNet | B_ratings_mean_std | 0.231 | 0.340 | 0.552 | 0.413 |
| ElasticNet | D_all_28 | 0.229 | 0.313 | 0.553 | 0.415 |
| Lasso | B_ratings_mean_std | 0.211 | 0.232 | 0.500 | 0.396 |
| Lasso | D_all_28 | 0.211 | 0.232 | 0.500 | 0.396 |
| RF | C_ratings_plus_direct | 0.206 | 0.083 | 0.638 | 0.549 |
| Ridge | B_ratings_mean_std | 0.122 | 0.223 | 0.707 | 0.508 |
| Ridge | C_ratings_plus_direct | 0.074 | 0.162 | 0.593 | 0.479 |
| Ridge | D_all_28 | 0.061 | 0.111 | 0.801 | 0.619 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | D_all_28 | 0.021 | 0.015 | 1.024 | 0.728 |
| Ridge | B_ratings_mean_std | -0.001 | -0.007 | 1.001 | 0.697 |
| ElasticNet | A_ratings_mean | -0.015 | 0.044 | 0.884 | 0.579 |
| RF | C_ratings_plus_direct | -0.028 | 0.050 | 0.973 | 0.644 |
| Direct_LLM | direct_prediction | -0.036 | -0.123 | 0.974 | 0.683 |
| RF | A_ratings_mean | -0.038 | -0.019 | 0.973 | 0.647 |
| ElasticNet | B_ratings_mean_std | -0.041 | 0.003 | 0.908 | 0.594 |
| RF | D_all_28 | -0.047 | -0.107 | 0.951 | 0.647 |
| Ridge | A_ratings_mean | -0.051 | -0.013 | 0.929 | 0.636 |
| ElasticNet | D_all_28 | -0.055 | -0.018 | 0.911 | 0.597 |
| Ridge | C_ratings_plus_direct | -0.061 | -0.003 | 0.957 | 0.659 |
| ElasticNet | C_ratings_plus_direct | -0.065 | 0.006 | 0.892 | 0.588 |
| RF | B_ratings_mean_std | -0.072 | -0.119 | 0.966 | 0.655 |
| Lasso | B_ratings_mean_std | -0.092 | -0.004 | 0.889 | 0.569 |
| Lasso | D_all_28 | -0.099 | -0.005 | 0.890 | 0.570 |
| Lasso | A_ratings_mean | -0.112 | -0.056 | 0.883 | 0.571 |
| Lasso | C_ratings_plus_direct | -0.125 | -0.061 | 0.885 | 0.571 |

**Best hold-out**: RF + B_ratings_mean_std (r=0.350)
**vs Direct LLM**: Δr = +0.077

## IRT Discrimination (a) (`a_2pl`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Lasso | D_all_28 | 0.471 | 0.463 | 0.319 | 0.255 |
| ElasticNet | D_all_28 | 0.398 | 0.376 | 0.383 | 0.295 |
| ElasticNet | B_ratings_mean_std | 0.380 | 0.353 | 0.387 | 0.303 |
| RF | D_all_28 | 0.371 | 0.365 | 0.347 | 0.278 |
| Lasso | B_ratings_mean_std | 0.346 | 0.368 | 0.342 | 0.267 |
| Ridge | B_ratings_mean_std | 0.330 | 0.405 | 0.787 | 0.635 |
| RF | A_ratings_mean | 0.288 | 0.202 | 0.370 | 0.295 |
| Ridge | D_all_28 | 0.282 | 0.361 | 0.839 | 0.656 |
| RF | B_ratings_mean_std | 0.277 | 0.153 | 0.367 | 0.279 |
| RF | C_ratings_plus_direct | 0.266 | 0.395 | 0.362 | 0.294 |
| Ridge | C_ratings_plus_direct | 0.241 | 0.247 | 0.424 | 0.317 |
| Lasso | C_ratings_plus_direct | 0.231 | 0.250 | 0.349 | 0.263 |
| Ridge | A_ratings_mean | 0.192 | 0.217 | 0.458 | 0.324 |
| ElasticNet | C_ratings_plus_direct | 0.139 | 0.206 | 0.364 | 0.270 |
| ElasticNet | A_ratings_mean | 0.084 | 0.128 | 0.373 | 0.286 |
| Lasso | A_ratings_mean | -0.161 | -0.156 | 0.369 | 0.280 |
| Direct_LLM | direct_prediction | -0.297 | -0.333 | 0.387 | 0.306 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | D_all_28 | -0.007 | 0.051 | 1.141 | 0.650 |
| Ridge | B_ratings_mean_std | -0.009 | 0.032 | 1.116 | 0.617 |
| Ridge | C_ratings_plus_direct | -0.010 | 0.046 | 1.024 | 0.494 |
| ElasticNet | B_ratings_mean_std | -0.012 | 0.024 | 0.994 | 0.464 |
| RF | B_ratings_mean_std | -0.022 | 0.036 | 1.030 | 0.476 |
| ElasticNet | D_all_28 | -0.025 | 0.003 | 1.002 | 0.473 |
| Ridge | A_ratings_mean | -0.026 | 0.030 | 1.013 | 0.469 |
| RF | D_all_28 | -0.039 | 0.015 | 1.043 | 0.494 |
| Lasso | D_all_28 | -0.046 | 0.038 | 0.971 | 0.421 |
| RF | A_ratings_mean | -0.052 | 0.108 | 1.063 | 0.482 |
| RF | C_ratings_plus_direct | -0.058 | 0.048 | 1.069 | 0.501 |
| Lasso | B_ratings_mean_std | -0.070 | -0.030 | 0.971 | 0.422 |
| Direct_LLM | direct_prediction | -0.076 | -0.159 | 0.959 | 0.416 |
| ElasticNet | C_ratings_plus_direct | -0.138 | -0.164 | 0.983 | 0.443 |
| Lasso | C_ratings_plus_direct | -0.151 | -0.045 | 0.965 | 0.417 |
| ElasticNet | A_ratings_mean | -0.162 | -0.202 | 0.978 | 0.436 |
| Lasso | A_ratings_mean | -0.666 | -0.674 | 0.964 | 0.416 |

**Best hold-out**: Lasso + D_all_28 (r=0.471)
**vs Direct LLM**: Δr = +0.767

## Guessing (c) (`c_3pl_guess`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | D_all_28 | 0.192 | 0.177 | 0.177 | 0.136 |
| Ridge | C_ratings_plus_direct | 0.151 | 0.229 | 0.171 | 0.135 |
| Ridge | A_ratings_mean | 0.148 | 0.085 | 0.152 | 0.121 |
| Ridge | B_ratings_mean_std | 0.120 | 0.183 | 0.168 | 0.139 |
| RF | A_ratings_mean | 0.056 | 0.136 | 0.162 | 0.135 |
| Direct_LLM | direct_prediction | 0.019 | -0.108 | 0.143 | 0.114 |
| RF | C_ratings_plus_direct | -0.043 | 0.008 | 0.169 | 0.137 |
| RF | B_ratings_mean_std | -0.091 | -0.125 | 0.174 | 0.140 |
| RF | D_all_28 | -0.105 | -0.077 | 0.168 | 0.134 |
| Lasso | A_ratings_mean | nan | nan | 0.147 | 0.125 |
| ElasticNet | A_ratings_mean | nan | nan | 0.147 | 0.125 |
| Lasso | B_ratings_mean_std | nan | nan | 0.147 | 0.125 |
| ElasticNet | B_ratings_mean_std | nan | nan | 0.147 | 0.125 |
| Lasso | C_ratings_plus_direct | nan | nan | 0.147 | 0.125 |
| ElasticNet | C_ratings_plus_direct | nan | nan | 0.147 | 0.125 |
| Lasso | D_all_28 | nan | nan | 0.147 | 0.125 |
| ElasticNet | D_all_28 | nan | nan | 0.147 | 0.125 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | C_ratings_plus_direct | 0.228 | 0.225 | 0.173 | 0.142 |
| Ridge | D_all_28 | 0.174 | 0.190 | 0.189 | 0.149 |
| RF | C_ratings_plus_direct | 0.143 | 0.150 | 0.172 | 0.142 |
| Ridge | A_ratings_mean | 0.142 | 0.079 | 0.175 | 0.144 |
| Ridge | B_ratings_mean_std | 0.142 | 0.154 | 0.188 | 0.150 |
| RF | D_all_28 | 0.082 | 0.107 | 0.174 | 0.143 |
| RF | A_ratings_mean | -0.059 | -0.014 | 0.183 | 0.149 |
| RF | B_ratings_mean_std | -0.109 | -0.068 | 0.183 | 0.150 |
| Direct_LLM | direct_prediction | -0.164 | -0.143 | 0.185 | 0.151 |
| ElasticNet | B_ratings_mean_std | -1.000 | -0.998 | 0.173 | 0.145 |
| Lasso | B_ratings_mean_std | -1.000 | -0.998 | 0.173 | 0.145 |
| Lasso | C_ratings_plus_direct | -1.000 | -0.998 | 0.173 | 0.145 |
| ElasticNet | C_ratings_plus_direct | -1.000 | -0.998 | 0.173 | 0.145 |
| ElasticNet | A_ratings_mean | -1.000 | -0.998 | 0.173 | 0.145 |
| Lasso | A_ratings_mean | -1.000 | -0.998 | 0.173 | 0.145 |
| Lasso | D_all_28 | -1.000 | -0.998 | 0.173 | 0.145 |
| ElasticNet | D_all_28 | -1.000 | -0.998 | 0.173 | 0.145 |

**Best hold-out**: Ridge + D_all_28 (r=0.192)
**vs Direct LLM**: Δr = +0.173

## Classical Difficulty (`difficulty_classical`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | B_ratings_mean_std | 0.169 | 0.313 | 0.114 | 0.080 |
| Ridge | A_ratings_mean | 0.126 | 0.192 | 0.113 | 0.083 |
| Ridge | D_all_28 | -0.004 | 0.132 | 0.142 | 0.108 |
| Ridge | C_ratings_plus_direct | -0.064 | 0.090 | 0.135 | 0.097 |
| RF | A_ratings_mean | -0.067 | -0.032 | 0.118 | 0.093 |
| RF | C_ratings_plus_direct | -0.118 | 0.033 | 0.123 | 0.096 |
| RF | D_all_28 | -0.125 | -0.183 | 0.117 | 0.090 |
| RF | B_ratings_mean_std | -0.128 | -0.161 | 0.117 | 0.090 |
| Direct_LLM | direct_prediction | -0.286 | -0.202 | 0.203 | 0.150 |
| Lasso | A_ratings_mean | nan | nan | 0.108 | 0.083 |
| ElasticNet | A_ratings_mean | nan | nan | 0.108 | 0.083 |
| Lasso | B_ratings_mean_std | nan | nan | 0.108 | 0.083 |
| ElasticNet | B_ratings_mean_std | nan | nan | 0.108 | 0.083 |
| Lasso | C_ratings_plus_direct | nan | nan | 0.108 | 0.083 |
| ElasticNet | C_ratings_plus_direct | nan | nan | 0.108 | 0.083 |
| Lasso | D_all_28 | nan | nan | 0.108 | 0.083 |
| ElasticNet | D_all_28 | nan | nan | 0.108 | 0.083 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Direct_LLM | direct_prediction | -0.104 | -0.052 | 0.215 | 0.170 |
| RF | C_ratings_plus_direct | -0.107 | -0.082 | 0.157 | 0.124 |
| RF | A_ratings_mean | -0.107 | -0.078 | 0.156 | 0.121 |
| RF | D_all_28 | -0.123 | -0.086 | 0.155 | 0.120 |
| RF | B_ratings_mean_std | -0.138 | -0.121 | 0.155 | 0.121 |
| Ridge | B_ratings_mean_std | -0.184 | -0.165 | 0.173 | 0.136 |
| Ridge | A_ratings_mean | -0.218 | -0.153 | 0.160 | 0.126 |
| Ridge | D_all_28 | -0.262 | -0.252 | 0.182 | 0.144 |
| Ridge | C_ratings_plus_direct | -0.304 | -0.236 | 0.167 | 0.132 |
| ElasticNet | A_ratings_mean | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | B_ratings_mean_std | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | B_ratings_mean_std | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | C_ratings_plus_direct | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | C_ratings_plus_direct | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | A_ratings_mean | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | D_all_28 | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | D_all_28 | -1.000 | -1.000 | 0.146 | 0.116 |

**Best hold-out**: Ridge + B_ratings_mean_std (r=0.169)
**vs Direct LLM**: Δr = +0.455