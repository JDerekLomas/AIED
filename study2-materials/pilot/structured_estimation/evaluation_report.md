# Structured IRT Estimation: Evaluation Report

Generated: 2026-02-01 10:55


## IRT Difficulty (b) (`b_2pl`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| RF | C_ratings_plus_direct | 0.242 | 0.053 | 0.535 | 0.486 |
| RF | D_all_28 | 0.239 | 0.048 | 0.510 | 0.483 |
| ElasticNet | C_ratings_plus_direct | 0.205 | 0.230 | 0.471 | 0.381 |
| ElasticNet | A_ratings_mean | 0.205 | 0.230 | 0.471 | 0.381 |
| RF | A_ratings_mean | 0.181 | 0.002 | 0.532 | 0.478 |
| RF | B_ratings_mean_std | 0.148 | 0.021 | 0.552 | 0.483 |
| Direct_LLM | direct_prediction | 0.138 | 0.089 | 0.520 | 0.434 |
| Ridge | A_ratings_mean | 0.137 | 0.188 | 0.484 | 0.412 |
| Ridge | C_ratings_plus_direct | 0.126 | 0.170 | 0.494 | 0.411 |
| Lasso | C_ratings_plus_direct | 0.117 | 0.108 | 0.467 | 0.377 |
| Lasso | A_ratings_mean | 0.117 | 0.108 | 0.467 | 0.377 |
| Ridge | D_all_28 | 0.075 | 0.217 | 0.614 | 0.458 |
| Lasso | B_ratings_mean_std | 0.025 | 0.057 | 0.472 | 0.381 |
| Lasso | D_all_28 | 0.025 | 0.057 | 0.472 | 0.381 |
| Ridge | B_ratings_mean_std | -0.003 | 0.123 | 0.623 | 0.515 |
| ElasticNet | B_ratings_mean_std | -0.013 | 0.108 | 0.514 | 0.419 |
| ElasticNet | D_all_28 | -0.013 | 0.108 | 0.514 | 0.419 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| RF | A_ratings_mean | -0.021 | 0.014 | 0.976 | 0.662 |
| Direct_LLM | direct_prediction | -0.037 | -0.052 | 0.959 | 0.656 |
| RF | C_ratings_plus_direct | -0.046 | -0.081 | 0.976 | 0.676 |
| RF | B_ratings_mean_std | -0.054 | 0.008 | 0.997 | 0.673 |
| RF | D_all_28 | -0.072 | -0.079 | 0.982 | 0.671 |
| Ridge | C_ratings_plus_direct | -0.074 | -0.127 | 0.956 | 0.674 |
| Ridge | A_ratings_mean | -0.081 | -0.118 | 0.939 | 0.648 |
| Ridge | D_all_28 | -0.128 | -0.155 | 1.066 | 0.750 |
| Ridge | B_ratings_mean_std | -0.145 | -0.183 | 1.038 | 0.731 |
| ElasticNet | A_ratings_mean | -0.157 | -0.124 | 0.892 | 0.582 |
| ElasticNet | C_ratings_plus_direct | -0.160 | -0.130 | 0.892 | 0.582 |
| ElasticNet | B_ratings_mean_std | -0.250 | -0.240 | 0.911 | 0.600 |
| ElasticNet | D_all_28 | -0.255 | -0.241 | 0.913 | 0.602 |
| Lasso | C_ratings_plus_direct | -0.484 | -0.515 | 0.890 | 0.580 |
| Lasso | A_ratings_mean | -0.484 | -0.515 | 0.890 | 0.580 |
| Lasso | B_ratings_mean_std | -0.493 | -0.530 | 0.890 | 0.581 |
| Lasso | D_all_28 | -0.493 | -0.530 | 0.890 | 0.581 |

**Best hold-out**: RF + C_ratings_plus_direct (r=0.242)
**vs Direct LLM**: Δr = +0.104

## IRT Discrimination (a) (`a_2pl`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| RF | B_ratings_mean_std | 0.359 | 0.460 | 0.369 | 0.263 |
| RF | D_all_28 | 0.343 | 0.481 | 0.361 | 0.256 |
| RF | C_ratings_plus_direct | 0.224 | 0.418 | 0.369 | 0.268 |
| RF | A_ratings_mean | 0.197 | 0.406 | 0.385 | 0.284 |
| Ridge | D_all_28 | 0.118 | 0.159 | 0.692 | 0.552 |
| Ridge | B_ratings_mean_std | 0.110 | 0.299 | 0.502 | 0.387 |
| Ridge | C_ratings_plus_direct | 0.081 | 0.099 | 0.559 | 0.470 |
| Ridge | A_ratings_mean | 0.024 | -0.048 | 0.496 | 0.373 |
| ElasticNet | A_ratings_mean | -0.058 | -0.152 | 0.405 | 0.324 |
| ElasticNet | B_ratings_mean_std | -0.074 | -0.035 | 0.406 | 0.327 |
| ElasticNet | C_ratings_plus_direct | -0.127 | -0.235 | 0.411 | 0.340 |
| ElasticNet | D_all_28 | -0.132 | -0.099 | 0.416 | 0.342 |
| Lasso | A_ratings_mean | -0.226 | -0.133 | 0.408 | 0.334 |
| Lasso | B_ratings_mean_std | -0.226 | -0.133 | 0.408 | 0.334 |
| Lasso | C_ratings_plus_direct | -0.226 | -0.133 | 0.408 | 0.334 |
| Lasso | D_all_28 | -0.226 | -0.133 | 0.408 | 0.334 |
| Direct_LLM | direct_prediction | -0.340 | -0.483 | 0.407 | 0.322 |

### LOO-CV (all items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | C_ratings_plus_direct | 0.010 | -0.066 | 1.051 | 0.569 |
| Ridge | D_all_28 | -0.013 | -0.065 | 1.108 | 0.627 |
| Direct_LLM | direct_prediction | -0.030 | -0.207 | 0.961 | 0.417 |
| RF | C_ratings_plus_direct | -0.043 | -0.109 | 0.973 | 0.434 |
| RF | A_ratings_mean | -0.058 | -0.112 | 0.973 | 0.424 |
| Ridge | B_ratings_mean_std | -0.064 | -0.062 | 1.095 | 0.589 |
| RF | D_all_28 | -0.075 | -0.007 | 0.978 | 0.421 |
| Ridge | A_ratings_mean | -0.086 | -0.151 | 1.048 | 0.528 |
| RF | B_ratings_mean_std | -0.104 | 0.020 | 0.980 | 0.414 |
| ElasticNet | A_ratings_mean | -0.138 | -0.266 | 0.986 | 0.460 |
| ElasticNet | C_ratings_plus_direct | -0.149 | -0.298 | 0.989 | 0.464 |
| ElasticNet | B_ratings_mean_std | -0.168 | -0.266 | 0.994 | 0.464 |
| ElasticNet | D_all_28 | -0.173 | -0.279 | 0.996 | 0.466 |
| Lasso | B_ratings_mean_std | -0.186 | -0.234 | 0.970 | 0.429 |
| Lasso | C_ratings_plus_direct | -0.186 | -0.234 | 0.970 | 0.429 |
| Lasso | A_ratings_mean | -0.186 | -0.234 | 0.970 | 0.429 |
| Lasso | D_all_28 | -0.186 | -0.234 | 0.970 | 0.429 |

**Best hold-out**: RF + B_ratings_mean_std (r=0.359)
**vs Direct LLM**: Δr = +0.699

## Guessing (c) (`c_3pl_guess`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | B_ratings_mean_std | 0.104 | 0.041 | 0.194 | 0.169 |
| Ridge | D_all_28 | 0.080 | 0.076 | 0.213 | 0.187 |
| Ridge | C_ratings_plus_direct | 0.052 | 0.038 | 0.175 | 0.147 |
| Ridge | A_ratings_mean | 0.043 | -0.090 | 0.160 | 0.133 |
| RF | C_ratings_plus_direct | -0.001 | 0.150 | 0.190 | 0.153 |
| Direct_LLM | direct_prediction | -0.087 | 0.028 | 0.149 | 0.118 |
| RF | D_all_28 | -0.116 | -0.005 | 0.196 | 0.162 |
| RF | A_ratings_mean | -0.142 | 0.006 | 0.201 | 0.162 |
| RF | B_ratings_mean_std | -0.243 | -0.136 | 0.200 | 0.168 |
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
| Ridge | C_ratings_plus_direct | 0.279 | 0.277 | 0.169 | 0.136 |
| Ridge | D_all_28 | 0.187 | 0.185 | 0.183 | 0.149 |
| Ridge | A_ratings_mean | 0.176 | 0.133 | 0.175 | 0.144 |
| Ridge | B_ratings_mean_std | 0.131 | 0.096 | 0.187 | 0.155 |
| RF | A_ratings_mean | 0.084 | 0.101 | 0.176 | 0.143 |
| Direct_LLM | direct_prediction | 0.075 | 0.073 | 0.187 | 0.144 |
| RF | C_ratings_plus_direct | 0.068 | 0.148 | 0.177 | 0.145 |
| RF | B_ratings_mean_std | 0.055 | 0.060 | 0.176 | 0.146 |
| RF | D_all_28 | 0.018 | 0.075 | 0.178 | 0.148 |
| ElasticNet | B_ratings_mean_std | -1.000 | -0.983 | 0.173 | 0.145 |
| Lasso | B_ratings_mean_std | -1.000 | -0.983 | 0.173 | 0.145 |
| Lasso | C_ratings_plus_direct | -1.000 | -0.983 | 0.173 | 0.145 |
| ElasticNet | C_ratings_plus_direct | -1.000 | -0.983 | 0.173 | 0.145 |
| ElasticNet | A_ratings_mean | -1.000 | -0.983 | 0.173 | 0.145 |
| Lasso | A_ratings_mean | -1.000 | -0.983 | 0.173 | 0.145 |
| Lasso | D_all_28 | -1.000 | -0.983 | 0.173 | 0.145 |
| ElasticNet | D_all_28 | -1.000 | -0.983 | 0.173 | 0.145 |

**Best hold-out**: Ridge + B_ratings_mean_std (r=0.104)
**vs Direct LLM**: Δr = +0.190

## Classical Difficulty (`difficulty_classical`)

### Hold-out (20 items)

| Model | Features | Pearson r | Spearman ρ | RMSE | MAE |
|-------|----------|-----------|------------|------|-----|
| Ridge | C_ratings_plus_direct | 0.265 | 0.132 | 0.108 | 0.081 |
| Direct_LLM | direct_prediction | 0.260 | 0.125 | 0.167 | 0.131 |
| RF | B_ratings_mean_std | 0.195 | 0.189 | 0.110 | 0.085 |
| Ridge | D_all_28 | 0.168 | 0.158 | 0.128 | 0.097 |
| RF | A_ratings_mean | 0.160 | 0.176 | 0.111 | 0.088 |
| RF | C_ratings_plus_direct | 0.078 | 0.021 | 0.113 | 0.092 |
| RF | D_all_28 | 0.067 | 0.114 | 0.114 | 0.088 |
| Ridge | A_ratings_mean | 0.043 | -0.156 | 0.113 | 0.087 |
| Ridge | B_ratings_mean_std | 0.014 | 0.035 | 0.137 | 0.100 |
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
| Direct_LLM | direct_prediction | -0.005 | -0.008 | 0.199 | 0.162 |
| RF | B_ratings_mean_std | -0.131 | -0.136 | 0.155 | 0.126 |
| Ridge | A_ratings_mean | -0.156 | -0.134 | 0.158 | 0.128 |
| RF | A_ratings_mean | -0.167 | -0.155 | 0.156 | 0.127 |
| RF | D_all_28 | -0.189 | -0.199 | 0.158 | 0.128 |
| Ridge | C_ratings_plus_direct | -0.193 | -0.163 | 0.163 | 0.129 |
| Ridge | B_ratings_mean_std | -0.233 | -0.260 | 0.178 | 0.146 |
| RF | C_ratings_plus_direct | -0.246 | -0.278 | 0.158 | 0.129 |
| Ridge | D_all_28 | -0.267 | -0.273 | 0.188 | 0.150 |
| Lasso | A_ratings_mean | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | A_ratings_mean | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | B_ratings_mean_std | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | B_ratings_mean_std | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | C_ratings_plus_direct | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | C_ratings_plus_direct | -1.000 | -1.000 | 0.146 | 0.116 |
| Lasso | D_all_28 | -1.000 | -1.000 | 0.146 | 0.116 |
| ElasticNet | D_all_28 | -1.000 | -1.000 | 0.146 | 0.116 |

**Best hold-out**: Ridge + C_ratings_plus_direct (r=0.265)
**vs Direct LLM**: Δr = +0.006