# Tuning, calibration and cost

## Successive halving (72 fits in 4s)

|   iter |   candidates |   samples_each |   best_log_loss |
|-------:|-------------:|---------------:|----------------:|
|      0 |           48 |           2000 |          0.4901 |
|      1 |           16 |           6000 |          0.4833 |
|      2 |            6 |          18000 |          0.4827 |
|      3 |            2 |          54000 |          0.4829 |

Best parameters: `{'l2_regularization': np.float64(0.0046), 'learning_rate': np.float64(0.0425), 'max_iter': 330, 'max_leaf_nodes': 6, 'min_samples_leaf': 13}`

## Models compared on new orders

| model                        |    AUC |   Brier |   log_loss |   calibration_error |   mean_abs_error_vs_truth |    fit_s |   predict_us_per_order |
|:-----------------------------|-------:|--------:|-----------:|--------------------:|--------------------------:|---------:|-----------------------:|
| Logistic regression          | 0.8325 |  0.1642 |     0.4818 |              0.0071 |                    0.0107 |   0.1737 |                 0.5504 |
| Gradient boosting (tuned)    | 0.8326 |  0.1642 |     0.482  |              0.0094 |                    0.0129 |   0.5535 |                 1.8722 |
| Overfit boosting             | 0.8085 |  0.1791 |     0.5278 |              0.0561 |                    0.0862 |   3.9489 |                 1.9313 |
| Overfit boosting + isotonic  | 0.8169 |  0.172  |     0.5008 |              0.0237 |                    0.0667 |  13.0534 |                 5.8639 |
| True probabilities (ceiling) | 0.833  |  0.164  |     0.4812 |              0.0065 |                    0      | nan      |               nan      |
