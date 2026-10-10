# Late-delivery classifiers

Trained on 80,000 past orders, scored on 20,000 new ones.

| model                        |    AUC |   Brier |   log_loss |    fit_s |   predict_us_per_order |
|:-----------------------------|-------:|--------:|-----------:|---------:|-----------------------:|
| Mode late rate (spreadsheet) | 0.7322 |  0.1908 |     0.5429 |   0.0043 |                 0.0257 |
| Logistic regression          | 0.8325 |  0.1642 |     0.4818 |   0.1698 |                 0.5009 |
| Gradient boosting (defaults) | 0.8313 |  0.1648 |     0.4835 |   0.7587 |                 1.0745 |
| True probabilities (ceiling) | 0.833  |  0.164  |     0.4812 | nan      |               nan      |

## Permutation importance (gradient boosting)

AUC lost when one input is shuffled. Large = the model relies on it.

|                  |   auc_drop |
|:-----------------|-----------:|
| shipping_mode    |     0.2372 |
| supplier_id      |     0.0849 |
| market           |     0.0163 |
| order_month      |     0.0077 |
| department       |     0.0003 |
| unit_price       |     0.0001 |
| payment_type     |    -0.0001 |
| customer_segment |    -0.0001 |
| order_dow        |    -0.0001 |
| discount_rate    |    -0.0002 |
| quantity         |    -0.0002 |
