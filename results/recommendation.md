# Supplier recommendation

20,000 new orders, 3 candidate suppliers each. Late penalty = 25% of order value.
Every choice is graded with the simulator's true probabilities.

| strategy              |   hit_at_1 |   ndcg |   true_late_rate |   regret_pct_of_value |   total_cost |
|:----------------------|-----------:|-------:|-----------------:|----------------------:|-------------:|
| Random supplier       |     0.3314 | 0.8097 |           0.5763 |                1.6497 |  6.42214e+06 |
| Cheapest quote        |     0.2574 | 0.7548 |           0.6964 |                2.6341 |  6.4752e+06  |
| Lowest past late rate |     0.528  | 0.8745 |           0.4531 |                0.9255 |  6.38311e+06 |
| Model expected cost   |     0.9028 | 0.9856 |           0.4857 |                0.0318 |  6.33494e+06 |

## Regret (% of order value) for different costs of a delay

|   late_penalty |   Random supplier |   Cheapest quote |   Lowest past late rate |   Model expected cost |
|---------------:|------------------:|-----------------:|------------------------:|----------------------:|
|           0    |             1.958 |            0     |                   4.277 |                 0     |
|           0.05 |             1.371 |            0.001 |                   3.081 |                 0.001 |
|           0.1  |             0.945 |            0.163 |                   2.046 |                 0.003 |
|           0.15 |             0.921 |            0.728 |                   1.414 |                 0.009 |
|           0.25 |             1.65  |            2.634 |                   0.926 |                 0.032 |
|           0.4  |             3.257 |            6.007 |                   0.707 |                 0.01  |
|           0.6  |             5.648 |           10.752 |                   0.664 |                 0.01  |
