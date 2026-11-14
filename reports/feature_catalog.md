# Feature Catalog

| Feature Name | Entity | Data Type | Description | Source | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_avg_rating` | User | `float64` | Mean explicit rating given by the user across historical interactions. | train_interactions | Training & Inference |
| `user_activity_count` | User | `int64` | Total number of historical interactions recorded for the user. | train_interactions | Training & Inference |
| `user_preferred_genre` | User | `string` | Most frequently interacted primary genre for the user. | train_interactions + clean_items | Training & Inference |
| `days_since_last_interaction` | User | `float64` | Days elapsed between the user's most recent interaction and dataset cutoff. | train_interactions | Training & Inference |
| `item_popularity_count` | Item | `int64` | Number of historical ratings received by the item. | item_popularity | Training & Inference |
| `item_avg_rating` | Item | `float64` | Raw average explicit rating of the item. | item_popularity | Training & Inference |
| `item_bayesian_rating` | Item | `float64` | Smoothed rating combining item average and global mean with prior weight m=10. | item_popularity | Training & Inference |
| `primary_genre` | Item | `string` | Primary genre category of the item. | clean_items | Training & Inference |
| `genre_affinity_score` | User-Item Interaction | `float64` | Share of the user's historical interactions matching the candidate item's primary genre. | Derived cross-feature | Training & Inference |