import tqdm
import os
import numpy as np
import time
import pandas as pd

from pu_finance.utils import load_data_one_year
from pu_finance.metrics import get_ranking_scores


from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score


### SETTINGS ###

# for reproducibility
random_state = 42
# use metadata?
use_metadata = False

###################

# path to save results
path_to_save_res = os.path.abspath("./results/lr_baseline.csv")
# path to load data
base_data_dir = os.path.abspath("./data")

if use_metadata:
    # We need to drop for simple impute to work?
    cat_features = [0, 1, 2]
    drop_nan = True
else:
    cat_features = []
    drop_nan = False

# iterate over splits
results = []
for folder in tqdm.tqdm(os.listdir(base_data_dir)):
    full_path = os.path.join(base_data_dir, folder)
    X_train, X_test, y_train, y_test = load_data_one_year(
        full_path, use_metadata=use_metadata, drop_nan=drop_nan
    )
    # declare classifier
    clf = Pipeline(
        [
            (
                "tr",
                make_column_transformer(
                    [OneHotEncoder(handle_unknown="ignore"), cat_features],
                    remainder="passthrough",
                ),
            ),
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    random_state=random_state, class_weight="balanced", max_iter=1000
                ),
            ),
        ]
    )

    # Fit + Generate predictions (timed)
    time_s = time.time()

    clf.fit(X_train, y_train)
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    time_e = time.time() - time_s

    # Get results
    metrics = get_ranking_scores(y_test, y_test_proba)
    cur_res = {
        "split": int(folder.split("_")[1]),
        "num_train": y_train.shape[0],
        "num_pos_train": y_train.sum(),
        "num_test": y_test.shape[0],
        "num_pos_test": y_test.sum(),
        "time": time_e,
    }
    cur_res.update(metrics)
    results.append(cur_res)
    # break

# Save and Print results
df_res = pd.DataFrame(results, columns=cur_res.keys())
df_res.sort_values("split", inplace=True)
df_res.to_csv(
    path_to_save_res,
    index=False,
)

print(f"\nLR Average scores: \n{df_res[list(metrics.keys())].mean().to_string()}")
