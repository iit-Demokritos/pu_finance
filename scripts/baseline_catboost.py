import tqdm
import os
import numpy as np
import time
import pandas as pd

from pu_finance.utils import load_data_one_year, METADATA_TO_USE
from pu_finance.metrics import get_ranking_scores

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier


### SETTINGS ###

# for reproducibility
random_state = 42
# which metadata to use
metadata_to_use = ["CIK", "SIC", "State of Inc"]
# drop rows with any NaN in train+test?
# If we use meta-data have this as true for the time being
# because many companies have no State for example
drop_nan = True
###################


# path to save results
path_to_save_res = os.path.abspath("./results/catboost_baseline_meta_noCIK.csv")
# path to load data
base_data_dir = os.path.abspath("./data")


cat_features = list(range(len(metadata_to_use)))

# iterate over splits
results = []
for folder in tqdm.tqdm(os.listdir(base_data_dir)):
    full_path = os.path.join(base_data_dir, folder)
    X_train, X_test, y_train, y_test = load_data_one_year(
        full_path, metadata_to_use=metadata_to_use, drop_nan=drop_nan
    )

    # use this for early stopping
    cur_X_train, cur_X_val, cur_y_train, cur_y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=random_state
    )

    # You can change this @izavits to use focal loss
    # Use scale_pos_weight = None, as this attribute is only for Logloss
    # loss = "Focal:focal_alpha=0.25;focal_gamma=3"
    # scale_pos_weight = None

    # This is the Default
    loss = "Logloss"
    scale_pos_weight = 2

    clf = CatBoostClassifier(
        # these are CIK, SIC, State of Inc
        cat_features=cat_features,
        # the loss function to use. Default is 'Logloss'.
        loss_function=loss,
        scale_pos_weight=scale_pos_weight,
        # for early stopping
        early_stopping_rounds=100,
        thread_count=10,
        # no verbose and logs
        allow_writing_files=False,
        verbose=False,
        # reproducibility
        random_seed=random_state,
    )

    clf.fit(cur_X_train, cur_y_train, eval_set=(cur_X_val, cur_y_val), verbose=0)

    # Fit + Generate predictions (timed)
    time_s = time.time()

    # Optimal threshold
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    dt_discr = DecisionTreeClassifier(random_state=42, max_depth=1)
    dt_discr.fit(clf.predict_proba(cur_X_val)[:, 1].reshape(-1, 1), cur_y_val)
    opt_threshold = dt_discr.tree_.threshold[0]

    time_e = time.time() - time_s

    # Get results
    metrics = get_ranking_scores(y_test, y_test_proba, threshold=opt_threshold)
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

print(f"\nCatBoost Average scores: \n{df_res[list(metrics.keys())].mean().to_string()}")
