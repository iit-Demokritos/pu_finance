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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

from imblearn.ensemble import RUSBoostClassifier
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
path_to_save_res = os.path.abspath("./results/baselines_avg_meta.csv")
# path to load data
base_data_dir = os.path.abspath("./data")

cat_features = list(range(len(metadata_to_use)))

models = {
    "LR": Pipeline(
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
    ),
    "CatBoost": CatBoostClassifier(
        # these are CIK, SIC, State of Inc
        cat_features=cat_features,
        early_stopping_rounds=100,
        scale_pos_weight=2,
        thread_count=10,
        # no verbose and logs
        allow_writing_files=False,
        verbose=False,
        # reproducibility
        random_seed=random_state,
    ),
    # "RusBoost": Pipeline(
    #     [
    #         (
    #             "tr",
    #             make_column_transformer(
    #                 [OneHotEncoder(handle_unknown="ignore"), cat_features],
    #                 remainder="passthrough",
    #             ),
    #         ),
    #         ("imp", SimpleImputer(strategy="mean")),
    #         ("sc", StandardScaler(with_mean=False)),
    #         (
    #             "clf",
    #             RUSBoostClassifier(random_state=random_state, algorithm="SAMME"),
    #         ),
    #     ]
    # ),
}

results_all = []
for model_name, clf in models.items():
    results_per_model = []
    print(f"Running for {model_name} ...")
    time_s = time.time()
    for folder in tqdm.tqdm(os.listdir(base_data_dir)):
        full_path = os.path.join(base_data_dir, folder)
        X_train, X_test, y_train, y_test = load_data_one_year(
            full_path, metadata_to_use=metadata_to_use, drop_nan=drop_nan
        )
        # Fit + Generate predictions (timed)

        if model_name == "CatBoost":
            # use this for early stopping
            cur_X_train, cur_X_val, cur_y_train, cur_y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.1,
                stratify=y_train,
                random_state=random_state,
            )
            clf.fit(
                cur_X_train, cur_y_train, eval_set=(cur_X_val, cur_y_val), verbose=0
            )
        else:
            clf.fit(X_train, y_train)
        y_test_proba = clf.predict_proba(X_test)[:, 1]

        metrics = get_ranking_scores(y_test, y_test_proba)
        cur_res = {"split": int(folder.split("_")[1])}
        cur_res.update(metrics)
        results_per_model.append(cur_res)
    df_res = pd.DataFrame(results_per_model, columns=cur_res.keys())

    print(
        f"\n{model_name} average scores: \n{df_res[metrics.keys()].mean().to_string()}\n"
    )

    time_e = time.time() - time_s
    results_all.append(
        {"model": model_name, "time": time_e, **df_res[metrics.keys()].mean().to_dict()}
    )


# Save and Print results
df_res = pd.DataFrame(results_all, columns=["model", "time"] + list(metrics.keys()))
df_res.sort_values("average_precision", inplace=True)
df_res.to_csv(
    path_to_save_res,
    index=False,
)
print(f"\nALL Average scores: \n{df_res.to_string()}")
