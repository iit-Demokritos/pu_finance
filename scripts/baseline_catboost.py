import tqdm
import os
import time
import pandas as pd

from pu_finance.utils import load_data_one_year, METADATA_TO_USE
from pu_finance.metrics import get_ranking_scores

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


### SETTINGS ###

# for reproducibility
random_state = 42
# which metadata to use
metadata_to_use = []  # ["CIK", "SIC", "State of Inc"]

###################


# path to save results
path_to_save_res = os.path.abspath("./results/catboost.csv")
# path to load data
base_data_dir = os.path.abspath("./data")


cat_features = list(range(len(metadata_to_use)))

# iterate over splits
results = []
for folder in tqdm.tqdm(os.listdir(base_data_dir)):
    full_path = os.path.join(base_data_dir, folder)
    X_train, X_test, y_train, y_test = load_data_one_year(
        full_path,
        metadata_to_use=metadata_to_use,
    )

    # use this for early stopping
    cur_X_train, cur_X_val, cur_y_train, cur_y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=random_state
    )

    # declare classifier
    tr = Pipeline(
        [
            (
                "tr",
                make_column_transformer(
                    [OneHotEncoder(handle_unknown="ignore"), cat_features],
                    remainder="passthrough",
                ),
            ),
            ("imp", SimpleImputer(strategy="mean")),
        ]
    )
    cur_X_train = tr.fit_transform(cur_X_train)
    cur_X_val = tr.transform(cur_X_val)
    X_test = tr.transform(X_test)

    # # Use scale_pos_weight = None, as this attribute is only for Logloss
    # loss = "Focal:focal_alpha=0.75;focal_gamma=1"
    # scale_pos_weight = None

    # # This is the Default
    loss = "Logloss"
    scale_pos_weight = 2

    clf = CatBoostClassifier(
        # these are CIK, SIC, State of Inc
        # cat_features=cat_features,
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
        task_type="CPU",
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
