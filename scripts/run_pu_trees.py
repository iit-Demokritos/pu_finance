import os
import time

from sklearn.base import TransformerMixin

from pu_tree_simplified._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified._pu_classes import DecisionTreeClassifier
import numpy as np
import pandas as pd
import tqdm
from pu_finance.utils import load_data_one_year
from pu_finance.metrics import get_ranking_scores
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

random_state = 42


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# which metadata to use
metadata_to_use = ["SIC", "State of Inc"]
# drop rows with any NaN in train+test?
# If we use meta-data have this as true for the time being
# because many companies have no State for example
drop_nan = True

# path to save results
path_to_save_res = os.path.abspath("../results/PUHRF_baseline_meta_noCIK.csv")
# path to load data
base_data_dir = os.path.abspath("../data")

cat_features = list(range(len(metadata_to_use)))

# iterate over splits
results = []
for folder in tqdm.tqdm(os.listdir(base_data_dir)):
    full_path = os.path.join(base_data_dir, folder)
    X_train, X_test, y_train, y_test = load_data_one_year(
        full_path, metadata_to_use=metadata_to_use, drop_nan=drop_nan
    )

    prior_y = np.mean(y_train)

    # PU-HDT
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
            ('to_dense', DenseTransformer()),
            (
                "clf",
                #DecisionTreeClassifier(random_state=random_state, max_depth=5),
                PURF_SIMP(random_state=random_state, max_samples=np.count_nonzero(y_train == 0),
                          pu_biased_bootstrap=True)
            ),
        ]
    )
    time_s = time.time()
    clf.fit(X_train, y_train, clf__p_y=prior_y)
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

# Save and Print results
df_res = pd.DataFrame(results, columns=cur_res.keys())
df_res.sort_values("split", inplace=True)
df_res.to_csv(
    path_to_save_res,
    index=False,
)

print(f"\nPU-HDT Average scores: \n{df_res[list(metrics.keys())].mean().to_string()}")
