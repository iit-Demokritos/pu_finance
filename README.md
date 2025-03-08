# PU Finance

> PU learning methods for financial use cases

## Data
### Data points
Misstatement data have been split into 12 folds. Each split is a test year (e.g. split_2003).
For each test year the training instances are from 3 years back (e.g. 2000-2002).

### Labels
For each test year, the test instances are labeled without label noise (i.e., without flipping labels for unknown misstatements).
For the corresponding training instances, positives examples that are suposed to be uknown at the time of training have been flipped to negative.
Thus, for each training set, there are hidden positives into the negatives. The hidden positives are those having restatement date past the time of training.
