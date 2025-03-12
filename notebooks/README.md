# Notebooks for faster development

To utilize the structure add the current path to root:

```python
import sys
import os
sys.path.append(os.path.abspath("../"))
```

Also do not forget to add reload functions for faster developing:



```python
%load_ext autoreload
%autoreload 2
```