from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

#setup(
#    ext_modules=cythonize("pu_tree_simplified/*.pyx"),
#    include_dirs=[numpy.get_include()]
#)




extensions = [
    Extension("pu_finance._pu_criterion", ["pu_finance/_pu_criterion.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension("pu_finance._pu_splitter", ["pu_finance/_pu_splitter.pyx"],
include_dirs=[numpy.get_include()]),
    Extension("pu_finance._pu_tree", ["pu_finance/_pu_tree.pyx"],
              include_dirs=[numpy.get_include()]),
Extension("pu_finance._utils", ["pu_finance/_utils.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="pu_finance",
    ext_modules=cythonize(extensions),
)