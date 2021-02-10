# Build Cython extensions to run fast

from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules=cythonize('semimarkov_forecaster/simulate_traj__cython.pyx', build_dir="build"),
	script_args=['build_ext'],
	#prefix="./",
	#install_base="./",
	options={"build_ext": {"inplace": True}})
