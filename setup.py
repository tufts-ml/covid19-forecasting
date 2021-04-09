''' Script to build the Cython extensions
'''

from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules=cythonize('aced_hmm/simulate_traj__cython.pyx', build_dir="build"),
	script_args=['build_ext'],
	options={"build_ext": {"inplace": True}})