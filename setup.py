''' Script to compile this module & build optional Cython extensions for speed
'''

import os
from distutils.core import setup

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

def read_requirements_list():
    with open("requirements.txt", 'r') as f:
        lines = [line.strip() for line in f.readlines() if len(line) > 4]
    return lines

def make_cython_extension_modules():
    return cythonize(
        'aced_hmm/simulator/simulate_traj__cython.pyx',
        build_dir="build")

def make_list_of_subpackages():
    ''' Traverse subdirectories recursively and add to list
    '''
    package_list = []
    # Traverse root directory, and list directories as dirs and files as files
    for root, dirpath_list, fpath_list in os.walk('aced_hmm/'):
        subpkg_path = root.strip(os.path.sep).replace(os.path.sep, '.')
        for fpath in fpath_list:
            if fpath == "__init__.py":
                package_list.append(subpkg_path)
    return package_list

########################################################################
# Main function

requirements_list = read_requirements_list()

setup(
    name="aced_hmm",
    version="0.1.20210412",
    author="Gian Marco Visani and Michael C. Hughes",
    description=(
        "Code for a simulation model of patient trajectories in the hospital"),
    license="MIT",
    keywords=[
        "COVID-19",
        "Hidden Markov model",
        "Approximate Bayesian computation",
        ],
    packages=make_list_of_subpackages(),
    package_data = {
        # If any subpackage contains these files, include them:
        '': ['*.txt', '*.md', '*.json', '*.csv'],
    },
    include_package_data=True,
    long_description='',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License"],
    setup_requires=requirements_list,
    install_requires=requirements_list,
    zip_safe=False,
    ext_modules=make_cython_extension_modules(),
    options={"build_ext": {"inplace": True}},
)
