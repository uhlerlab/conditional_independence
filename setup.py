import setuptools

setuptools.setup(
    name='conditional_independence',
    version='0.1a.003',
    description='Parametric and non-parametric conditional independence tests.',
    long_description='',
    author='Chandler Squires',
    author_email='chandlersquires18@gmail.com',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>3.5.0',
    zip_safe=False,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'scipy',
        'dataclasses',
        'numpy',
        # 'scikit_sparse',
        'numexpr',
        'scikit_learn',
        'typing',
        'pygam',
        'tqdm',
        # 'numba',
        'ipdb',
    ]
)

