from distutils.core import setup

setup(
    name='elm',
    version='0.1',
    description='Extreme Learning Machine',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn==0.17.1', 'numpy==1.10.4'],
    py_modules = ['elm.elm', 'elm.cob_elm', 'elm.ml_elm'],
    package_dir = {'': 'src'}
)
