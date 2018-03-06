from distutils.core import setup

setup(
    name='sas-dataset',
    version='1.0',
    packages=['sas-dataset'],
    url='https://github.com/LC-John/SAS-artificial-dataset',
    license='BSD',
    author='DrLC',
    author_email='zhang_hz@pku.edu.cn',
    description='simple-add-sequence dataset for sequence classification',
    requires=["numpy", "random", "pickle", "gzip"]
)