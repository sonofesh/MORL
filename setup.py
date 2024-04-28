from setuptools import setup, find_packages

setup(
    name='morl',
    version='0.0.1',
    description='',
    url='',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['environments/frozen_lake_plus/img/*']}
)