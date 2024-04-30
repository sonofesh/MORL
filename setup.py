from setuptools import setup, find_packages

setup(
    name='morl',
    version='0.0.1',
    description='',
    url='',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['environments/frozen_lake_plus/img/*'], 'demo': ['models/checkpoints/demo_qtable.json']},
    data_files=[('checkpoints', ['models/checkpoints/demo_qtable.json'])]
)