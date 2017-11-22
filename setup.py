from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='experiments_template',
    version='0.0.1',
    description='Template for best practices in TensorFlow API usage.',
    long_description=readme,
    author='Dan Busbridge',
    author_email='dan.busbridge@gmail.com',
    url='https://github.com/dbusbridge/experiments_template',
    packages=find_packages(exclude=('tests', 'docs'))
)
