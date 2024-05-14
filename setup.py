from setuptools import setup, find_packages

packages = ['wrongbutusefulsbi'] + ['wrongbutusefulsbi.' + p for p in find_packages('wrongbutusefulsbi')]

with open("README.rst", encoding='utf8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# read version number
__version__ = open('wrongbutusefulsbi/__init__.py').readlines()[-1].split(' ')[-1].strip().strip("'\"")

setup(
    name='wrongbutusefulsbi',
    version=__version__,
    packages=packages,
    include_package_data=True,
    author='Ryan Kelly',
    author_email='ryan@kiiii.com',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/RyanJafefKelly/wrongbutusefulsbi',
    install_requires=requirements,
    license='GPLv3',
    python_requires='>=3.7',
    zip_safe=False,
)
