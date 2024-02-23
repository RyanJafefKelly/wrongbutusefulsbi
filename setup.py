from setuptools import setup

with open("README.rst", encoding='utf8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='wrongbutusefulsbi',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author='Ryan Kelly',
    author_email='ryan@kiiii.com',
    url='https://github.com/RyanJafefKelly/wrongbutusefulsbi',
    install_requires=requirements,
    license='GPLv3',
    python_requires='>=3.7',
    packages=['wrongbutusefulsbi'],
)
