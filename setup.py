from setuptools import setup

setup(
    name='darsakx',
    version='1.0.0',
    description='X-ray Telescope: Ray trace',
    author='Neeraj',
    author_email='neerajt454@gmail.com',
    url='https://github.com/Neerajti32/darsakx',
    packages=['darsakx'],
    install_requires=['numpy', 'scipy','multiprocessing','matplotlib',' tabulate'],
)