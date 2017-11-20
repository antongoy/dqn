from distutils.core import setup

setup(
    name='DQM',
    version='0.0.1',
    install_requires=['numpy', 'torchvision', 'gym'],
    packages=['dqn']
)
