from distutils.core import setup

setup(
    name='DQN',
    version='0.0.1',
    install_requires=['numpy', 'torchvision', 'gym', 'torch'],
    entry_points={
        'console_scripts': [
            'dqn_learn = dqn.learn:main',
            'dqn_play = dqn.play:main',
            'dqn_explore = dqn.explore:main'
        ]
    },
    packages=['dqn']
)
