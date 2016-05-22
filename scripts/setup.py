from setuptools import setup

setup(
    name='bachbot',
    version='0.1',
    py_modules=['bachbot'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        bachbot=bachbot:cli
    ''',
)
