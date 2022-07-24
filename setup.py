from setuptools import setup

setup(
    name='Keystroke authentication',
    version='0.0.1',
    packages=['numpy', 'pandas', 'keyboard', 'matplotlib', 'keras', 'sklearn',
              'pickle', 'diceware', 'os', 'tensorflow'],
    url='https://github.com/otisthescribe/keystroke_authentication',
    license='Apache License 2.0',
    author='Grzegorz Kmita, Michał Moskal',
    author_email='gkmita@student.agh.edu.pl, mimoskal@student.agh.edu.pl',
    description='Use of keystrokes dynamics and neural network as components of authentication method'
)
