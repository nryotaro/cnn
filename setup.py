from setuptools import setup, find_packages

setup(
    name='hello_cnn',
    version='0.0.1',
    description='A text classifier that leveragge CNN',
    url='https://github.com/nryotaro/hello_cnn',
    author='Nakamura, Ryotaro',
    author_email='nakamura.ryotaro.kzs@gmail.com',
    license='Copyright (c) 2017 Nakamura, Ryotaro',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6'],
    packages=find_packages(exclude=['tests']),
    install_requires=['tensorflow'],
    python_requires='>=3.6.1',
    extras_require={'dev': ['pytest', 'jupyter']},
    entry_points={
        'console_scripts': [
            'hello_cnn=hello_cnn:main',
        ],
    })
