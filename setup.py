import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setuptools.setup(
    name='AutoScore',
    version='0.0.1',
    author='Digital Medicine Lab, Duke NUS Medical School',
    author_email='wuqiming@u.nus.edu',
    description='AutoScore: An Interpretable Machine Learning-Based Automatic Clinical Score Generator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nliulab/AutoScore-Python',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    packages=['AutoScore'],
    python_requires='>=3.8',
    install_requires=requirements)
