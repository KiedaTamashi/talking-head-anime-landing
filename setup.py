from setuptools import setup, find_packages
from codecs import open
from os import path
# here = path.abspath(path.dirname(__file__))
setup(
    name='talkingHeadAnimeLanding',
    version='0.1',
    description='Customize New Features Based-on Colored Logger',
    # long_description=str(open(path.join(here, "Learning tracker")).read()),
    # The project's main homepage.
    url='opconty - Overview',
    # Author details
    author='tanzhenwei',
    author_email='kiedatamashi@gmail.com',
    # Choose your license
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: System :: Logging',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    py_modules=["talkingHeadAnime"],
    install_requires=[
                      'waifulabs',
                      'torch >= 1.4.0',
                      'pillow >= 6.2.2',
                      'dlib>=19.19',
                      'numpy >= 1.17.l2',
                      'opencv-python >= 4.1.0.30'
                      ]
)