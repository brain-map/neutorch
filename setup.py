#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='neutorch',
    version='0.0.1',
    description='Deep Learning for brain connectomics using PyTorch',
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    url='https://github.com/brain-connectome/neutorch',
    packages=find_packages(exclude=['bin']),
    entry_points='''
        [console_scripts]
        neutrain-sem=neutorch.train.semantic:main
        neutrain-organelle=neutorch.train.organelle:main
        neutrain-mito=neutorch.train.mitochondria:main
        neutrain-pre=neutorch.train.pre_synapses:main
        neutrain-denoise=neutorch.train.denoise:main
        neutrain-post=neutorch.train.post_synapses:main
        neutrain-affs=neutorch.train.affinity_map:main
        neutrain-affs-vol=neutorch.train.whole_brain_affinity_map:main
        neutrain-ba=neutorch.train.boundary_aug:main
    ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: Apache Software License",
        'Topic :: Communications :: Email',
        'Topic :: Software Development :: Bug Tracking',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
)