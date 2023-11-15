import setuptools
setuptools.setup(
    name="chrislib",
    version="0.0.1",
    author="Chris Careaga",
    author_email="careagc256@gmail.com",
    description="",
    url="",
    packages=setuptools.find_packages(),
    license="",
    python_requires=">3.6",
    install_requires=[
        'beautifulsoup4>=4.12.2',
        'gdown>=4.7.1',
        'huggingface-hub>=0.16.4',
        'imageio>=2.31.3',
        'kornia>=0.7.0',
        'matplotli>=3.7.2',
        'numpy>=1.24.4',
        'opencv-python>=4.8.0.76',
        'Pillow>=10.0.0',
        'requests>=2.31.0',
        'scikit-image>=0.21.0',
        'scip>=1.10.1',
        'torch>=2.0.1',
        'torchvision>=0.15.2',
        'tqdm>=4.66.1',
    ]
)
