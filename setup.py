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
        'beautifulsoup4>=4.11.2',
        'imageio>=2.31.3',
        'kornia>=0.7.0',
        'matplotlib>=3.7.1',
        'numpy>=1.23.5',
        'opencv-python>=4.8.0.76',
        'Pillow>=9.4.0',
        'requests>=2.31.0',
        'scikit-image>=0.19.3',
        'scipy>=1.10.1',
        'torch>=2.0.1',
        'torchvision>=0.15.2',
        'tqdm>=4.66.1',
    ]
)
