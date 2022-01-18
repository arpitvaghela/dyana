from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning",
        "blobfile",
        "numpy",
        "sympy",
        "wandb",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
    ],
)
