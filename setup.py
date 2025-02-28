from setuptools import setup, find_packages

setup(
    name="visual_search_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "sqlalchemy",
        "pillow",
        "numpy",
        "fastapi",
    ],
)
