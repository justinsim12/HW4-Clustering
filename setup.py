from setuptools import setup, find_packages

setup(
    name="cluster_analysis",
    version="0.1.0",
    author="Justin Sim",
    author_email="justin.sim@ucsf.edu",
    description="A simple implementation of KMeans and Silhouette Score",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/justinsim12/HW4-Clustering",  
    packages=find_packages(include=["cluster", "cluster.*"]),  
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest","scikit-learn"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)