from setuptools import setup, find_packages

setup(
    name="cluster_analysis",  # Change to a unique package name
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple implementation of KMeans and Silhouette Score",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cluster_analysis",  # Update with your repo
    packages=find_packages(include=["cluster", "cluster.*"]),  # Ensure submodules are included
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