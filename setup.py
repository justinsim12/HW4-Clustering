from setuptools import setup, find_packages

setup(
    name="kmeans_clustering",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple implementation of KMeans and Silhouette Score",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kmeans_clustering",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)