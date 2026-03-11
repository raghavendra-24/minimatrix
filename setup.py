from setuptools import setup, find_packages

setup(
    name="minimatrix",
    version="0.2.0",
    author="Raghavendra Raju Palagani",
    author_email="raghavendrapalagani671@gmail.com",
    description="A pure-Python linear algebra library with a custom Matrix type — built for learning, correctness, and elegance.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raghavendra-24/minimatrix",
    packages=find_packages(include=["minimatrix", "minimatrix.*"]),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Education",
        "Development Status :: 4 - Beta",
    ],
    keywords="matrix linear-algebra mathematics pure-python education fraction linalg",
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
