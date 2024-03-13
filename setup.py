import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miluv",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<>",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pymlg @ git+https://github.com/decargroup/pymlg@main",
    ],
)
