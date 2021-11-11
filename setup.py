import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="significantdigits-pkg",
    version="0.0.1",
    author="Verificarlo contributors",
    author_email="verificarlo@googlegroups.com",
    description="Significant digits computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/verificarlo/significantdigits",
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    scripts=['significantdigits/scripts/significantdigits']
)
