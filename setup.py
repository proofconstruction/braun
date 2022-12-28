import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("braun/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]

installreqs = list()
with open("requirements.txt") as reqs:
    for requirement in reqs:
        installreqs.append(requirement)

setuptools.setup(
    name="braun",
    version=version,
    author="proofconstruction",
    description="Lots of model trains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/proofconstruction/braun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=installreqs,
)
