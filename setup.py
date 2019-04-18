from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding='utf-8') as req:
    requirements = req.read().split()


setup(
    name="happytal",
    # used in setuptools-git-version https://github.com/pyfidelity/setuptools-git-version
    version_format='{tag}',
    author="Christophe Goudet",
    author_email="goudetchristophe@gmail.com",
    description="A set of tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/cgoudet/pytools",
    packages=find_packages(),
    setup_requires=['setuptools-git-version'],
    install_requires=requirements
)
