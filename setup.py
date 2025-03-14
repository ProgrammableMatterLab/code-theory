import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="block",
    version="0.1.0",
    author="Guy Cohen",
    author_email="gcohen.dev@gmail.com",
    description="library for evaluating interactions between magnet configurations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ProgrammableMatterLab/code-theory",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="MIT",  # Explicitly specify the license
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # Add dependencies if needed
    # install_requires=['dependency1', 'dependency2'],
)
