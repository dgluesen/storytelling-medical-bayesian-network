import setuptools

with open("../README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="medical_storytelling",
    version="0.0.1",
    author="Dennis Gluesenkamp, Ritvik Marwaha",
    author_email=" data@gluesenkamp.info, rmarwaha@posteo.net ",
    description="Data Storytelling for medical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dgluesen/storytelling-medical-bayesian-network",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

