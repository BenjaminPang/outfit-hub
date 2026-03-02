from setuptools import setup, find_packages


# Read requirements.txt for dependencies
with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="outfit_hub",
    version="0.0.1",
    author="pkc",
    description="A unified framework for fashion outfit dataset management and processing.",
    # Automatically find all sub-packages (core, datasets, processors, etc.)
    packages=find_packages(),
    # Include non-python files (like registry.yaml) specified in MANIFEST.in or package_data
    include_package_data=True,
    package_data={
        "outfit_hub": ["*.yaml"],
    },
    install_requires=required,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)