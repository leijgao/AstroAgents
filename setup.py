from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="astroagents",
    version="0.1.0",
    author="AstroAgents Team",
    author_email="your.email@example.com",
    description="Multi-Agent AI for Hypothesis Generation from Mass Spectrometry Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AstroAgents/AstroAgents",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "astroagents=AstroAgents:main",
        ],
    },
) 