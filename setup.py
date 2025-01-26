from setuptools import setup, find_packages

setup(
    name="topsis-adityapandey-102217092",
    version="0.5",
    author="Aditya Pandey",
    author_email="asadityasonu@gmail.com",
    description="A Python package for TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/asadityasonu/Topsiss-aditya-102217092",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.6',
    keywords="topsis decision-making MCDM ranking",
    project_urls={
        "Bug Tracker": "https://github.com/asadityasonu/Topsiss-aditya-102217092/issues",
        "Documentation": "https://github.com/asadityasonu/Topsiss-aditya-102217092",
        "Source Code": "https://github.com/asadityasonu/Topsiss-aditya-102217092",
    },
)