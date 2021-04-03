#!/usr/bin/env python

from setuptools import setup, find_packages


def main():
    setup(
        name="shap4j-data-converter",
        version="0.0.2",
        description="Data converter for shap4j",
        install_requires=['shap', 'numpy'],
        packages=find_packages(),
        scripts=['bin/shap4jconv'],

        author="Xin Yin",
        author_email="xydrolase@gmail.com",
        url="https://github.com/xydrolase/shap4j-data-converter"
    )


if __name__ == "__main__":
    main()
