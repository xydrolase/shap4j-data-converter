from setuptools import setup


def main():
    setup(
        name="shap4j-data-converter",
        version="0.0.1",
        description="Data converter for shap4j",
        install_requires=['shap', 'numpy'],
        scripts=['bin/shap4jconv'],

        author="Xin Yin",
        author_email="xydrolase@gmail.com"
    )


if __name__ == "__main__":
    main()
