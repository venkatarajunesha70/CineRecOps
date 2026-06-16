from setuptools import setup, find_packages

setup(
    name="cinerecops",
    version="1.0.0",
    description="Production-Ready Movie Recommendation Platform with MLOps",
    author="CineRecOps Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15.0",
        "mlflow>=2.9.0",
        "fastapi>=0.109.0",
        "pydantic>=2.5.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "cinerecops-train=training.train:main",
            "cinerecops-serve=serving.server:main",
        ]
    },
)
