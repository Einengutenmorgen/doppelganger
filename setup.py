from setuptools import setup, find_packages

setup(
    name="doppelganger",  # Replace with your actual project name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "emoji>=1.7.0",
        "openai>=1.58.1",
        'rouge-score',
        'transformers',
        'torch',
        'einops',
        'numpy<2',
        'scikit-learn',
        'python-dotenv',
    ],
    scripts=['bin/run_preprocessor', 'bin/create_personas','bin/create_posts', 'bin/evaluate'],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
            "python-dotenv>=0.19.0",
            "mypy>=0.910",
            "types-emoji>=2.0.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
        ],
    },
    python_requires=">=3.8",
)