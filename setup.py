from setuptools import setup, find_packages

# Load the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tina",  # Package name
    version="0.1.1",  # Updated version
    author="Edwin Kestler",
    author_email="your-email@example.com",
    description="TINA - Optimized Layers for PyTorch and ONNX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdwinKestler/TINA",  # Project URL
    packages=find_packages(include=["tina", "tina.*"]),  # Include the tina package and subpackages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[  # Dependencies
        "torch>=1.9",
        "onnx>=1.10",
        "onnxruntime>=1.10",
        "numpy>=1.18",
    ],
    entry_points={  # Console scripts or commands
        "console_scripts": [
            "tina-example=tina.examples.pytorch.mnist_example:main",
        ],
    },
)
