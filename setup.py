from setuptools import setup, find_packages

setup(
    name="heartbeat_analysis",
    version="0.1.0",
    author="dmt",
    description="Streamlit app for heartbeat classification and signal processing analysis",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "scipy",
        "matplotlib",
        "librosa",
        "soundfile",
        "tensorflow",   # or tensorflow-cpu if GPU not needed
    ],
    extras_require={
        "dev": ["black", "flake8", "pytest"]
    },
    entry_points={
        "console_scripts": [
            "heartbeat-app=main:main"
        ]
    },
    python_requires=">=3.8",
)
