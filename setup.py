from setuptools import setup

setup(
    name='Spotify_Recommender',
    version='1.0',
    install_requires=[
        # Svi ostali paketi iz requirements.txt
        "numpy==1.24.1",
        "pandas==2.2.2",
        "scikit-learn==1.3.1",
        # Dodaj sve ostale pakete iz requirements.txt
    ],
    extras_require={
        "windows": ["pywin32==308"],  # Samo za Windows
    },
)
