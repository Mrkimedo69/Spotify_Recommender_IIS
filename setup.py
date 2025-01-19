from setuptools import setup, find_packages

# Učitaj sadržaj requirements.txt
def parse_requirements(filename):
    """Parse a requirements.txt file into a list of dependencies."""
    with open(filename, "r") as file:
        return file.read().splitlines()

setup(
    name='Spotify_Recommender',
    version='1.0',
    description='Spotify song recommender system using machine learning and data processing techniques.',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),  # Učitaj sve iz requirements.txt
    extras_require={
        "windows": ["pywin32==308"],  # Dodaj specifične ovisnosti za Windows ako treba
    },
    entry_points={
        'console_scripts': [
            'spotify-recommender=agent_app:main',  # Ako imaš `main` funkciju u `agent_app.py`
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Verzija Pythona koja je potrebna
)
