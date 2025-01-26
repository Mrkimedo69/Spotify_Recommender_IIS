from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name='Spotify_Recommender',
    version='1.0',
    description='Spotify song recommender system using machine learning and data processing techniques.',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": ["pytest", "flake8"],
    },
    entry_points={
        'console_scripts': [
            'spotify-recommender=agent_app:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
