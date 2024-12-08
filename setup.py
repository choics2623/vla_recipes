from setuptools import setup, find_packages

setup(
    name='vla_recipes',
    version='0.1.0',
    description='VLA Recipes for Finetuning and Training (Included LLM, VLM Training)',
    author='ChangSu Choi',
    author_email='choics2623@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'accelerate',
        'transformers',
        'datasets',
        'numpy',
        'wandb',
    ],
    python_requires='>=3.6',
)
