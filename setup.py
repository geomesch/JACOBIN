from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('jacobin/__init__.py', 'r') as f:
    ver = next(f).split('=')[1].strip()[1:-1]

setup(	
      install_requires=['numpy', 'jax', 'jaxlib'],
      include_package_data=True,
      name="jacobin",
      version=ver,
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      python_requires=">=3.7",
      classifiers=[
              "Programming Language :: Python :: 3.7",
	      "Programming Language :: Python :: 3.8",
	      "Programming Language :: Python :: 3.9",
	      "Programming Language :: Python :: 3.10",
	      "Programming Language :: Python :: 3.11",
	      "Development Status :: 5 - Production/Stable",
	      "Topic :: Scientific/Engineering",
              "Operating System :: OS Independent"])
