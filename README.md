# IBNE_ScalableFS

This repository contains code for a feature selection pipeline designed for machine learning tasks. The main entry point is `pipeline.py`, which leverages modular functions provided in the `feature_selection` package. Pipeline parameters are configured in `opt.py` and are passed as command-line arguments when executing the pipeline.

## Table of Contents
1. [Overview](##overview)
2. [Directory Structure](##directory-structure)
3. [Installation](##installation)
4. [Usage](##usage)
5. [Examples](##examples)
6. [Citation](##citation)
7. [Contributing](##contributing)
8. [License](##license)

## Overview

The feature selection pipeline in this repository is designed to improve the efficiency and accuracy of machine learning in clinical environment models by identifying and selecting the most relevant features.

All feature selection logic is encapsulated in the `feature_selection` package, enabling users to modify and extend functionalities easily.

## Directory Structure

	├── pipeline.py # Main script to run the feature selection pipeline 
	├── feature_selection/ # Package containing feature selection methods 
	│ ├── init.py # Initializes the package
	| ├── featureSelection.py # Contains the feature selection functions 
	│ ├── general.py # Importing function and basic 
	│ ├── models.py # training and testing
	│ ├── plots.py # Plot wrapper
	│ └── statistics.py # Wrapper for statistic functions
	├── opt.py # Pipeline configuration parameters 
	├── SINPAIN # directory containing results from SINPAIN dataset
	├── SINPAIN # directory containing results from SINPAIN dataset
	└── README.md # Project documentation
	
## Installation

To install dependencies, clone the repository and run:

```bash
git clone https://github.com/maruotto/IBNE_ScalableFS.git
cd IBNE_ScalableFS
pip install -r requirements.txt
```
## Usage

To run the feature selection pipeline, use `pipeline.py` with parameters specified in cli, they are coded in `opt.py`. The only parameter that can't be controlled through the cli are the seeds in input to the algorithm. They are contained in `models.py`


## Citation
If you use this codebase in your research, please consider citing us. 
For further information, please write to ida23@ru.is


``` bibtex
```

By citing our work, you help support the open-source and academic community, making it easier for others to find and build on this project.


## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or bug fixes.

## Licence
This project is licensed under the MIT License. See the `LICENSE` file for details.
