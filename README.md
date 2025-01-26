# Topsis Python Package

**Developed by:** Aditya Pandey - 102217092

## Overview

The Topsis (Technique for Order of Preference by Similarity to Ideal Solution) method is a popular multi-criteria decision-making technique. It helps to evaluate and select the best alternative from a set of options based on their proximity to the ideal solution. This Python package implements the Topsis method, making it easy to apply the technique on datasets with multiple criteria.

## Features
- Computes Topsis scores based on input data, weights, and impacts.
- Ranks alternatives according to the Topsis methodology.
- Supports data input through Excel files.
- Provides an easy-to-use command-line interface for seamless integration.

## Installation

To install the Topsis Python package, use `pip` with the following command:

```bash
pip install 102217092-Aditya-topsis
```

## Usage

### Command-Line Input

To use the Topsis package from the command line, run the following command:

```sh
python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>
```

- **Input File Type**: The input file must be an Excel file.
- **Data Format**: The second to last columns of the data file MUST contain numeric values.
- **Impacts**: Impacts should be either '+' (positive) or '-' (negative).
- **Weights and Impacts**: Weights and impacts should be enclosed in double quotes and separated by commas.
- **Output**: The output will include a 'Topsis Score' column and a 'Rank' column added to the data. The results will be saved to a CSV file specified in the command-line arguments.

## Example

### Command-Line Input Example

```sh
python 102217092.py data.csv "1,1,1,1,1" "+,+,-,+,-" 102217092-result.csv
```




