# Statistical Analysis Tool by Dimitrios Tsiplakis

This project provides a statistical analysis tool written in Python. It allows you to perform various statistical calculations on a given dataset, including mean, median, mode, variance, standard deviation, and confidence intervals. The tool can handle both population and sample data.

## Features

- Compute basic statistics: mean, median, mode, range, variance, standard deviation, and more.
- Calculate skewness and kurtosis.
- Perform hypothesis testing and calculate p-values.
- Calculate confidence intervals for both small and large samples.
- Read input data from a CSV file.
- Save and update results in a JSON file.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: `pandas`, `scipy`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/statistical-analysis-tool.git
    cd statistical-analysis-tool
    ```

2. Install the required libraries:
    ```bash
    pip install pandas scipy
    ```

### Usage

1. Prepare your input data in a CSV file named `data.csv` with a single column of numbers.

2. Run the script to perform the statistical analysis:
    ```bash
    python statistical_analysis.py
    ```

3. The script will read the data from `data.csv`, compute the statistics, and save the results to `data.json`.


### Licence

This project is licensed under the MIT License. See the LICENSE file for details.
