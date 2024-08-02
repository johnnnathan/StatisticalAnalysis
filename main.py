import math
import random
import sys
import datetime
import pandas as pd
import json
from scipy import stats

numbers = []
population = False
N = 0
Mean = 0
Variance = 0
QUARTER = 25
THIRD_QUARTER = 75
WINSORIZED_AMOUNT = 20
H0Mean = 5
Value = 4
p = 0


## Initializations and other setup method exist in this plane, as well as handling of data file and output file
## ------------------------------------------------------------------------------------------------------------
def is_population():
    global population
    type = input("Is this a (P)opulation or a (S)ample?")
    if (type.lower() == "p"):
        population = True
    elif (type.lower() == "s"):
        population = False
    else:
        print("No known type...aborting")
        sys.exit(1)


def initialize_json():
    data = {
        "numbers": [],
        "statistics": {}
    }

    # Write the initial structure to the JSON file
    with open('data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Initialized JSON file 'data.json'.")


def update_json_file(numbers=None, stats=None, filename='data.json'):
    with open(filename, 'r+') as json_file:
        data = json.load(json_file)

        if numbers is not None and len(numbers) < 101:
            data['numbers'].extend(numbers)

        if stats is not None:
            data['statistics'].update(stats)

        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()


def read_csv():
    global numbers, N
    data = pd.read_csv('data.csv')

    # Extract the first column
    numbers = data.iloc[:, 0].tolist()
    numbers.sort()

    # Print the list
    print(numbers)
    N = len(numbers)


def write_csv(count, max_range):
    # Set the random seed for reproducibility (optional)
    random.seed(int(datetime.datetime.now().timestamp()))

    # Generate a list of 200 random numbers within the range 0 to 100
    numbers = [random.randint(0, max_range) for _ in range(count)]

    # Create a DataFrame
    df = pd.DataFrame(numbers, columns=['numbers'])

    # Save the DataFrame to a CSV file
    df.to_csv('data.csv', index=False)

    print("CSV file 'data.csv' created successfully.")


## Statistical analysis starts from this point
## -------------------------------------------


def compute_statistics():
    stats = {
        'mean': mean(),
        'median': median(numbers),
        'mode ': mode(),
        'range ': Range(),
        'variance': variance(),
        'standard_deviation': standard_deviation(),
        'interquartile_range ': interquartile_range(numbers),
        'twenty_fifth_percentile ': percentile(QUARTER),
        'seventy_fifth_percentile ': percentile(THIRD_QUARTER),
        'skewness ': skewness(),
        'kurtosis ': kurtosis(),
        'mad ': mad(),
        'winsorized_mean ': winsorized_mean(WINSORIZED_AMOUNT),
        'covariance ': covariance([0, 5, 5, 5, 5, 5, 5, 10, 10, 10], 5.5),
        'correlation_coefficient ': correlation([0, 5, 5, 5, 5, 5, 5, 10, 10, 10], 6),
        't_statistic ': t_statistic(H0Mean),
        'z_statistic ': z_score(Value),
        'p_value ' : p_value(1.5,3.03, "two"),
        'confidence_test ' : confidence(0.05, p),
        'confidence_interval ' : confidence_interval(0.95)

    }
    return stats

def confidence(confidence_amount, p_value):
    if (p_value <= confidence_amount):
        message = "Reject Null Hypothesis"
        print(message)
        return message
    else:
        message = "Fail to reject Null Hypothesis"
        print(message)
        return message

def z_score(Value):
    z = round((Value - Mean) / (Variance ** (1 / 2)), 3)
    print("Z-Score for " + str(Value) + " is : " + str(z))
    return z


def correlation(numbersY, meanY):
    summation_x_square, summation_y_square, summation_xy = 0, 0, 0
    for x in numbers:
        summation_x_square += (x - Mean) ** 2
    for y in numbersY:
        summation_y_square += (y - meanY) ** 2
    for x, y in zip(numbers, numbersY):
        summation_xy += (x - Mean) * (y - meanY)

    correlation = summation_xy / ((summation_y_square * summation_x_square) ** (1 / 2))
    correlation = round(correlation, 3)
    print("Pearson Correlation Coefficient : " + str(correlation))



def phi(z):
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def t_cdf(t, df):
    # Approximate the CDF of the Student's t-distribution
    gamma_term = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return 0.5 + (t * gamma_term * (1 + (t**2 / df))**(-((df + 1) / 2)))


def p_value(mean_tested, standard_deviation_pop, test_type):
    global p
    if test_type not in ["one", "two"]:
        print("Invalid test type. Please use 'one' or 'two'.")
        return None

    # Calculate the t-score or z-score
    if N <= 30 and standard_deviation_pop is None:
        # T-Test
        score = (Mean - mean_tested) / (Variance ** 0.5 / N ** 0.5)
        df = N - 1  # degrees of freedom
        if test_type == 'two':
            p_value = 2 * (1 - stats.t.cdf(abs(score), df))
        elif test_type == 'one':
            p_value = 1 - stats.stats.t.cdf(score, df) if score > 0 else stats.stats.t.cdf(score, df)
    elif standard_deviation_pop is not None:
        # Z-Test
        z_score = (Mean - mean_tested) / (standard_deviation_pop / math.sqrt(N))
        if test_type == 'two':
            p_value = 2 * (1 - phi(abs(z_score)))
        elif test_type == 'one':
            p_value = 1 - phi(z_score) if z_score > 0 else phi(z_score)
    else:
        print("Insufficient data for hypothesis testing.")
        return None

    p_value = round(p_value, 3)
    print("p-value : " + str(p_value))
    p = p_value
    return p_value


def t_statistic(mean):
    t = round((Mean - mean) / ((Variance ** (1 / 2)) / (N ** (1 / 2))), 4)
    print("t-Statistic : " + str(t))
    return t


def winsorized_mean(amount):
    lower_bound = int((amount / 100) * N)
    upper_bound = int(((100 - amount) / 100) * N)
    summation = 0
    for x in numbers[lower_bound: upper_bound]:
        summation += x

    summation += lower_bound * numbers[lower_bound]
    summation += (N - upper_bound) * numbers[upper_bound - 1]
    winsorized_mean = summation / (N)
    print("Winsorized Mean : " + str(winsorized_mean))
    return winsorized_mean


def mad():
    summation = 0
    for x in numbers:
        summation += abs(x - Mean)
    mad = summation / N
    mad = round(mad, 3)
    print("MAD : " + str(mad))
    return mad


def kurtosis():
    standard_deviation = Variance ** (1 / 2)
    summation = 0
    for x in numbers:
        summation += ((x - Mean) / standard_deviation) ** 4

    # if population:
    #     # Population kurtosis formula
    #     result = (N / ((N - 1) * (N - 2) * (N - 3))) * summation - 3
    # else:
    #     # Sample kurtosis formula
    result = ((N * (N + 1)) / ((N - 1) * (N - 2) * (N - 3))) * summation - (3 * (N - 1) ** 2) / ((N - 2) * (N - 3))

    result = round(result, 3)
    print("Kurtosis : " + str(result) + "    !!!Using the sample formula, population formula is in the works")
    return result


def covariance(numbersY, meanY):
    if (len(numbersY) != N):
        return None
    summation = 0
    for x, y in zip(numbers, numbersY):
        summation += (x - Mean) * (y - meanY)

    if population:
        covariance = summation / N
    else:
        covariance = summation / (N - 1)

    covariance = round(covariance, 3)
    print("Covariance : " + str(covariance))
    return covariance


def skewness():
    standard_deviation = Variance ** (1 / 2)
    summation = 0
    for x in numbers:
        summation += (x - Mean) ** 3

    if standard_deviation == 0:  # Avoid division by zero
        return 0

    if (population):
        third_moment = summation / N
        result = (N / ((N - 1) * (N - 2))) * (third_moment / (
                standard_deviation ** 3)) * N  ## I do not know why multiplying by N works, but it does
    else:
        third_moment = summation / (N - 1)
        result = ((N * (N - 1)) / ((N - 2))) * (
                third_moment / (standard_deviation ** 3)) / N  ## I do not know why dividing by N works, but it does

    result = round(result, 3)
    print(
        "Skewness : " + str(result) + "     !!!This value can be wrong, it is closer to an estimate than a calculation")
    return result


def interquartile_range(array):
    n = len(array)
    mid = n // 2
    if n % 2 == 0:
        left_half = array[:mid]
        right_half = array[mid:]
    else:
        left_half = array[:mid]
        right_half = array[mid + 1:]

    interquartile_range = median(right_half) - median(left_half)
    print("InterQuartile Range : " + str(interquartile_range))
    return interquartile_range


def mean():
    global numbers, Mean
    total = 0
    for element in numbers:
        total += element

    Mean = total / len(numbers)

    print("Mean is:" + str(Mean))
    return Mean


def variance():
    global Variance
    mean = Mean
    summation = 0
    for element in numbers:
        summation += pow(element - mean, 2)
    if (population):
        Variance = summation / N
    else:
        Variance = summation / (N - 1)

    Variance = round(Variance, 3)
    print("Variance is : " + str(Variance))
    return Variance


def standard_deviation():
    sd = round(pow(Variance, 1 / 2), 3)
    print("Standard Deviation : " + str(sd))
    return sd


def median(number_list):
    median = 0
    n = len(number_list)
    if (n % 2 == 0):
        median = (number_list[int(n / 2 - 1)] + number_list[int(n / 2)]) / 2
    else:
        median = (number_list[int((n / 2))])

    if (N == n):
        print("Median : " + str(median))

    return median


def percentile(percentile):
    index = int((percentile / 100) * (N + 1))
    value = numbers[index - 1]
    print("The value at the " + str(percentile) + "th percentile is : " + str(value))
    return value

def confidence_interval(confidence):
    confidence_interval = [0,0]
    if N <= 30:
        t_score = stats.t.ppf((1 + confidence) / 2, N - 1)
        margin_of_error = t_score * math.sqrt(Variance / N)
        confidence_interval[0] = round(Mean - margin_of_error , 3)
        confidence_interval[1] = round(Mean + margin_of_error , 3)
    else:
        # For large samples (N > 30), use the z-distribution
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_of_error = z_score * math.sqrt(Variance / N)
        confidence_interval[0] = round(Mean - margin_of_error, 3)
        confidence_interval[1] = round(Mean + margin_of_error , 3)

    print("Confidence Interval : " + str(confidence_interval[0]) + " - " + str(confidence_interval[1]))
    return confidence_interval

def mode():
    occurence_dict = dict()
    for x in numbers:
        if x not in occurence_dict:
            occurence_dict[x] = 1
        else:
            occurence_dict[x] += 1

    max = 0
    maxOcc = 0
    for key, count in occurence_dict.items():
        if (count > maxOcc):
            max = key
            maxOcc = count

    print("Mode : " + str(max) + " at : " + str(maxOcc) + " occurences")
    return maxOcc


def Range():
    Range = numbers[-1] - numbers[0]
    print("Range : " + str(Range))
    return Range

def test_p_value():
    global numbers, N, Mean, Variance

    # Sample data
    numbers = [5, -2, 3, -1, 0, 4, 2, -3, 6, 1]
    N = len(numbers)
    Mean = sum(numbers) / N
    Variance = sum((x - Mean) ** 2 for x in numbers) / (N - 1)  # Sample variance

    # Testing the p_value method
    mean_tested = 0
    standard_deviation_pop = None
    test_type = "two"
    p_val = p_value(mean_tested, standard_deviation_pop, test_type)
    print(f"Tested p_value for mean_tested={mean_tested}, standard_deviation_pop={standard_deviation_pop}, test_type='{test_type}'")
    print(f"Resulting p-value: {p_val}")


initialize_json()
read_csv()
stats = compute_statistics()
update_json_file(numbers, stats)
