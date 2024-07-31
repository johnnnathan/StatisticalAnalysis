import random
import sys
import datetime
from scipy.stats import skew
import pandas as pd
import json

numbers = []
population = False
N = 0
Mean = 0
Variance = 0


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
        'interquartile_range ' : interquartile_range(numbers),
        'twenty_fifth_percentile ' : percentile(25),
        'seventy_fifth_percentile ' : percentile(75),
        'skewness ' : skewness(),
        'kurtosis ' : kurtosis(),


    }
    return stats

def kurtosis():
    standard_deviation =  Variance ** (1/2)
    summation = 0
    for x in numbers:
        summation += ((x-Mean)/standard_deviation)**4

    # if population:
    #     # Population kurtosis formula
    #     result = (N / ((N - 1) * (N - 2) * (N - 3))) * summation - 3
    # else:
    #     # Sample kurtosis formula
    result = ((N * (N + 1)) / ((N - 1) * (N - 2) * (N - 3))) * summation - (3 * (N - 1) ** 2) / ((N - 2) * (N - 3))


    print("Kurtosis : " + str(result) + "    !!!Using the sample formulat, population formula is in the works")
    return result

def skewness():
    standard_deviation = Variance ** (1/2)
    summation = 0
    for x in numbers:
        summation += (x - Mean) ** 3


    if standard_deviation == 0:  # Avoid division by zero
        return 0

    if (population):
        third_moment = summation / N
        result = (N / ((N - 1) * (N - 2))) * (third_moment / (standard_deviation ** 3))*N ## I do not know why multiplying by N works, but it does
    else:
        third_moment = summation / (N - 1)
        result = ((N * (N - 1)) / ((N - 2))) * (third_moment / (standard_deviation ** 3))/N ## I do not know why dividing by N works, but it does

    print("Skewness : " + str(result) + "     !!!This value can be wrong, it is closer to an estimate than a calculation")
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

    print("Variance is : " + str(Variance))
    return Variance

def standard_deviation():
    sd = pow(Variance, 1/2)
    print ("Standard Deviation : " + str(sd))
    return sd

def median(number_list):
    median = 0
    n = len(number_list)
    if (n % 2 == 0):
        median = (number_list[int(n/2 - 1)] + number_list[int(n/2)]) / 2
    else:
        median =  (number_list[int((n/2))])

    if (N == n):
        print("Median : " + str(median))

    return median

def percentile(percentile):
    index = int((percentile/100)*(N + 1))
    value = numbers[index - 1]
    print("The value at the " + str(percentile) + "th percentile is : " + str(value))
    return value


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


initialize_json()
write_csv(10, 10)
read_csv()
stats = compute_statistics()
update_json_file(numbers,stats)
