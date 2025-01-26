import sys
import pandas as pd
import numpy as np


def topsis(input_file, weights, impacts, result_file):
    try:
        data = pd.read_csv(input_file)

        if data.shape[1] < 3:
            raise ValueError("Input file must have at least three columns (ID, Criteria1, Criteria2, ...).")

        criteria = data.columns[1:]
        matrix = data.iloc[:, 1:].to_numpy()

        weights = list(map(float, weights.split(',')))
        impacts = impacts.split(',')

        if len(weights) != len(criteria) or len(impacts) != len(criteria):
            raise ValueError("Number of weights and impacts must match the number of criteria.")

        if not all(i in ['+', '-'] for i in impacts):
            raise ValueError("Impacts must be '+' or '-'.")

        norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

        weighted_matrix = norm_matrix * weights

        ideal_best = [max(weighted_matrix[:, i]) if impacts[i] == '+' else min(weighted_matrix[:, i]) for i in
                      range(len(criteria))]
        ideal_worst = [min(weighted_matrix[:, i]) if impacts[i] == '+' else max(weighted_matrix[:, i]) for i in
                       range(len(criteria))]

        dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

        scores = dist_worst / (dist_best + dist_worst)

        data['Topsis Score'] = scores
        data['Rank'] = pd.Series(scores).rank(ascending=False).astype(int)

        data.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        topsis(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
