import json
import csv
import os


class HackerRankReader:
    def __init__(self, dataset_csv_path):
        self.dataset_csv_path = dataset_csv_path
        self.data = []

    def read_csv_to_data(self):
        with open(self.dataset_csv_path) as dataset_csv:
            reader = csv.DictReader(dataset_csv)
            # Iterate all exercises
            for row in reader:
                # Exercises can have up to 3 test cases.
                test_cases_keys = ["test_case_1", "test_case_2", "test_case_3"]

                # Create array to store all the test cases
                test_cases = []

                # Collect Test Cases from csv columns
                for test_case_column_key in test_cases_keys:
                    if row[test_case_column_key] != "":
                        test_case = {}
                        test_case_raw = row[test_case_column_key].split(":")
                        test_case["input"] = test_case_raw[0]
                        test_case["output"] = test_case_raw[1]
                        test_cases.append(test_case)

                # Construct Exercise Object
                exercise = {
                    "id": row["id"],
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                    "pdf_path": os.getcwd() + "/" + row["pdf_path"],
                    "test_cases": test_cases
                }

                # Append to data
                self.data.append(exercise)

    def get_data(self):
        return self.data

    def dump_to_file(self):
        with open('data.json', 'w') as outfile:
            json.dump(self.data, outfile)


if __name__ == "__main__":
    dataset_reader = HackerRankReader("datasets/hackerrank/dataset.csv")
    dataset_reader.read_csv_to_data()
    dataset_reader.dump_to_file()
