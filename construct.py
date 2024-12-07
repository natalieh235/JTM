import csv

path = "../musicnet/musicnet"
metadata = "../musicnet/_metadata.csv"

metadata_train_path = "../musicnet/_metadata_small.csv"
metadata_test_path = "../musicnet/_metadata_small_test.csv"
train_path = "../musicnet/musicnet/train_small"


def create_small_musicnet(train_size, test_size):
    metadata_train = []
    metadata_test = []
    fieldnames = None
    with open(metadata) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        i = 0
        for row in reader:
            if i < train_size:
                metadata_train.append(row)
            else:
                metadata_test.append(row)

            i += 1
            if i == train_size + test_size:
                break
    
    with open(metadata_train_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

        for row in metadata_train:
            writer.writerow(row)
    
    with open(metadata_test_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

        for row in metadata_test:
            writer.writerow(row)

if __name__ == "__main__":
    create_small_musicnet(10, 5)
