import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    triplets = []
    for version, version_group in data.groupby('version'):
        filenames = version_group['file'].unique()
        for i, anchor_filename in enumerate(filenames):
            anchor = version_group[version_group['file'] == anchor_filename].iloc[0]
            positive_filename = np.random.choice(filenames[filenames != anchor_filename])
            positive = version_group[version_group['file'] == positive_filename].iloc[0]
            other_versions = data[data['version'] != version]
            negative_candidates = other_versions[other_versions['file'] == anchor_filename]
            if len(negative_candidates) > 0:
                negative = negative_candidates.sample(1).iloc[0]
                triplets.append((anchor, positive, negative))

    train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
    return train_triplets, test_triplets, data
