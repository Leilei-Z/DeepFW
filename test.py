import torch
from data_preparation import prepare_data
from models.FFAN import FFANModel
from models.classifier import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'dataset_demo/DIR-300.csv'
_, test_triplets, data = prepare_data(file_path)

input_dim = len(data.columns) - 2
hidden_dim = 128
num_classes = len(data['version'].unique())

mamba_model = FFANModel(input_dim, hidden_dim, device).to(device)
classifier = Classifier(hidden_dim * 3, num_classes).to(device)

mamba_model.load_state_dict(torch.load('./saved_models/' + file_path.split('/')[-1].split('.')[0] + '.pth'))
classifier.load_state_dict(torch.load('./saved_models/'+ file_path.split('/')[-1].split('.')[0] + '_cls.pth'))

version_to_index = {version: idx for idx, version in enumerate(data['version'].unique())}
test_triplets = [(torch.tensor(triplet[0][2:].astype(float).values, dtype=torch.float32).to(device).unsqueeze(0),
                  torch.tensor([version_to_index[triplet[0]['version']]], dtype=torch.long).to(device))
                 for triplet in test_triplets]

correct = 0
total = 0
all_labels = []
all_probs = []
with torch.no_grad():
    for anchor, label in test_triplets:
        encoded = mamba_model(anchor)
        output = classifier(encoded)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        all_labels.append(label.cpu().numpy())
        all_probs.append(output.cpu().numpy())

print(f'Accuracy of the classifier on the test samples: {100 * correct / total:.2f}%')
