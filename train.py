import torch
import torch.optim as optim
from data_preparation import prepare_data
from models.FFAN import FFANModel
from models.classifier import Classifier
from Losses.HCTCL import HCTCLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'dataset_demo/DIR-300.csv'
train_triplets, _, data = prepare_data(file_path)

input_dim = len(data.columns) - 2  # 特征向量维度
hidden_dim = 128  # 隐藏层维度
num_classes = len(data['version'].unique())

mamba_model = FFANModel(input_dim, hidden_dim, device).to(device)
classifier = Classifier(hidden_dim * 3, num_classes).to(device)
criterion_hctc = HCTCLoss(margin=1, lambda_reg=0.1, num_classes=num_classes, encoding_dim=hidden_dim * 3, device=device).to(device)
criterion_ce = torch.nn.CrossEntropyLoss().to(device)

optimizer_mamba = optim.Adam(mamba_model.parameters(), lr=0.001)
optimizer_cls = optim.Adam(classifier.parameters(), lr=0.01)
hctc_optimizer = optim.SGD(criterion_hctc.parameters(), lr=0.001)

version_to_index = {version: idx for idx, version in enumerate(data['version'].unique())}
train_triplets = [(torch.tensor(triplet[0][2:].astype(float).values, dtype=torch.float32).to(device).unsqueeze(0),
                   torch.tensor(triplet[1][2:].astype(float).values, dtype=torch.float32).to(device).unsqueeze(0),
                   torch.tensor(triplet[2][2:].astype(float).values, dtype=torch.float32).to(device).unsqueeze(0),
                   torch.tensor([version_to_index[triplet[0]['version']]], dtype=torch.long).to(device))
                  for triplet in train_triplets]

batch_size = 32
num_epochs = 50
batched_train_triplets = [train_triplets[i:i + batch_size] for i in range(0, len(train_triplets), batch_size)]
if len(batched_train_triplets[-1]) < batch_size:
    batched_train_triplets = batched_train_triplets[:-1]

lambda_cls = 1
lambda_hctc = 1

for epoch in range(num_epochs):
    epoch_ce_loss = 0.0
    epoch_hctc_loss = 0.0
    for batch_triplets in batched_train_triplets:
        optimizer_mamba.zero_grad()
        optimizer_cls.zero_grad()
        hctc_optimizer.zero_grad()
        batch_ce_loss = 0.0
        batch_hctc_loss = 0.0
        for anchor, positive, negative, label in batch_triplets:
            anchor_output = mamba_model(anchor)
            positive_output = mamba_model(positive)
            negative_output = mamba_model(negative)
            ce_loss = criterion_ce(classifier(anchor_output), label)
            hctc_loss = criterion_hctc(anchor_output, negative_output, label)
            batch_ce_loss += ce_loss.item()
            batch_hctc_loss += hctc_loss.item()
            loss = lambda_cls * ce_loss + lambda_hctc * hctc_loss
            loss.backward()
        optimizer_mamba.step()
        optimizer_cls.step()
        hctc_optimizer.step()
        batch_ce_loss /= len(batch_triplets)
        batch_hctc_loss /= len(batch_triplets)
        epoch_ce_loss += batch_ce_loss
        epoch_hctc_loss += batch_hctc_loss
    epoch_ce_loss /= len(batched_train_triplets)
    epoch_hctc_loss /= len(batched_train_triplets)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train CE Loss: {epoch_ce_loss:.4f}, Train HCTC Loss: {epoch_hctc_loss:.4f}')

torch.save(mamba_model.state_dict(), './saved_models/' + file_path.split('/')[-1].split('.')[0] + '.pth')
torch.save(classifier.state_dict(), './saved_models/'+ file_path.split('/')[-1].split('.')[0] + '_cls.pth')
