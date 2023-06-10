import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations des données
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Téléchargement de l'ensemble de données MNIST
train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

# Création des dataloaders
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Définition du modèle
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Entraînement du modèle
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Aplanir les images MNIST en un vecteur de 784 longueurs
        images = images.view(images.shape[0], -1)
    
        # Passer les données à travers le modèle
        output = model(images)
        # Calcul de la perte
        loss = criterion(output, labels)
        
        # Mise à zéro des gradients
        optimizer.zero_grad()
        # Effectuer une étape de backpropagation et d'optimisation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# Évaluation du modèle
correct_count, all_count = 0, 0
for images,labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
