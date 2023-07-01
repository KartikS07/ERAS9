from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, image_dataset, transform=None):
        self.image_dataset = image_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label
