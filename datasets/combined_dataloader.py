from torch.utils.data import DataLoader, Dataset

class CombinedDataLoader():
    def __init__(self, data_loaders):
        """
        Initializes the CombinedDataLoader with multiple DataLoaders.
        Args:
            *data_loaders: Any number of DataLoaders to combine.
        """
        self.data_loaders = data_loaders

    def __iter__(self):
        """
        Combines the iterators of all DataLoaders sequentially.
        """
        for data_loader in self.data_loaders:
            for data in data_loader:
                yield data

    def __len__(self):
        """
        Calculates the total length by summing up the lengths of all DataLoaders.
        """
        return sum([len(data_loader) for data_loader in self.data_loaders])



if __name__=='__main__':
    # Example usage:
    class ExampleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]



    # Create example datasets and dataloaders
    dataset1 = ExampleDataset([1, 2, 3, 4])
    dataset2 = ExampleDataset([5, 6, 7, 8])

    data_loader1 = DataLoader(dataset1, batch_size=2, shuffle=False)
    data_loader2 = DataLoader(dataset2, batch_size=2, shuffle=False)

    # Combine DataLoaders
    combined_loader = CombinedDataLoader([data_loader1, data_loader2])
    
    print(len(combined_loader))

    # Iterate through the combined DataLoader
    for batch in combined_loader:
        print(batch)

    # Output:
    # [1, 2]
    # [3, 4]
    # [5, 6]
    # [7, 8]