################### - Imports - #############################################################
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

from .datasets import pick_the_dataset


##################################
############################################################################################


################### - Helper Functions - #############################################################
def create_dataset_splits(dataset):
    # Shuffle the dataset
    # dataset = dataset.shuffle()

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.25, random_state=42)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset

##################################
############################################################################################


################### - Main Execution - #############################################################
def create_dataloaders(args):
    """Create required dataloaders for the given dataset
        Args:
            args: for picking the dataset: args.dataset_name, args.preprocess, args.do_analysis
        Returns:
            train_loader, val_loader, test_loader
    """

    dataset, metadata = pick_the_dataset(args)
    train_dataset, val_dataset, test_dataset = create_dataset_splits(dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, metadata
