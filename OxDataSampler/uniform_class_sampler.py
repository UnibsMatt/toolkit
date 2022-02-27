import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class UniformClassSampler(Sampler):
    def __init__(self, labels: np.ndarray, batch_size=32, iterations: int = 100, classes_proportion: tuple = None):
        """
        Args:
            labels: numpy array of the labels of the class of the dataset
            batch_size: batch size
            iterations: number of iteration per epoch
            classes_proportion: proportions of samples between classes es. 3 classes -> [ 20, 50, 30]
        """
        super().__init__(labels)
        # batch size
        self.batch_size = batch_size
        # iterations to do at each epoch
        self.iterations = iterations
        # labels of all the samples in dataset
        self.labels = labels
        # labels of the unique classes
        self.classes, self.samples_per_class = np.unique(labels, return_counts=True)
        # number of different classes
        self.n_classes = len(self.classes)
        # proportions in classes
        self.classes_proportion = self.check_proportion(classes_proportion)
        # samples per class after proportion
        self.proportioned_samples_per_class = np.round(self.samples_per_class * self.classes_proportion).astype(int)
        # number of samples in the batch based on proportion
        self.proportioned_samples_per_batch = np.round(self.batch_size * self.classes_proportion).astype(int)
        # tensor of index of dimension [n_classes, max(len(samples))]
        self.index_tensor = self.create_index_tensor()

    def __iter__(self):
        for iteration in range(self.iterations):

            batch = torch.LongTensor()
            for class_index, index_tensor_class in enumerate(self.index_tensor):
                # removing samples equals Nan
                index_tensor_class = index_tensor_class[~torch.isnan(index_tensor_class)]
                # random permutation of samples in the same class
                permutation = torch.randperm(len(index_tensor_class))[
                    : self.proportioned_samples_per_batch[class_index]
                ]
                # concatenation
                batch = torch.cat((batch, index_tensor_class[permutation].type(torch.LongTensor)))
            # shuffle along all samples between classes
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations

    def check_proportion(self, classes_proportion) -> tuple:
        if classes_proportion is None:
            prop = int(100 / self.n_classes)
            # equals division between classes
            classes_proportion = [prop for i in range(self.n_classes)]

        else:
            assert len(classes_proportion) == self.n_classes, "The proportions must the same number of classes "
            assert sum(classes_proportion) == 100, "The sum of proportion must be 1"

        return np.array(classes_proportion) / 100

    def create_index_tensor(self):
        index_per_class = [np.where(self.labels == cl)[0] for cl in self.classes]
        # creation of a matrix [n_classes, max(samples per class)] fill with nan
        index_tensor = torch.Tensor([float("NaN")]).repeat(self.n_classes, max(self.samples_per_class))
        for class_ind in range(self.n_classes):
            # from numpy to Longtensor
            row_tensor = torch.LongTensor(index_per_class[class_ind])
            # copy of the values
            index_tensor[class_ind, : len(row_tensor)] = row_tensor
        return index_tensor

    def sampler_info(self):
        """
        Print the infos about the sampler
        """
        info = f"Batch size: {self.batch_size}\n"
        info += f"Nuber of classes: {self.n_classes}\n"
        info += f"Samples per class: {self.samples_per_class}\n"
        info += f"Proportion: {self.classes_proportion}\n"
        info += f"Sampler proportioned per class: {self.proportioned_samples_per_class}\n"
        info += f"Sampler proportioned per batch: {self.proportioned_samples_per_batch}\n"

        print(info)
