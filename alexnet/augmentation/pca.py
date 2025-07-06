import torch
from tqdm import tqdm

class ImageNetPCA:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader

        self._running_sum = torch.zeros(3)
        self._pixel_count = 0
        self._running_outer_product = torch.zeros((3, 3))

        self.eigenvalues = torch.zeros(3)
        self.eigenvectors = torch.zeros((3, 3))
    
    def fit(self) -> tuple[torch.Tensor, torch.Tensor]:
        with tqdm(total=len(self.dataloader.dataset), desc="Fitting PCA") as pbar:
            for batch in self.dataloader:
                image: torch.Tensor = batch['pixel_values'] # (batch_size, 3, height, width)
                
                self._running_sum += image.sum(dim=(0, 2, 3))
                self._pixel_count += image.numel() // 3 # Assuming 3 channels

                flattened_channels = image.transpose(1, 0).reshape(3, -1) # reshape to (3, batch_size * height * width)

                outer_product = torch.mm(flattened_channels, flattened_channels.t())
                self._running_outer_product += outer_product

                pbar.update(image.size(0))

        mean_vector = self._running_sum / self._pixel_count
        mean_outer_product = self._running_outer_product / self._pixel_count
        covariance_matrix = mean_outer_product - torch.outer(mean_vector, mean_vector)

        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(covariance_matrix)

        return self.eigenvalues, self.eigenvectors
