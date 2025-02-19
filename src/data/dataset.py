from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class FireDataset(Dataset):
    """
    A PyTorch dataset for loading fire frames from a sequence, where each sample contains two consecutive frames
    from a fire sequence: frame_n and frame_n+1.

    Args:
        root_dir (str): The root directory where fire frames are stored.
        indices (list of int): A list of indices corresponding to subfolders (fire sequences) in the root directory.
        num_frames (int, optional): The number of frames to sample for each sequence (default is 10).
        transform (callable, optional): A function/transform to apply to each image (default is None).

    Attributes:
        root_dir (str): The root directory where fire frames are stored.
        indices (list of int): The indices for the fire sequences to load.
        num_frames (int): The number of frames in each sequence to load.
        transform (callable): The optional transform function to apply to each image.

    Methods:
        __len__(): Returns the total number of samples in the dataset (num_samples * num_frames).
        __getitem__(idx): Retrieves a tuple (frame_n, frame_n+1) for a given index.
    """

    def __init__(self, root_dir, indices, num_frames=10, transform=None):
        self.root_dir = root_dir
        self.indices = indices
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset, which is the product of
        the number of indices and the number of frames per sequence.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.indices) * self.num_frames  # Each sample has num_frames frames

    def __getitem__(self, idx):
        """
        Retrieves the frames corresponding to a given index `idx` and returns the pair (frame_n, frame_n+1).

        Args:
            idx (int): The index for the desired sample. This will be split into the sample index and frame index.

        Returns:
            tuple: A tuple containing two torch.Tensor objects:
                - frame_n (Tensor): The current frame (frame_n) from the sequence.
                - frame_n_plus_1 (Tensor): The next frame (frame_n+1) from the sequence.
        """
        sample_idx = idx // self.num_frames
        frame_idx = idx % self.num_frames

        frame_n_path = os.path.join(self.root_dir, f'fire_{self.indices[sample_idx]}', f'frame_{frame_idx}.png')
        frame_n_plus_1_path = os.path.join(self.root_dir, f'fire_{self.indices[sample_idx]}', f'frame_{frame_idx + 1}.png')

        frame_n = Image.open(frame_n_path).convert('L')
        frame_n_plus_1 = Image.open(frame_n_plus_1_path).convert('L')

        if self.transform:
            frame_n = self.transform(frame_n)
            frame_n_plus_1 = self.transform(frame_n_plus_1)

        return frame_n, frame_n_plus_1


class EnsembleFireDataset(Dataset):
    """
    A PyTorch dataset for loading images related to fire ensembles, including original frames, ensemble frames, and
    update frames, from multiple fire sequences.

    Args:
        root_dir (str): The root directory where fire data is stored.
        indices (list of int): A list of indices corresponding to fire sequence subfolders.
        transform (callable, optional): A function/transform to apply to each image (default is None).

    Attributes:
        root_dir (str): The root directory where fire data is stored.
        indices (list of int): The indices of the fire sequences to load.
        transform (callable): An optional transform to apply to each image.
        data (list of tuples): A list of tuples containing paths to the original, ensemble, and update images.

    Methods:
        _load_dataset(): Loads the dataset by iterating over the indices and collecting the image paths.
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves a sample consisting of the original image, ensemble image, and update images for the given index.
    """

    def __init__(self, root_dir, indices, transform=None):
        self.root_dir = root_dir
        self.indices = indices
        self.transform = transform
        self.data = self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset by iterating over the fire sequence directories and collecting image paths.

        Returns:
            list of tuples: A list of tuples where each tuple contains:
                - org_image_path (str): Path to the original image.
                - ensemble_image_path (str): Path to the ensemble image.
                - update_images_paths (list of str): List of paths to the update images.
        """
        data = []
        for idx in self.indices:
            fire_folder_path = os.path.join(self.root_dir, f'fire_{idx}')
            if os.path.isdir(fire_folder_path):
                for frame_folder in os.listdir(fire_folder_path):
                    frame_folder_path = os.path.join(fire_folder_path, frame_folder)
                    if os.path.isdir(frame_folder_path):
                        org_image_path = os.path.join(frame_folder_path, 'org.png')
                        ensemble_image_path = os.path.join(frame_folder_path, 'ensemble.png')
                        update_images = []
                        update_folder_path = os.path.join(frame_folder_path, 'update')
                        if os.path.exists(update_folder_path) and os.path.isdir(update_folder_path):
                            for update_image in os.listdir(update_folder_path):
                                update_image_path = os.path.join(update_folder_path, update_image)
                                if os.path.exists(update_image_path):
                                    update_images.append(update_image_path)
                        if os.path.exists(org_image_path) and os.path.exists(ensemble_image_path):
                            data.append((org_image_path, ensemble_image_path, update_images))
        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample consisting of the original image, ensemble image, and a list of update images for the given index.

        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            tuple: A tuple containing:
                - org_image (Tensor): The original image.
                - ensemble_image (Tensor): The ensemble image.
                - update_images (list of Tensor): A list of update images.
        """
        org_image_path, ensemble_image_path, update_images_paths = self.data[idx]

        org_image = Image.open(org_image_path).convert('L')
        ensemble_image = Image.open(ensemble_image_path).convert('L')
        update_images = [Image.open(img_path).convert('L') for img_path in update_images_paths]

        if self.transform:
            org_image = self.transform(org_image)
            ensemble_image = transforms.ToTensor()(ensemble_image)
            update_images = [self.transform(img) for img in update_images]

        return org_image, ensemble_image, update_images
