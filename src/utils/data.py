import random
import os
import math
import yaml
import copy
import numpy as np
from PIL import Image
from os import path
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms


class Generator():
    """
    Class to generate forest fire simulation data and process ensemble images.

    This class is responsible for generating raw simulation data as well as ensemble
    images from updated simulation frames. It uses a Simulator instance to perform
    the simulation and then processes the outputs for storage and visualization.

    Attributes:
        config_path (str): Path to the configuration YAML file.
        data_dir_path (str): Directory path where simulation images are saved.
        np_dir_path (str): Directory path where numpy arrays are saved.
        sample_size (int): Size (width and height) of the simulation sample.
        frame_size (int): Size of each output frame image.
        generations (int): Number of generations (simulation iterations).
        max_burned_area_percentage (float): Maximum allowed burned area percentage.
        simulator (Simulator): Simulator instance (initialized during simulation).
        transform: Transformation to apply to images (if any).
        info (bool): Flag to control additional informational output (e.g., plots).
    """
    def __init__(self,
                 config_path,
                 data_dir_path,
                 np_dir_path,
                 sample_size=128,
                 frame_size=64,
                 generations=61,
                 max_burned_area_percentage=0.3
                 ):
        """
        Initialize the Generator instance.

        Args:
            config_path (str): Path to the configuration file.
            data_dir_path (str): Directory path to save simulation images.
            np_dir_path (str): Directory path to save numpy arrays.
            sample_size (int, optional): Size of the simulation sample. Defaults to 128.
            frame_size (int, optional): Size of each frame image. Defaults to 64.
            generations (int, optional): Number of simulation generations. Defaults to 61.
            max_burned_area_percentage (float, optional): Maximum allowed burned area percentage. Defaults to 0.3.
        """
        self.config_path = config_path
        self.data_dir_path = data_dir_path
        self.np_dir_path = np_dir_path
        self.sample_size = sample_size
        self.frame_size = frame_size
        self.generations = generations
        self.max_burned_area_percentage = max_burned_area_percentage
        self.simulator = None
        self.transform = None
        self.info = False
        os.makedirs(self.np_dir_path, exist_ok=True)

    def __call__(self, type, *args, **kwargs):
        """
        Call the Generator to produce simulation data.

        Depending on the provided type ('raw' or 'ensemble'), it either generates
        raw simulation data or generates simulation updates and processes ensemble images.

        Args:
            type (str): Either 'raw' for raw simulation or 'ensemble' for ensemble updates.
            *args, **kwargs: Additional arguments passed to the respective generation method.

        Returns:
            int or list: The number of fires generated (for 'raw') or updates generated (for 'ensemble').
        """
        if type == 'raw':
            return self.generate(*args, **kwargs)
        elif type == 'ensemble':
            updates_generated = self.generate_updates(*args, **kwargs)
            self.process_and_save_ensemble_images(self.data_dir_path, transform=self.transform, info=self.info)
            return updates_generated

    def is_within_max_burned_area(self, burned_area):
        """
        Check whether the burned area in each simulation is within the allowed maximum.

        Args:
            burned_area (list): List of burned area values (one per generation).

        Returns:
            bool: True if all burned areas are within the allowed maximum, False otherwise.
        """
        max_allowed_area = self.sample_size * self.sample_size * self.max_burned_area_percentage
        return all(area <= max_allowed_area for area in burned_area)

    def generate(self, num_samples, fire_num, tolerance=10, interval=5, info=False):
        """
        Generate raw forest fire simulations.

        Args:
            num_samples (int): Number of simulation samples (fires) to generate.
            fire_num (int): Starting fire number identifier.
            tolerance (int, optional): Tolerance for non-increasing burned area before rejection. Defaults to 10.
            interval (int, optional): Interval for saving simulation frames. Defaults to 5.
            info (bool, optional): If True, print additional simulation info. Defaults to False.

        Returns:
            int: Total number of fires generated.
        """
        fires_generated = 0
        with tqdm(total=num_samples, colour="#6565b5", position=0, desc="Generating fires") as pbar:
            while fires_generated < num_samples:
                self.simulator = Simulator(self.config_path, self.sample_size)
                img_dir_path = self.data_dir_path + f"fire_{fire_num}/"

                burned_area, forest_states = self.simulator(generations=self.generations, np_dir_path=self.np_dir_path, fire_num=fire_num, info=False)

                if self.simulator.is_continuously_increasing(burned_area, tolerance) and self.is_within_max_burned_area(burned_area):
                    self.simulator.save_fire_states(self.simulator.forest, forest_states, interval=interval, size=self.frame_size, img_dir_path=img_dir_path, info=info)
                    fires_generated += 1
                    fire_num += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"Fire {fire_num} generated.")
                else:
                    pbar.set_postfix_str(f"Fire generation {fire_num} regenerating...")
        return fires_generated

    def generate_updates(self, num_samples, fire_num, num_updates, frames, interval=5, tolerance=10, info=False):
        """
        Generate updated states for selected simulation frames.

        Args:
            num_samples (int): Number of simulation samples.
            fire_num (int): Starting fire number identifier.
            num_updates (int): Number of updated states to generate per selected frame.
            frames (list): List of frame indices for which to generate updates.
            interval (int, optional): Interval (number of generations) between updates. Defaults to 5.
            tolerance (int, optional): Tolerance for non-increasing burned area. Defaults to 10.
            info (bool, optional): If True, print additional simulation info. Defaults to False.

        Returns:
            int: Total number of updates generated.
        """
        total_updates = num_samples * len(frames)
        updates_generated = 0
        frame_indices = frames

        with tqdm(total=total_updates, colour="#6565b5", position=0, desc="Generating updates") as pbar:
            while updates_generated < total_updates:
                self.simulator = Simulator(self.config_path, self.sample_size)
                img_dir_path = self.data_dir_path + f"fire_{fire_num}/"

                burned_area, forest_states = self.simulator(generations=self.generations, np_dir_path=self.np_dir_path, fire_num=fire_num, info=False)
                if self.simulator.is_continuously_increasing(burned_area, tolerance) and self.is_within_max_burned_area(burned_area):
                    fire_num += 1
                    for frame_index in frame_indices:
                        if frame_index < len(forest_states):
                            _ = self.simulator.generate_updated_states(forest_states, frame_index, num_updates, save_path=img_dir_path, size=self.frame_size, interval=interval)
                            updates_generated += 1
                            pbar.update(1)
                            pbar.set_postfix_str(f"Update {updates_generated} generated at frame {frame_index}.")

                            if updates_generated >= total_updates:
                                break
                        else:
                            pbar.set_postfix_str(f"Frame index {frame_index} out of range for fire {updates_generated}.")

                    if updates_generated >= total_updates:
                        break
                else:
                    pbar.set_postfix_str(f"Fire generation {updates_generated} regenerating...")
        return updates_generated

    def process_and_save_ensemble_images(self, root_dir, transform=None, info=False):
        """
        Process individual update images to create ensemble images.

        For each fire folder in the given directory, this function processes all updated images
        by averaging them to create an ensemble image. Optionally, the original, processed, and
        ensemble images can be displayed.

        Args:
            root_dir (str): Root directory containing fire simulation folders.
            transform: Transformation(s) to apply to the images. If None, a default tensor conversion is used.
            info (bool, optional): If True, display the images. Defaults to False.
        """
        for fire_folder in os.listdir(root_dir):
            fire_folder_path = os.path.join(root_dir, fire_folder)
            if os.path.isdir(fire_folder_path):
                for frame_folder in os.listdir(fire_folder_path):
                    frame_folder_path = os.path.join(fire_folder_path, frame_folder)
                    if os.path.isdir(frame_folder_path):
                        org_image_path = os.path.join(frame_folder_path, 'org.png')
                        update_dir = os.path.join(frame_folder_path, 'update')

                        processed_images = []
                        for img_name in os.listdir(update_dir):
                            img_path = os.path.join(update_dir, img_name)
                            image = Image.open(img_path).convert('L')
                            if transform is None:
                                transform = transforms.Compose([transforms.ToTensor()])
                            processed_image = transform(image)
                            processed_images.append(processed_image.numpy())

                        # Create ensemble image by averaging
                        ensemble_image = np.mean(processed_images, axis=0)

                        # Save the ensemble image
                        ensemble_image_path = os.path.join(frame_folder_path, 'ensemble.png')
                        ensemble_pil_image = transforms.ToPILImage()(torch.tensor(ensemble_image))
                        ensemble_pil_image.save(ensemble_image_path)

                        # Display the original, processed, and ensemble images
                        if info:
                            fig, axes = plt.subplots(1, len(processed_images) + 2, figsize=(20, 10))
                            axes[0].imshow(Image.open(org_image_path).convert('L'), cmap='gray')
                            axes[0].set_title('Original')

                            for i, img in enumerate(processed_images):
                                axes[i + 1].imshow(img[0], cmap='gray')
                                axes[i + 1].set_title(f'Processed {i+1}')

                            axes[-1].imshow(ensemble_image[0], cmap='gray')
                            axes[-1].set_title('Ensemble')

                            plt.show()


class Simulator():
    """
    Simulator for forest fire spread.

    This class loads environmental data (forest, ignition, altitude, density) and simulates
    the spread of a forest fire over a number of generations. It provides methods to update the
    forest state, calculate environmental influences (slope, wind), and save simulation results.
    """
    def __init__(self, config_path, sample_size=128):
        """
        Initialize the Simulator with configuration and simulation parameters.

        Args:
            config_path (str): Path to the configuration YAML file.
            sample_size (int, optional): Size of the simulation sample. Defaults to 128.
        """
        self.V = 5.  # need to find the true wind data
        # self.p_h = 0.58
        # self.a = 0.078
        # self.c_1 = 0.045
        # self.c_2 = 0.131
        self.p_h = random.uniform(0.20, 0.35)*1.
        self.a = random.uniform(0., 0.14)*1.
        self.c_1 = random.uniform(0., 0.12)*1.
        self.c_2 = random.uniform(0., 0.40)
        # custormize colorbar
        self.cmap = mpl.colors.ListedColormap(['orange', 'yellow', 'green', 'black'])
        self.cmap.set_over('0.25')
        self.cmap.set_under('0.75')
        self.bounds = [1.0, 2.02, 2.27, 3.5, 5.1]
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)
        # load data
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.data_paths = config['data_paths']
        # update the data paths
        for key in self.data_paths:
            if key != 'root':
                self.data_paths[key] = path.join(self.data_paths['root'], self.data_paths[key])
        self.sample_shape = (sample_size, sample_size)
        self.n_row = self.sample_shape[0]
        self.n_col = self.sample_shape[1]
        self.forest = Image.open(self.data_paths['forest'])
        self.ignition_ = np.loadtxt(self.data_paths['ignition'])
        self.altitude = Image.open(self.data_paths['altitude'])
        self.density = Image.open(self.data_paths['density'])
        self.forest = np.array(self.forest) / 255
        self.forest[self.forest < -999.] = 0.
        self.forest = self.forest/np.max(self.forest)
        self.altitude = np.array(self.altitude)/np.max(self.altitude)
        self.density = np.array(self.density)
        self.density = np.round(self.density/np.max(self.density))
        self.forest = resize(self.forest, self.sample_shape)
        self.altitude = resize(self.altitude, self.sample_shape)
        self.density = resize(self.density, self.sample_shape)
        self.density = np.round(self.density/np.max(self.density))
        self.ignition = self.random_ignition(np.array(self.forest).shape[0], np.array(self.forest).shape[1])
        self.fields_1_sim = np.zeros((1, 100))
        self.vegetation_matrix = self.forest
        self.density_matrix = self.density.tolist()
        self.altitude_matrix = self.altitude.tolist()
        self.wind_matrix = self.get_wind()
        self.new_forest = self.ignition.tolist()
        self.slope_matrix = self.get_slope(self.altitude_matrix)
        self.forest_states = []

    def __call__(self, *args, **kwargs):
        """
        Make the Simulator callable.

        This allows the instance to be called directly to run the simulation.

        Returns:
            tuple: Burned area (list) and forest states (list) from the simulation.
        """
        # call the run method
        return self.run(*args, **kwargs)

    def run(self, generations, np_dir_path, fire_num, info=False):
        """
        Run the forest fire simulation.

        Args:
            generations (int): Number of generations (iterations) to simulate.
            np_dir_path (str): Directory path to save numpy arrays of forest states.
            fire_num (int): Identifier for the simulation run.
            info (bool, optional): If True, print simulation details. Defaults to False.

        Returns:
            tuple: A tuple containing the list of burned area values and the list of forest states.
        """
        burned_area, forest_states = self.simulate_forest_fire(self.update_forest, self.new_forest, generations=generations, np_dir_path=np_dir_path, fire_num=fire_num, info=info)
        return burned_area, forest_states

    def is_continuously_increasing(self, values, tolerance=10):
        """
        Determine whether the provided values are continuously increasing (within tolerance).

        Args:
            values (list): List of numerical values (e.g., burned area per generation).
            tolerance (int, optional): Maximum allowed consecutive non-increasing values. Defaults to 10.

        Returns:
            bool: True if the sequence is continuously increasing within the tolerance, otherwise False.
        """
        non_growth_count = 0
        # Check if the values are continuously increasing
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                non_growth_count = 0  # Reset the count if there's growth
            else:
                non_growth_count += 1  # Increment the count if there's no growth

            if non_growth_count > tolerance:
                return False

        return True

    def to_image(self, img_array, size=64):
        """
        Convert a boolean array to a grayscale PIL Image and resize it.

        Args:
            img_array (numpy.ndarray): Boolean array representing image data.
            size (int, optional): Target size (width and height) of the output image. Defaults to 64.

        Returns:
            PIL.Image: Resized grayscale image.
        """
        # Convert boolean array to uint8
        img_array = img_array.astype(np.uint8) * 255
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        # Resize the image
        img = img.resize((size, size), Image.LANCZOS)
        return img

    def plot_fire_states(self, initial_forest, forest_array, index):
        """
        Plot the original forest, the binary forest state, and a combined image.

        Args:
            initial_forest (numpy.ndarray): Original forest image.
            forest_array (numpy.ndarray): Forest state array at a specific iteration.
            index (int): Generation index for labeling.
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(initial_forest)
        plt.title("Original Forest")
        plt.subplot(1, 3, 2)
        plt.imshow(forest_array >= 3, cmap='gray', interpolation="none")
        plt.title(f"Forest Array at iteration {index}")
        plt.subplot(1, 3, 3)
        combined_forest = initial_forest + forest_array
        plt.imshow(combined_forest, cmap=self.cmap, norm=self.norm, interpolation="none")
        plt.title(f"Combined Forest at iteration {index}")
        plt.show()
        plt.close()

    def save_fire_states(self, initial_forest, forest_states, interval=5, size=64, img_dir_path='./', info=False):
        """
        Save simulation forest states as images at specified intervals.

        Args:
            initial_forest (numpy.ndarray): Original forest image.
            forest_states (list): List of forest states from the simulation.
            interval (int, optional): Interval between saved frames. Defaults to 5.
            size (int, optional): Size of the output images. Defaults to 64.
            img_dir_path (str, optional): Directory to save the images. Defaults to './'.
            info (bool, optional): If True, display the fire states using plots. Defaults to False.
        """
        # Create the directory if it doesn't exist
        os.makedirs(img_dir_path, exist_ok=True)
        label = 0
        for i, forest_array in enumerate(forest_states):
            if i >= 10 and i % interval == 0:
                # plot the forest states
                if info:
                    self.plot_fire_states(initial_forest, forest_array, i)
                img = self.to_image(forest_array >= 3, size)  # Convert to PIL Image and resize
                img = img.point(lambda p: 255 if p > 0 else 0)
                img.save(f"{img_dir_path}frame_{label}.png")

                label += 1

    def simulate_forest_fire(self, update_forest, initial_forest, generations=61, np_dir_path='./', fire_num=0, info=False):
        """
        Simulate the spread of a forest fire over a specified number of generations.

        Args:
            update_forest (function): Function to update the forest state.
            initial_forest (list or numpy.ndarray): Initial forest state.
            generations (int, optional): Number of simulation iterations. Defaults to 61.
            np_dir_path (str, optional): Directory to save numpy arrays. Defaults to './'.
            fire_num (int, optional): Identifier for the simulation run. Defaults to 0.
            info (bool, optional): If True, print simulation information. Defaults to False.

        Returns:
            tuple: A tuple containing a list of burned area values and a list of forest states.
        """
        burned_area = []
        self.forest_states = []
        new_forest = copy.deepcopy(initial_forest)
        # plt.imshow(new_forest)
        for i in range(generations):
            new_forest = copy.deepcopy(update_forest(new_forest))
            forest_array = np.array(new_forest)
            burned_area.append(np.sum(forest_array >= 3))
            if info:
                print(f"Generation {i}: Burned Area = {burned_area[-1]}")
            self.forest_states.append(forest_array)
        np.save(f"{np_dir_path}forest_{fire_num}.npy", np.array(self.forest_states))
        return burned_area, self.forest_states

    def generate_updated_states(self, forest_states, frame_index, num_updates, save_path, size, interval):
        """
        Generate and save updated simulation states for a selected frame.

        Args:
            forest_states (list): List of forest states from the simulation.
            frame_index (int): Index of the frame to update.
            num_updates (int): Number of updated states to generate.
            save_path (str): Directory to save updated images.
            size (int): Size of the output images.
            interval (int): Number of generations between updates.

        Returns:
            list: A list containing the updated forest states.
        """
        selected_frame = forest_states[frame_index]
        updated_states = []
        save_path = os.path.join(save_path, f"{frame_index}")
        os.makedirs(save_path, exist_ok=True)
        save_path_update = os.path.join(save_path, "update")
        os.makedirs(save_path_update, exist_ok=True)
        org_img_array = np.array(selected_frame) >= 3
        org_img = self.to_image(org_img_array, size=size)
        org_img.save(os.path.join(save_path, "org.png"))
        # Generate the updated states
        for i in range(num_updates):
            new_forest = copy.deepcopy(selected_frame)
            for _ in range(interval):
                new_forest = np.array(self.update_forest(new_forest.tolist()))
            updated_states.append(new_forest)
            # Save each updated state as an image
            img_array = new_forest >= 3
            img = self.to_image(img_array, size=size)
            img = img.point(lambda p: 255 if p > 0 else 0)
            img.save(os.path.join(save_path_update, f"{i+1}.png"))

        return updated_states

    # helper functions
    def random_ignition(self, dim_x, dim_y):
        """
        Generate a random ignition state for the forest.

        Args:
            dim_x (int): Number of rows in the forest.
            dim_y (int): Number of columns in the forest.

        Returns:
            numpy.ndarray: A 2D array with ignition points.
        """
        field = np.ones((dim_x, dim_y)) * 2
        x = random.randint(30, dim_x - 30)
        y = random.randint(30, dim_y - 30)
        for i in range(x, x + 4):
            for j in range(y, y + 4):
                field[i, j] = 3
        return field

    def centre_ignition(self, dim_x, dim_y):
        """
        Generate a centrally located ignition state.

        Args:
            dim_x (int): Number of rows in the forest.
            dim_y (int): Number of columns in the forest.

        Returns:
            numpy.ndarray: A 2D array with a central ignition point.
        """
        field = np.ones((dim_x, dim_y)) * 2
        x = round(dim_x / 2)
        y = round(dim_y / 2)
        for i in range(x, x + 3):
            for j in range(y, y + 3):
                field[i, j] = 3
        return field

    def tg(self, x):
        """
        Compute the angle in degrees using the arctan of the input.

        Args:
            x (float): Input value.

        Returns:
            float: Angle in degrees.
        """
        return math.degrees(math.atan(x))

    def get_slope(self, altitude_matrix):
        """
        Calculate the slope for each cell in the altitude matrix based on its 3x3 neighborhood.

        Args:
            altitude_matrix (list of lists): 2D matrix of altitude values.

        Returns:
            list: A 2D matrix where each element is a 3x3 matrix of slope values.
        """
        slope_matrix = [[0 for col in range(self.n_col)] for row in range(self.n_row)]
        for row in range(self.n_row):
            for col in range(self.n_col):
                sub_slope_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if row == 0 or row == self.n_row-1 or col == 0 or col == self.n_col-1:  # margin is flat
                    slope_matrix[row][col] = sub_slope_matrix
                    continue
                current_altitude = altitude_matrix[row][col]
                sub_slope_matrix[0][0] = self.tg((current_altitude - altitude_matrix[row-1][col-1])/1.414)
                sub_slope_matrix[0][1] = self.tg(current_altitude - altitude_matrix[row-1][col])
                sub_slope_matrix[0][2] = self.tg((current_altitude - altitude_matrix[row-1][col+1])/1.414)
                sub_slope_matrix[1][0] = self.tg(current_altitude - altitude_matrix[row][col-1])
                sub_slope_matrix[1][1] = 0
                sub_slope_matrix[1][2] = self.tg(current_altitude - altitude_matrix[row][col+1])
                sub_slope_matrix[2][0] = self.tg((current_altitude - altitude_matrix[row+1][col-1])/1.414)
                sub_slope_matrix[2][1] = self.tg(current_altitude - altitude_matrix[row+1][col])
                sub_slope_matrix[2][2] = self.tg((current_altitude - altitude_matrix[row+1][col+1])/1.414)
                slope_matrix[row][col] = sub_slope_matrix
        return slope_matrix

    def calc_pw(self, theta, c_1, c_2, V):
        """
        Calculate the wind influence factor based on the wind angle.

        Args:
            theta (float): Wind angle in degrees.
            c_1 (float): Constant parameter.
            c_2 (float): Constant parameter.
            V (float): Wind speed.

        Returns:
            float: Wind influence factor.
        """
        t = math.radians(theta)
        ft = math.exp(V*c_2*(math.cos(t)-1))
        return math.exp(c_1*V)*ft

    def get_wind(self):
        """
        Generate a 3x3 wind matrix based on predefined wind angles.

        Returns:
            list: A 3x3 matrix containing wind influence values.
        """
        wind_matrix = [[0 for col in [0, 1, 2]] for row in [0, 1, 2]]
        # thetas = [[0, 180, 180], #need to define the exact angle
        #          [180, 0, 180],
        #          [180, 180, 0]]
        thetas = [[180, 180, 180],
                  [180, 0, 180],
                  [180, 180, 180]]

        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                wind_matrix[row][col] = self.calc_pw(thetas[row][col], self.c_1, self.c_2, self.V)
        wind_matrix[1][1] = 0
        return wind_matrix

    def burn_or_not_burn(self, abs_row, abs_col, neighbour_matrix, p_h, a):
        """
        Decide whether a cell should ignite based on neighbor states and environmental factors.

        Args:
            abs_row (int): Absolute row index of the cell.
            abs_col (int): Absolute column index of the cell.
            neighbour_matrix (list): 3x3 matrix representing neighbor cell states.
            p_h (float): Base probability factor.
            a (float): Slope influence factor.

        Returns:
            int: Returns 3 if the cell ignites (burns), otherwise 2.
        """
        p_veg = self.vegetation_matrix[abs_row][abs_col]
        p_den = {0: -0.4, 1: 0, 2: 0.3}[self.density_matrix[abs_row][abs_col]]
        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                if neighbour_matrix[row][col] == 3:  # we only care there is a neighbour that is burning
                    # print(row,col)
                    slope = self.slope_matrix[abs_row][abs_col][row][col]
                    p_slope = math.exp(a * slope)
                    p_wind = self.wind_matrix[row][col]
                    p_burn = p_h * (0.5 + p_veg * 10.) * (1 + p_den) * p_wind * p_slope
                    if p_burn > random.random():
                        return 3  # start burning
        return 2  # not burning

    def update_forest(self, old_forest):
        """
        Update the forest state for one simulation step.

        The function updates each cell based on its current state and the states of its neighbors.

        Args:
            old_forest (list): 2D list representing the current forest state.

        Returns:
            list: Updated 2D forest state.
        """
        result_forest = [[1 for i in range(self.n_col)] for j in range(self.n_row)]
        for row in range(1, self.n_row-1):
            for col in range(1, self.n_col-1):

                if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                    result_forest[row][col] = old_forest[row][col]  # no fuel or burnt down
                if old_forest[row][col] == 3:
                    if random.random() < 0.4:
                        result_forest[row][col] = 3  # TODO need to change back here
                    else:
                        result_forest[row][col] = 4
                if old_forest[row][col] == 2:
                    neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]for row_vec in old_forest[row-1:row+2]]
                    # print(neighbours)
                    result_forest[row][col] = self.burn_or_not_burn(row, col, neighbours, self.p_h, self.a)
        return result_forest
