"""
Synthetic MNIST Digit Generator
Generates simple geometric digit images for training
"""

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

class SyntheticMNIST:
    """Generate synthetic MNIST-like digits using geometric shapes"""
    
    def __init__(self, num_samples=50000, img_size=28):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = []
        self.labels = []
        
    def generate_digit_0(self):
        """Generate digit 0 as ellipse/circle"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random variations
        thickness = np.random.randint(2, 5)
        center_x = self.img_size // 2 + np.random.randint(-2, 3)
        center_y = self.img_size // 2 + np.random.randint(-2, 3)
        radius_x = np.random.randint(8, 12)
        radius_y = np.random.randint(8, 12)
        
        # Draw outer ellipse
        draw.ellipse([center_x-radius_x, center_y-radius_y, 
                     center_x+radius_x, center_y+radius_y], 
                     outline=255, width=thickness)
        
        return img
    
    def generate_digit_1(self):
        """Generate digit 1 as vertical line"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random variations
        thickness = np.random.randint(2, 5)
        x_pos = self.img_size // 2 + np.random.randint(-3, 4)
        y_start = np.random.randint(3, 6)
        y_end = self.img_size - np.random.randint(3, 6)
        
        # Draw vertical line
        draw.line([x_pos, y_start, x_pos, y_end], fill=255, width=thickness)
        
        # Optional top stroke
        if np.random.random() > 0.5:
            draw.line([x_pos-2, y_start+2, x_pos, y_start], fill=255, width=2)
        
        return img
    
    def generate_digit_2(self):
        """Generate digit 2 with curves and lines"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top arc
        draw.arc([8, 4, 20, 14], start=180, end=0, fill=255, width=thickness)
        # Diagonal
        draw.line([20, 10, 8, 22], fill=255, width=thickness)
        # Bottom line
        draw.line([8, 22, 20, 22], fill=255, width=thickness)
        
        return img
    
    def generate_digit_3(self):
        """Generate digit 3 with two curves"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top arc
        draw.arc([10, 4, 22, 13], start=180, end=0, fill=255, width=thickness)
        # Bottom arc
        draw.arc([10, 15, 22, 24], start=180, end=0, fill=255, width=thickness)
        
        return img
    
    def generate_digit_4(self):
        """Generate digit 4 with vertical and horizontal lines"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Vertical line (right)
        draw.line([18, 4, 18, 24], fill=255, width=thickness)
        # Diagonal (left)
        draw.line([10, 4, 10, 16], fill=255, width=thickness)
        # Horizontal
        draw.line([10, 16, 22, 16], fill=255, width=thickness)
        
        return img
    
    def generate_digit_5(self):
        """Generate digit 5"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top line
        draw.line([10, 4, 20, 4], fill=255, width=thickness)
        # Vertical
        draw.line([10, 4, 10, 14], fill=255, width=thickness)
        # Bottom arc
        draw.arc([10, 14, 22, 24], start=180, end=0, fill=255, width=thickness)
        
        return img
    
    def generate_digit_6(self):
        """Generate digit 6 as circle with tail"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Bottom circle
        draw.ellipse([10, 14, 22, 24], outline=255, width=thickness)
        # Top curve
        draw.arc([10, 4, 18, 16], start=90, end=270, fill=255, width=thickness)
        
        return img
    
    def generate_digit_7(self):
        """Generate digit 7 as horizontal line and diagonal"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top line
        draw.line([8, 4, 20, 4], fill=255, width=thickness)
        # Diagonal
        draw.line([20, 4, 10, 24], fill=255, width=thickness)
        
        return img
    
    def generate_digit_8(self):
        """Generate digit 8 as two stacked circles"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top circle
        draw.ellipse([10, 4, 20, 12], outline=255, width=thickness)
        # Bottom circle
        draw.ellipse([10, 16, 20, 24], outline=255, width=thickness)
        
        return img
    
    def generate_digit_9(self):
        """Generate digit 9 as circle with tail"""
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        
        thickness = np.random.randint(2, 4)
        
        # Top circle
        draw.ellipse([10, 4, 22, 14], outline=255, width=thickness)
        # Bottom tail
        draw.arc([14, 12, 22, 24], start=270, end=90, fill=255, width=thickness)
        
        return img
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        print(f"Generating {self.num_samples} synthetic digits...")
        
        generators = [
            self.generate_digit_0,
            self.generate_digit_1,
            self.generate_digit_2,
            self.generate_digit_3,
            self.generate_digit_4,
            self.generate_digit_5,
            self.generate_digit_6,
            self.generate_digit_7,
            self.generate_digit_8,
            self.generate_digit_9
        ]
        
        for i in range(self.num_samples):
            # Random digit
            digit = np.random.randint(0, 10)
            img = generators[digit]()
            
            # Add random rotation
            angle = np.random.randint(-15, 16)
            img = img.rotate(angle, fillcolor=0)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            self.data.append(img_array)
            self.labels.append(digit)
            
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1}/{self.num_samples} images")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print("Dataset generation complete!")
        return self.data, self.labels
    
    def save_dataset(self, filepath='synthetic_mnist.npz'):
        """Save generated dataset"""
        np.savez_compressed(filepath, data=self.data, labels=self.labels)
        print(f"Dataset saved to {filepath}")
    
    @staticmethod
    def load_dataset(filepath='synthetic_mnist.npz'):
        """Load previously generated dataset"""
        data = np.load(filepath)
        return data['data'], data['labels']


class SyntheticMNISTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for synthetic MNIST"""
    
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


if __name__ == "__main__":
    # Test generation
    generator = SyntheticMNIST(num_samples=1000)
    data, labels = generator.generate_dataset()
    generator.save_dataset('test_synthetic.npz')
    print(f"Generated data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
