
# import some packages you need here
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir,augment=False):
        """
        Args:
            data_dir (string): Directory with all the images.
        """
        self.root_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        
        # 이미지 전처리 설정
        if augment==False:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),  # 데이터를 0에서 1사이로 변환해줌.
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터의 평균과 표준편차
            ])
        else:
            self.transform = transforms.Compose([   # agument 추가 하지만 성능이 낮아져서 사용 안함.
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = io.imread(img_name)  # 이미지 로드
        image = image[:, :, 0] if image.ndim == 3 else image  # RGB 이미지라면 그레이스케일로 변환
        
        # 파일 이름에서 라벨 추출
        label = int(self.file_names[idx].split('_')[1].split('.')[0])
        
        # 이미지 전처리
        image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    data_dir ='C://Users//Islab//Downloads//mnist-classification//mnist-classification//data//train'  # 실제 이미지가 저장된 폴더 경로
    mnist_dataset = MNIST(data_dir, augment=False)
    print(f"Dataset Length without augmentation: {len(mnist_dataset)}")
    image, label = mnist_dataset[0]
    print(f"First Image Shape without augmentation: {image.shape}, Label: {label}")

    # 데이터 증강을 적용한 데이터셋
    mnist_augmented = MNIST(data_dir, augment=True)
    print(f"Dataset Length with augmentation: {len(mnist_augmented)}")
    image_aug, label_aug = mnist_augmented[0]
    print(f"First Image Shape with augmentation: {image_aug.shape}, Label: {label_aug}")

    # 이미지 출력 예시
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()
    plt.imshow(image_aug.squeeze(), cmap='gray')
    plt.title(f"Label: {label_aug}")
    plt.show()


