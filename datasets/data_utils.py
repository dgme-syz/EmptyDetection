from PIL import Image
from os import listdir
from os.path import join
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader

# 读取图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in 
               ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# 定义数据集处理以及获取函数
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # 转换为Tensor格式
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    return transform(img)

class DatasetFromFolder(Dataset):
    def __init__(self, dir_):
        super(DatasetFromFolder, self).__init__()
        labels = [0, 1]
        self.img_filenames = []
        for x in labels:
            for img in listdir(join(dir_, str(x))):
                if is_image_file(img):
                    self.img_filenames.append((join(dir_, str(x), img), x))
    def __getitem__(self, index):
        img, label = self.img_filenames[index]
        return preprocess(Image.open(img)), label 
    def __len__(self):
        return len(self.img_filenames)
    
if __name__ == '__main__':
    BaseFold = Path(__file__).parent.absolute().__str__()
    dataset = DatasetFromFolder(join(BaseFold, "train_paking"))
    train_data, test_data = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    for batch in train_loader:
        imgs, labels = batch
        for img, label in zip(imgs, labels):
            img_pil = transforms.ToPILImage()(img)  # 将张量转换为PIL图像对象
            img_pil.show()  # 显示图像
            print(label)
            break
        break


