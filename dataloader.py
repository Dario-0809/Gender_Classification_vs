from library import *
from convert_pic import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def make_datapath_list(phase="train"):
    rootpath = "./data/gender-classification-dataset/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list

train_list = make_datapath_list("train")
val_list = make_datapath_list("val")

# print(train_list[1])
# print(train_list[47000])
# print(val_list[1])
# print(val_list[11000])
batch_size = 25

# MAKE DATASET
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        super().__init__()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[43:49]
            if label == "female":
                label = 0
            else:
                label  = img_path[43:47]
                label = 1

        elif self.phase == "val":
            label = img_path[41:47]
            if label == "female":
                label = 0
            else:
                label  = img_path[41:45]
                label = 1

        return img_transformed, label

train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")


# DATALOADER
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)

dataloader_dict = {"train": train_dataloader, "val": val_dataloader}