from torchvision import datasets
from tqdm import tqdm

path = 'E:/CodeforFuture/Data/MSTAR_SOC/'

def write_txt(path, mode='train'):

    dataset = datasets.ImageFolder(path + mode)

    tq = tqdm(dataset.imgs)
    for img_path, labels in tq:
        with open(path + mode +'.txt', 'a') as f:
            f.write(str(img_path) + ',' + str(labels) + '\n')


if __name__ == '__main__':
    path = 'E:/CodeforFuture/Data/MSTAR_SOC/'

    # write_txt(path, 'train')
    write_txt(path, 'test')

    print("write finished !")