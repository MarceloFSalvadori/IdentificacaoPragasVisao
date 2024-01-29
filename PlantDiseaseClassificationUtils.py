
from torchvision.utils import save_image

#******************************************************************************#
#******************************************************************************#
def showImagesFolder(dataset,iterations:int):
    
    img_num = 0

    for _ in range(iterations):
        for img, label in dataset:
            save_image(img,'img'+str(img_num)+'.png')
            img_num += 1
#******************************************************************************#
#******************************************************************************#
