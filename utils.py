import numpy as np
import pylab as plt
from PIL import Image
from torchvision import transforms


def get_image_shape(path):
    image = Image.open(path)
    return image.shape


def get_image(path, h, w):
    m_trans = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    image = Image.open(path)
    image = m_trans(image)
    image = image.unsqueeze(0)
    return image


def show_image(img):
    img = img.squeeze(0)
    pil = transforms.ToPILImage()
    img = pil(img)
    plt.imshow(img)
    plt.show()


def save_image(img, name):
    img = img.squeeze(0)
    pil = transforms.ToPILImage()
    img = np.array(pil(img))
    print(img.shape)
    plt.imsave("result/" + name + ".jpg", img)
