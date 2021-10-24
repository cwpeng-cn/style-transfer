import torch
from torch import optim
from model import Transfer
from utils import get_image, show_image, save_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载图片
CONTENT_IMAGE_PATH = "./images/content.png"
STYLE_IMAGE_PATH = "./images/style.png"

content_image = get_image(CONTENT_IMAGE_PATH, 256).to(device)
style_image = get_image(STYLE_IMAGE_PATH, 256).to(device)

# 加载模型
model = Transfer(style_image, content_image, device)
net = model.basenet.to(device).eval()

# 定义损失
style_losses = model.style_losses
content_losses = model.content_losses

# 定义输入
# input_image = torch.randn(content_image.data.size()).cuda()
input_image = content_image.clone().to(device)

# define optimizer
optimizer = optim.SGD([input_image.requires_grad_()])

print("Start training......")

step = 0
while step < 300:
    input_image.data.clamp_(0, 1)
    optimizer.zero_grad()
    net(input_image)
    style_score = 0
    content_score = 0
    for style_loss in style_losses:
        style_score = style_score + 100000 * style_loss.loss
    for content_loss in content_losses:
        content_score = content_score + content_loss.loss
    loss = style_score + content_score
    loss.backward()
    if step % 10 == 0:
        print("step:", step, " style_loss:", style_score.data, " content_loss:", content_score.data)
    step += 1
    optimizer.step()

input_img = input_image.cpu().data.clamp_(0, 1)
show_image(input_img)
save_image(input_img, CONTENT_IMAGE_PATH.split("/")[-1])
print('End of the training')
