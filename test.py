from network import *
from tools import *

from PIL import Image
import torchvision.transforms as tt

n_classes = 2
image_path = '2.png'

ssd300 = SSD300(n_classes)
ssd300.load_state_dict(torch.load('./models/ssd300.pth', map_location=torch.device('cpu')))

transforms = tt.Compose([tt.ToTensor(), tt.Resize((300, 300))])
img = transforms(Image.open(image_path))[:3,:,:]
print(img.shape)
predict_plot(ssd300, img)
