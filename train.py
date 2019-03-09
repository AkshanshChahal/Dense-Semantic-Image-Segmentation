import time

from models.segnet import DenseSegNet
from augmentations import get_composed_augmentations
from loader.cityscapes import CityscapesLoader
from metrics import runningScore, averageMeter

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.models as models
from torch.optim import SGD

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Setup Augmentations
augmentations = None #cfg["training"].get("augmentations", None)
data_aug = get_composed_augmentations(augmentations)

# Setup Dataloader
data_loader = CityscapesLoader
data_path = "/datasets/cityscapes"
image_size = (256, 512)
batch_size = 8

t_loader = data_loader(
    data_path,
    is_transform=True,
    split="train",
    img_size=image_size,
    augmentations=data_aug,
)

v_loader = data_loader(
    data_path,
    is_transform=True,
    split="val",
    img_size=image_size,
)


n_classes = t_loader.n_classes
trainloader = data.DataLoader(
    t_loader,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
)
n_classes=20

valloader = data.DataLoader(
    v_loader, batch_size=batch_size, num_workers=1
)


# Setup Model
model = DenseSegNet(num_classes=n_classes)
vgg16 = models.vgg16(pretrained=True)
model.init_vgg16_params(vgg16)
model = model.to(device)

# Loss function
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

# optimier
optimizer = SGD(model.parameters(), lr = 0.01)

# Setup Metrics
running_metrics_val = runningScore(n_classes)
val_loss_meter = averageMeter()

num_epochs = 10
step = 0
epoch = 0
while epoch <= num_epochs:
    epoch += 1
    print("Starting epoch %s" % epoch)
    for (images, labels) in trainloader:
        step += 1
        start_ts = time.time()
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = cross_entropy2d(input=outputs, target=labels)

        loss.backward()
        optimizer.step()
        
        # print(images.shape, labels.shape)
        # torch.Size([2, 3, 512, 1024]) torch.Size([2, 512, 1024])
        print(step)
        
        if step%50 == 0:
            model.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val) in enumerate(valloader):
                    images_val = images_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs = model(images_val)
                    val_loss = cross_entropy2d(input=outputs, target=labels_val)

                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels_val.data.cpu().numpy()

                    running_metrics_val.update(gt, pred)
                    val_loss_meter.update(val_loss.item())
                    print("val step done")
                    if(i_val%10==0):
                        break

            print("Iter %d Loss: %.4f" % (step, val_loss_meter.avg))
            val_loss_meter.reset()

            score, class_iou = running_metrics_val.get_scores()
            for k, v in score.items():
                print("kv: ", k, v)
        #         logger.info("{}: {}".format(k, v))
        #         writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

            for k, v in class_iou.items():
                print("ikv: ", k, v)
        #         logger.info("{}: {}".format(k, v))
        #         writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)
            running_metrics_val.reset()