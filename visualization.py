from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def showpoints(points, gt=None, pred=None):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Point Cloud')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    if gt is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title('Ground Truth')
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=gt, s=5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    if pred is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title('Prediction')
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred, s=5)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

    plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--models_dir', type=str, default='E:\\folder\\CV\\HW4', help='directory of models')
parser.add_argument('--dataset', type=str, default='E:\\folder\\CV\\HW4\\shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--idx', type=int, default=1, help='model index')

opt = parser.parse_args()
print(opt)

#classes = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
classes = ['Skateboard']
for class_choice in classes:
    model_path = os.path.join(opt.models_dir, f'{class_choice}.pth')
    
    if not os.path.exists(model_path):
        print(f'Model for {class_choice} does not exist at {model_path}')
        continue
    
    d = ShapeNetDataset(
        root=opt.dataset,
        class_choice=[class_choice],
        split='test',
        data_augmentation=False)

    idx = opt.idx

    print(f"Visualizing model {class_choice}, sample {idx}/{len(d)}")
    point, seg = d[idx]
    print(point.size(), seg.size())
    point_np = point.numpy()

    cmap = plt.get_cmap("hsv", 10)  # Updated API
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    gt = cmap[seg.numpy() - 1, :]

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load model to CPU
    classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0])
    classifier.load_state_dict(state_dict)
    classifier.eval()

    point = point.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]
    print(pred_choice)

    pred_color = cmap[pred_choice.numpy()[0], :]

    showpoints(point_np, gt, pred_color)
