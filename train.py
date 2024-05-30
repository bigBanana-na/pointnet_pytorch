from __future__ import print_function
import os
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='E:/folder/CV/HW4/data', help='dataset path')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    opt = parser.parse_args()
    print(opt)

    # Create output folder
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # Load dataset
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=0
    )

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test'
    )
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=0
    )

    # Load model
    num_classes = len(dataset.classes)
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    # Optimizer and loss function
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    # Training loop
    for epoch in range(opt.nEpochs):
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target = points.transpose(2, 1).cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        # Test the model
        total_correct = 0
        total_testset = 0
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            points, target = points.transpose(2, 1).cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

        print("accuracy: %f" % (total_correct / float(total_testset)))

        # Save the model
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

if __name__ == '__main__':
    main()