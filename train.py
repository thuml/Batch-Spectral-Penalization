import torch
import torch.optim as optim
import torch.nn as nn
import model
import transform as tran
import adversarial1 as ad
import numpy as np
from read_data import ImageList
import argparse
import os
import torch.nn.functional as F

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='PyTorch BSP Example')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='A', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='C', metavar='S',
                    help='target dataset')
args = parser.parse_args()

def get_datasetname(args):
    visda = False
    office = False
    A = './data/Art.txt'
    C = './data/Clipart.txt'
    P = './data/Product.txt'
    R = './data/Real_World.txt'
    if args.src == 'A':
        src = A
    elif args.src == 'C':
        src = C
    elif args.src == 'P':
        src = P
    elif args.src == 'R':
        src = R
    if args.tgt == 'A':
        tgt = A
    elif args.tgt == 'C':
        tgt = C
    elif args.tgt == 'P':
        tgt = P
    elif args.tgt == 'R':
        tgt = R
    a = './data/amazon.txt'
    w = './data/webcam.txt'
    d = './data/dslr.txt'
    if args.src == 'a':
        src = a
        office = True
    elif args.src == 'w':
        src = w
        office = True
    elif args.src == 'd':
        src = d
        office = True
    if args.tgt == 'a':
        tgt = a
    elif args.tgt == 'w':
        tgt = w
    elif args.tgt == 'd':
        tgt = d
    visda_train = "./data/train_list.txt"
    visda_test = "./data/validation_list.txt"
    if args.src == 'visda':
        src = visda_train
        tgt = visda_test
        visda = True
    return src,tgt,office,visda


src,tgt,office,visda = get_datasetname(args)

batch_size = {"train": 36, "val": 36, "test": 4}
for i in range(10):
    batch_size["val" + str(i)] = 4

if visda == False:
    data_transforms = {
        'train': tran.transform_train(resize_size=256, crop_size=224),
        'val': tran.transform_train(resize_size=256, crop_size=224),
    }
    data_transforms = tran.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)
    dsets = {"train": ImageList(open(src).readlines(), transform=data_transforms["train"]),
             "val": ImageList(open(tgt).readlines(), transform=data_transforms["val"]),
             "test": ImageList(open(tgt).readlines(),transform=data_transforms["val"])}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                       shuffle=False, num_workers=4)

    for i in range(10):
        dsets["val" + str(i)] = ImageList(open(tgt).readlines(),
                                          transform=data_transforms["val" + str(i)])
        dset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dsets["val" + str(i)],
                                                                   batch_size=batch_size["val" + str(i)], shuffle=False,
                                                                   num_workers=4)

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val'] + ["val" + str(i) for i in range(10)]}
    dset_classes = range(65)
else:
    data_transforms = {
        'train': tran.Visda_train(resize_size=256, crop_size=224),
        'val': tran.Visda_train(resize_size=256, crop_size=224),
        'test': tran.Visda_eval(resize_size=256, crop_size=224),
    }

    dsets = {"train": ImageList(open(src).readlines(), transform=data_transforms["train"]),
             "val": ImageList(open(tgt).readlines(), transform=data_transforms["val"]),
             "test": ImageList(open(tgt).readlines(), transform=data_transforms["test"])}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                       shuffle=False, num_workers=64)

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    dset_classes = range(12)
if office == True:
    dset_classes = range(31)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test_target(loader, model, test_iter=0):
    with torch.no_grad():
        start_test = True
        if test_iter > 0:
            iter_val = iter(loader['val0'])
            for i in range(test_iter):
                data = iter_val.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        else:
            iter_val = [iter(loader['val'+str(i)]) for i in range(10)]
            for i in range(len(loader['val0'])):
                data = [iter_val[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].to(device)
                labels = labels.to(device)
                outputs = []
                for j in range(10):
                    output = model(inputs[j])
                    outputs.append(output)
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def Visda_test(loader, model,iter_num):
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(len(dset_classes))
        subclasses_tick = np.zeros(len(dset_classes))
        for (imgs, labels) in loader['test']:
            tick += 1
            imgs = imgs.to(device)
            pred = model(imgs)
            pred = F.softmax(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print(iter_num)
        for i in range(len(dset_classes)):
            print("\tClass {0} : {1}".format(i, subclasses_result[i]))
        print("\tAvg : {0}\n".format(subclasses_result.mean()))


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer


if visda == True:
    DANN = model.BSP_Res101(len(dset_classes))
else:
    DANN = model.BSP_Res50(len(dset_classes))
DANN = DANN.to(device)

DANN.train(True)
criterion = {"classifier":nn.CrossEntropyLoss(), "adversarial":nn.BCELoss()}
optimizer_dict =[{"params":filter(lambda p: p.requires_grad,DANN.model_fc.parameters()), "lr":0.1},{"params":filter(lambda p: p.requires_grad,DANN.bottleneck_layer.parameters()), "lr":1},{"params":filter(lambda p: p.requires_grad,DANN.classifier_layer.parameters()), "lr":1},{"params":filter(lambda p: p.requires_grad,DANN.ad_net.parameters()), "lr":1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)
train_cross_loss = train_transfer_loss = train_total_loss = train_sigma =0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])
for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])
test_interval = 500
num_iter = 50002
for iter_num in range(1, num_iter+1):
    DANN.train(True)
    if visda == True:
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.003, gamma=0.0001, power=0.75)
    else:
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.003, gamma=0.001, power=0.75)
    optimizer.zero_grad()
    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["val"])
    data_source = iter_source.next()
    data_target = iter_target.next()
    inputs_source, labels_source = data_source
    inputs_target, labels_target = data_target
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    dc_target = torch.from_numpy(np.array([[1],] * batch_size["train"] + [[0],] * batch_size["train"])).float()
    inputs = inputs.to(device)
    labels = labels_source.to(device)
    dc_target = dc_target.to(device)
    outC,outD,feature = DANN(inputs)
    feature_s = feature.narrow(0, 0, int(feature.size(0) / 2))
    feature_t = feature.narrow(0, int(feature.size(0) / 2), int(feature.size(0) / 2))
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0],2)
    sigma_loss = 0.0001 * sigma
    classifier_loss = criterion["classifier"](outC.narrow(0, 0, batch_size["train"]), labels)
    total_loss = classifier_loss
    transfer_loss = nn.BCELoss()(outD, dc_target)
    total_loss = total_loss + transfer_loss + sigma_loss
    total_loss.backward()
    optimizer.step()
    train_cross_loss += classifier_loss.item()
    train_transfer_loss += transfer_loss.item()
    train_total_loss += total_loss.item()
    train_sigma += sigma_loss.item()
    if iter_num % test_interval == 0:
        print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f}; Average Sigma Loss: {:.4f}; Average Training Loss: {:.4f}".format(
            iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),train_sigma / float(test_interval),
            train_total_loss / float(test_interval)))
        train_cross_loss = train_transfer_loss = train_total_loss =train_sigma = 0.0
    if (iter_num % 500) == 0:
        DANN.eval()
        if visda == False:
            test_acc = test_target(dset_loaders, DANN.predict_layer)
            print('test_acc:%.4f'%(test_acc))
        elif visda == True:
            Visda_test(dset_loaders, DANN.predict_layer, iter_num)



