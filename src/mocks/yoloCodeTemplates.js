// YOLO 모델을 위한 코드 템플릿 Mock 데이터

export const yoloFileStructures = {
  yolo_v5: [
    {
      name: 'yolo_v5',
      type: 'folder',
      children: [
        {
          name: 'models',
          type: 'folder',
          children: [
            { name: 'yolo.py', type: 'file' },
            { name: 'common.py', type: 'file' },
            { name: 'experimental.py', type: 'file' }
          ]
        },
        {
          name: 'utils',
          type: 'folder',
          children: [
            { name: 'datasets.py', type: 'file' },
            { name: 'general.py', type: 'file' },
            { name: 'metrics.py', type: 'file' },
            { name: 'torch_utils.py', type: 'file' }
          ]
        },
        { name: 'train.py', type: 'file' },
        { name: 'val.py', type: 'file' },
        { name: 'detect.py', type: 'file' },
        { name: 'export.py', type: 'file' },
        { name: 'data.yaml', type: 'file' },
        { name: 'hyp.yaml', type: 'file' }
      ]
    }
  ],
  
  yolo_v8: [
    {
      name: 'yolo_v8',
      type: 'folder',
      children: [
        {
          name: 'ultralytics',
          type: 'folder',
          children: [
            { name: 'models.py', type: 'file' },
            { name: 'trainer.py', type: 'file' },
            { name: 'predictor.py', type: 'file' },
            { name: 'validator.py', type: 'file' }
          ]
        },
        {
          name: 'data',
          type: 'folder',
          children: [
            { name: 'dataset.py', type: 'file' },
            { name: 'utils.py', type: 'file' }
          ]
        },
        { name: 'train.py', type: 'file' },
        { name: 'predict.py', type: 'file' },
        { name: 'validate.py', type: 'file' },
        { name: 'export.py', type: 'file' },
        { name: 'config.yaml', type: 'file' }
      ]
    }
  ],
  
  yolo_v11: [
    {
      name: 'yolo_v11',
      type: 'folder',
      children: [
        {
          name: 'ultralytics',
          type: 'folder',
          children: [
            { name: 'engine.py', type: 'file' },
            { name: 'models.py', type: 'file' },
            { name: 'trainer.py', type: 'file' },
            { name: 'predictor.py', type: 'file' }
          ]
        },
        {
          name: 'nn',
          type: 'folder',
          children: [
            { name: 'modules.py', type: 'file' },
            { name: 'tasks.py', type: 'file' }
          ]
        },
        { name: 'train.py', type: 'file' },
        { name: 'val.py', type: 'file' },
        { name: 'predict.py', type: 'file' },
        { name: 'export.py', type: 'file' },
        { name: 'config.yaml', type: 'file' }
      ]
    }
  ]
};

export const yoloCodeTemplates = {
  yolo_v5: {
    'train.py': {
      code: `"""
YOLOv5 Training Script
Train a YOLOv5 model on a custom dataset
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

# Add the current directory to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, colorstr, increment_path, init_seeds
from utils.torch_utils import ModelEMA, select_device, torch_distributed_zero_first

def train(hyp, opt, device):
    """
    Train YOLOv5 model
    """
    save_dir = Path(opt.save_dir)
    epochs, batch_size = opt.epochs, opt.batch_size
    weights, data_dict = opt.weights, opt.data
    
    # Directories
    save_dir.mkdir(parents=True, exist_ok=True)
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'
    
    # Configure
    init_seeds(1 + RANK)
    with open(data_dict, errors='ignore') as f:
        data_dict = yaml.safe_load(f)
    
    # Model
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)  # apply decay

    if opt.optimizer == 'Adam':
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    
    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Training step
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            
            # Forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device))
            
            # Backward
            scaler.scale(loss).backward()
            
            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        
        # Validation
        if RANK in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            
            # Save model
            if (not nosave) or final_epoch:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train,val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    
    opt = parser.parse_args()
    
    # Resume
    if opt.resume and not check_wandb_resume(opt):
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \\
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # Train
    train(hyp, opt, device)`,
      language: 'python'
    },
    
    'models/yolo.py': {
      code: `"""
YOLOv5 models
"""

import torch
import torch.nn as nn
from pathlib import Path

from models.common import C3, Conv, SPP, SPPF, Bottleneck, BottleneckCSP, Focus, Contract, Expand
from utils.general import make_divisible

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)
                    xy = (xy * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if torch.__version__ >= '1.10.0':  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \\
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)`,
      language: 'python'
    },
    
    'data.yaml': {
      code: `# YOLO dataset configuration file

# Dataset paths
path: ../datasets/coco  # dataset root dir
train: images/train2017  # train images (relative to 'path') 118287 images
val: images/val2017  # val images (relative to 'path') 5000 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# Download script/URL (optional)
download: |
  from utils.general import download, Path
  
  # Download labels
  segments = False  # segment or box labels
  dir = Path(yaml['path'])  # dataset root dir
  url = f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{'coco2017labels-segments.zip' if segments else 'coco2017labels.zip'}"
  download(url, dir=dir.parent)
  
  # Download data
  urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
          'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
          'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
  for url in urls:
      download(url, dir=dir.parent, delete=False)`,
      language: 'yaml'
    },
    
    'hyp.yaml': {
      code: `# YOLOv5 Hyperparameter Evolution Results
# Best generation: 287
# Last generation: 499
#    Metrics: 0.6201, 0.4266, 0.4767, 0.3728

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)`,
      language: 'yaml'
    }
  },
  
  yolo_v8: {
    'train.py': {
      code: `"""
YOLOv8 Training Script using Ultralytics
"""
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def train_yolov8(model_size='n', data_config='config.yaml', epochs=100, batch_size=16, device=''):
    """
    Train YOLOv8 model
    
    Args:
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        data_config (str): Path to data configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device (str): Device to use for training ('', '0', '0,1', 'cpu')
    """
    
    # Load a model
    model = YOLO(f'yolov8{model_size}.pt')  # load a pretrained model (recommended for training)
    
    # Check device
    if not device:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Train the model
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=640,
        save=True,
        save_period=50,
        cache=False,
        workers=8,
        project='runs/train',
        name='yolov8_experiment',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',  # 'SGD', 'Adam', 'AdamW', 'auto'
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=2.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    # Validate the model
    metrics = model.val()
    
    # Export the model
    model.export(format='onnx')
    
    return results, metrics

def validate_model(weights='runs/train/yolov8_experiment/weights/best.pt', data_config='config.yaml'):
    """Validate trained model"""
    model = YOLO(weights)
    metrics = model.val(data=data_config)
    return metrics

def predict_with_model(weights='runs/train/yolov8_experiment/weights/best.pt', source='path/to/images'):
    """Make predictions with trained model"""
    model = YOLO(weights)
    results = model(source)
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--data', type=str, default='config.yaml', help='Data config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device')
    
    args = parser.parse_args()
    
    # Train the model
    train_yolov8(
        model_size=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )`,
      language: 'python'
    },
    
    'ultralytics/trainer.py': {
      code: `"""
Custom YOLOv8 Trainer Class
Extends the base trainer with custom functionality
"""

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, colorstr
import torch
import torch.nn as nn
from pathlib import Path

class YOLOv8Trainer(BaseTrainer):
    """
    Custom trainer for YOLOv8 with additional features
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load and return model"""
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        """Return validation class"""
        from ultralytics.models.yolo.detect import DetectionValidator
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch['img']
            sz = random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride) // self.stride * self.stride
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]  # new shape
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            batch['img'] = imgs
        return batch
    
    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset"""
        from ultralytics.data.dataset import YOLODataset
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # augmentation
            hyp=self.args,  # hyperparameters
            rect=False,  # rectangular batches
            cache=getattr(self.args, 'cache', None),
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == 'train' else 0.5,
            prefix=colorstr(f'{mode}: '),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == 'train' else 1.0
        )
    
    def criterion(self, preds, batch):
        """Compute loss"""
        from ultralytics.utils.loss import v8DetectionLoss
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = v8DetectionLoss(self.model)
        return self.compute_loss(preds, batch)
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys`,
      language: 'python'
    },
    
    'config.yaml': {
      code: `# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO8 dataset https://github.com/ultralytics/hub/blob/master/example_datasets/coco8.zip
# Example usage: yolo train data=coco8.yaml
# parent
# ├── yolov8 (clone from https://github.com/ultralytics/ultralytics)
# └── datasets
#     └── coco8  ← downloads here (1 MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes for COCO dataset
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# Download script/URL (optional)
download: https://ultralytics.com/assets/coco8.zip`,
      language: 'yaml'
    }
  },
  
  yolo_v11: {
    'train.py': {
      code: `"""
YOLOv11 Training Script
Latest version with enhanced features
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import argparse

def train_yolov11(
    model_size='n', 
    data_config='config.yaml', 
    epochs=100, 
    batch_size=16, 
    device='auto',
    imgsz=640,
    project='runs/train',
    name='yolov11_experiment'
):
    """
    Train YOLOv11 model with enhanced features
    
    Args:
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        data_config (str): Path to data configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device (str): Device to use for training
        imgsz (int): Input image size
        project (str): Project save directory
        name (str): Experiment name
    """
    
    # Load YOLOv11 model
    model = YOLO(f'yolo11{model_size}.pt')
    
    # Enhanced training configuration for YOLOv11
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=imgsz,
        save=True,
        save_period=50,
        cache='disk',  # Enhanced caching
        workers=8,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,
        resume=False,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_json=True,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        crop_fraction=1.0,
        
        # Advanced features for YOLOv11
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
    )
    
    # Enhanced validation
    metrics = model.val(
        data=data_config,
        imgsz=imgsz,
        batch=batch_size,
        save_json=True,
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=False,
        device=device,
        dnn=False,
        plots=True,
        rect=False,
        split='val'
    )
    
    # Export model in multiple formats
    try:
        model.export(format='onnx', dynamic=True, simplify=True)
        model.export(format='torchscript')
        print("✅ Model exported successfully")
    except Exception as e:
        print(f"⚠️ Export failed: {e}")
    
    return results, metrics

def benchmark_model(model_path='yolo11n.pt', data_config='config.yaml', device='auto'):
    """Benchmark YOLOv11 model performance"""
    model = YOLO(model_path)
    
    # Run benchmark
    metrics = model.benchmark(
        data=data_config,
        imgsz=640,
        half=False,
        int8=False,
        device=device,
        verbose=True
    )
    
    return metrics

def hyperparameter_tuning(model_size='n', data_config='config.yaml', iterations=30):
    """
    Perform hyperparameter tuning using YOLOv11's built-in tuner
    """
    model = YOLO(f'yolo11{model_size}.pt')
    
    # Tune hyperparameters
    model.tune(
        data=data_config,
        epochs=30,
        iterations=iterations,
        optimizer='AdamW',
        plots=False,
        save=False,
        val=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv11 Training Script')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model size')
    parser.add_argument('--data', type=str, default='config.yaml', 
                       help='Data configuration file')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Training device')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Input image size')
    parser.add_argument('--project', type=str, default='runs/train', 
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11_experiment', 
                       help='Experiment name')
    parser.add_argument('--tune', action='store_true', 
                       help='Run hyperparameter tuning')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run model benchmark')
    
    args = parser.parse_args()
    
    if args.tune:
        print("🔧 Starting hyperparameter tuning...")
        hyperparameter_tuning(args.model, args.data)
    elif args.benchmark:
        print("📊 Running model benchmark...")
        model_path = f'yolo11{args.model}.pt'
        benchmark_model(model_path, args.data, args.device)
    else:
        print("🚀 Starting YOLOv11 training...")
        train_yolov11(
            model_size=args.model,
            data_config=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name
        )`,
      language: 'python'
    },
    
    'ultralytics/engine.py': {
      code: `"""
YOLOv11 Enhanced Engine
Advanced training and inference engine with latest features
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Union
import numpy as np

class YOLOv11Engine:
    """Enhanced engine for YOLOv11 with advanced features"""
    
    def __init__(self, model_config: Union[str, Dict], device: str = 'auto'):
        self.device = self._select_device(device)
        self.model_config = model_config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def _select_device(self, device: str) -> torch.device:
        """Select appropriate device for training/inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def load_model(self, weights: Optional[str] = None, verbose: bool = True):
        """Load YOLOv11 model with enhanced architecture"""
        from ultralytics.nn.tasks import DetectionModel
        
        if isinstance(self.model_config, str):
            with open(self.model_config, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = self.model_config
            
        self.model = DetectionModel(cfg, verbose=verbose)
        
        if weights:
            checkpoint = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            if verbose:
                print(f"✅ Loaded weights from {weights}")
                
        self.model.to(self.device)
        return self.model
    
    def setup_training(self, 
                      optimizer_type: str = 'AdamW',
                      learning_rate: float = 0.001,
                      weight_decay: float = 0.0005,
                      scheduler_type: str = 'cosine'):
        """Setup optimizer and scheduler for training"""
        
        # Enhanced optimizer setup
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.937,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Enhanced scheduler setup
        if scheduler_type.lower() == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=learning_rate * 0.01
            )
        elif scheduler_type.lower() == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate, total_steps=1000
            )
            
    def train_step(self, batch: Dict, criterion) -> Dict[str, float]:
        """Single training step with enhanced features"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(batch['img'])
        
        # Compute loss
        loss, loss_items = criterion(predictions, batch)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'total_loss': loss.item(),
            'box_loss': loss_items[0].item() if len(loss_items) > 0 else 0,
            'cls_loss': loss_items[1].item() if len(loss_items) > 1 else 0,
            'dfl_loss': loss_items[2].item() if len(loss_items) > 2 else 0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_step(self, batch: Dict, criterion) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(batch['img'])
            loss, loss_items = criterion(predictions, batch)
            
        return {
            'val_loss': loss.item(),
            'val_box_loss': loss_items[0].item() if len(loss_items) > 0 else 0,
            'val_cls_loss': loss_items[1].item() if len(loss_items) > 1 else 0,
            'val_dfl_loss': loss_items[2].item() if len(loss_items) > 2 else 0
        }
    
    def predict(self, 
               source: Union[str, np.ndarray], 
               conf: float = 0.25,
               iou: float = 0.45,
               max_det: int = 1000,
               classes: Optional[List[int]] = None) -> List[Dict]:
        """Enhanced prediction with post-processing"""
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(source, str):
                # Load image from path
                import cv2
                img = cv2.imread(source)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = source
            
            # Preprocess
            img_tensor = self._preprocess_image(img)
            
            # Inference
            predictions = self.model(img_tensor)
            
            # Post-process
            results = self._postprocess_predictions(
                predictions, conf, iou, max_det, classes
            )
            
        return results
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Resize and normalize
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        img = img.unsqueeze(0).to(self.device)
        return img
    
    def _postprocess_predictions(self, 
                               predictions: torch.Tensor,
                               conf: float,
                               iou: float,
                               max_det: int,
                               classes: Optional[List[int]]) -> List[Dict]:
        """Post-process model predictions"""
        from ultralytics.utils.ops import non_max_suppression
        
        # Apply NMS
        predictions = non_max_suppression(
            predictions, conf, iou, classes, False, max_det=max_det
        )
        
        results = []
        for pred in predictions:
            if pred is not None and len(pred):
                # Convert to CPU and numpy
                pred = pred.cpu().numpy()
                
                # Extract boxes, scores, classes
                boxes = pred[:, :4]
                scores = pred[:, 4]
                class_ids = pred[:, 5].astype(int)
                
                results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids
                })
            else:
                results.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'class_ids': np.array([])
                })
                
        return results
    
    def save_checkpoint(self, save_path: str, epoch: int, best_fitness: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_fitness': best_fitness,
            'model_config': self.model_config
        }
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved to {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model:
            self.model.load_state_dict(checkpoint['model'])
        if self.optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        print(f"📂 Checkpoint loaded from {checkpoint_path}")
        return checkpoint`,
      language: 'python'
    },
    
    'config.yaml': {
      code: `# YOLOv11 Enhanced Configuration
# Ultralytics YOLOv11 🚀, AGPL-3.0 license

# Dataset configuration
path: ../datasets/coco  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Class configuration
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# Enhanced model configuration for YOLOv11
model:
  # Backbone
  backbone: 'CSPDarknet'
  depth_multiple: 0.33  # model depth multiple
  width_multiple: 0.25  # layer channel multiple
  
  # Head
  head: 'YOLOv11Head'
  
  # Anchors (optional, YOLOv11 can be anchor-free)
  anchors: 3
  
# Enhanced training configuration
train:
  # Basic settings
  epochs: 300
  batch_size: 16
  workers: 8
  
  # Image settings
  imgsz: 640
  
  # Optimization
  optimizer: 'auto'  # auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp
  lr0: 0.01  # initial learning rate
  lrf: 0.01  # final learning rate factor
  momentum: 0.937  # SGD momentum/Adam beta1
  weight_decay: 0.0005  # optimizer weight decay
  warmup_epochs: 3.0  # warmup epochs
  warmup_momentum: 0.8  # warmup initial momentum
  warmup_bias_lr: 0.1  # warmup initial bias lr
  
  # Loss function weights
  box: 7.5  # box loss gain
  cls: 0.5  # cls loss gain
  dfl: 1.5  # dfl loss gain
  
  # Data augmentation
  hsv_h: 0.015  # image HSV-Hue augmentation
  hsv_s: 0.7  # image HSV-Saturation augmentation
  hsv_v: 0.4  # image HSV-Value augmentation
  degrees: 0.0  # image rotation (+/- deg)
  translate: 0.1  # image translation (+/- fraction)
  scale: 0.5  # image scale (+/- gain)
  shear: 0.0  # image shear (+/- deg)
  perspective: 0.0  # image perspective (+/- fraction)
  flipud: 0.0  # image flip up-down (probability)
  fliplr: 0.5  # image flip left-right (probability)
  mosaic: 1.0  # image mosaic (probability)
  mixup: 0.0  # image mixup (probability)
  copy_paste: 0.0  # segment copy-paste (probability)
  erasing: 0.4  # random erasing (probability)
  crop_fraction: 1.0  # image crop fraction
  
  # Advanced augmentation (YOLOv11 specific)
  auto_augment: 'randaugment'  # auto augmentation policy
  
# Validation configuration
val:
  imgsz: 640
  batch_size: 32
  save_json: true
  save_hybrid: false
  conf: 0.001  # confidence threshold
  iou: 0.6  # NMS IoU threshold
  max_det: 300  # maximum detections per image
  half: false  # use FP16 half-precision inference
  device: ''  # device to use
  dnn: false  # use OpenCV DNN for ONNX inference
  plots: true  # save plots during val
  rect: false  # rectangular val
  split: 'val'  # dataset split to use for validation

# Export configuration
export:
  format: 'onnx'  # export format
  dynamic: true  # ONNX/TF/TensorRT: dynamic axes
  simplify: true  # ONNX: simplify model
  int8: false  # CoreML/TF INT8 quantization
  half: false  # FP16 quantization
  
# Predict configuration
predict:
  source: ''  # source directory for images or videos
  imgsz: 640  # inference size
  conf: 0.25  # confidence threshold
  iou: 0.45  # NMS IoU threshold
  max_det: 1000  # maximum detections per image
  device: ''  # device to use
  show: false  # show results
  save: true  # save results
  save_txt: false  # save results as txt
  save_conf: false  # save results with confidence scores
  save_crop: false  # save cropped prediction boxes
  show_labels: true  # show object labels in plots
  show_conf: true  # show object confidence scores in plots
  vid_stride: 1  # video frame-rate stride
  line_width: 3  # bounding box thickness
  visualize: false  # visualize model features
  augment: false  # apply image augmentation to prediction sources
  agnostic_nms: false  # class-agnostic NMS
  retina_masks: false  # use high-resolution segmentation masks
  boxes: true  # show boxes in segmentation predictions

# Download script/URL (optional)
download: |
  # Download script for custom dataset
  import os
  from pathlib import Path
  
  # Define paths
  dir = Path(yaml['path'])
  
  # Create directories
  for p in 'images', 'labels':
      (dir / p).mkdir(parents=True, exist_ok=True)
      for q in 'train', 'val':
          (dir / p / q).mkdir(parents=True, exist_ok=True)
  
  # Download your custom dataset here
  print("Dataset directory structure created")
  print("Please add your images and labels to the appropriate directories")`
      ,language: 'yaml'
    }
  }
};

// 사용 가능한 템플릿 목록
export const availableTemplates = [
  {
    algorithm: 'yolo_v5',
    name: 'YOLOv5',
    description: 'Popular and reliable object detection model',
    features: ['PyTorch native', 'Modular design', 'Extensive documentation']
  },
  {
    algorithm: 'yolo_v8',
    name: 'YOLOv8',
    description: 'Latest Ultralytics YOLO with improved performance',
    features: ['Unified framework', 'Easy to use', 'Multiple tasks support']
  },
  {
    algorithm: 'yolo_v11',
    name: 'YOLOv11',
    description: 'Cutting-edge YOLO model with enhanced accuracy',
    features: ['State-of-the-art performance', 'Enhanced architecture', 'Advanced augmentation']
  }
];

// Mock API functions
export const mockFetchCodeTemplate = async (algorithm, projectId) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));
  
  if (!yoloFileStructures[algorithm]) {
    throw new Error(`Template not found for algorithm: ${algorithm}`);
  }
  
  return {
    algorithm,
    projectId,
    fileStructure: yoloFileStructures[algorithm],
    files: yoloCodeTemplates[algorithm],
    lastModified: new Date().toISOString(),
    version: '1.0.0'
  };
};

export const mockSaveCodeTemplate = async (algorithm, projectId, files) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 300));
  
  // Update the mock data
  yoloCodeTemplates[algorithm] = { ...yoloCodeTemplates[algorithm], ...files };
  
  return {
    success: true,
    snapshotId: `snapshot_${Date.now()}`,
    algorithm,
    projectId,
    savedAt: new Date().toISOString()
  };
};


