# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset as ValDataset3
from mit_semseg.dataset_1_channel import ValDataset as ValDataset1
from mit_semseg.models import ModelBuilder, ParallelSegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color_29.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data
    # # ! 4CH
    # img = img[:,:,:3]
    # # !
    # # ! 1CH
    # img = np.repeat(np.expand_dims(img, axis = 2), 3, axis=2)
    # # !

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader_rgb, loader_d, cfg, gpu_id, result_queue, no_vis):
    segmentation_module.eval()

    # Create a new folder within the base result directory for the images
    if cfg.VAL.visualize and not no_vis:
        full_dir_name = os.path.basename(os.path.dirname(os.path.dirname(cfg.DATASET.list_val)))
        resolved_dir = os.path.realpath(cfg.DIR)
        weather_type = loader_rgb.dataset.root_dataset.split('/')[-1]
        base_result_dir = os.path.join("/Data/fusion_out", f"{resolved_dir.split('/')[-1]}{weather_type}")
        if not os.path.exists(base_result_dir):
            os.makedirs(base_result_dir)
        test_set_name = '_'.join(full_dir_name.split('_')[2:])
        result_dir = os.path.join(base_result_dir, f'results_{test_set_name}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    for batch_data_rgb, batch_data_d in zip(loader_rgb, loader_d):
        # process data
        batch_data_rgb = batch_data_rgb[0]
        batch_data_d = batch_data_d[0]
        seg_label = as_numpy(batch_data_rgb['seg_label'][0])
        img_resized_list_rgb = batch_data_rgb['img_data']
        img_resized_list_d = batch_data_d['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img, depth in zip(img_resized_list_rgb, img_resized_list_d):
                feed_dict = batch_data_rgb.copy()
                feed_dict['img_data_rgb'] = img
                feed_dict['img_data_d'] = depth
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue.put_nowait((acc, pix, intersection, union))

        if cfg.VAL.visualize and not no_vis:
            visualize_result(
                (batch_data_rgb['img_ori'], seg_label, batch_data_rgb['info']),
                pred,
                result_dir
            )
            
        


def worker(cfg, gpu_id, start_idx, end_idx, result_queue, test_set, no_vis):
    torch.cuda.set_device(gpu_id)

    if not test_set:
        # Dataset and Loader
        dataset_val_rgb = ValDataset3(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET,
            start_idx=start_idx, end_idx=end_idx)
        dataset_val_d = ValDataset1(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET,
            start_idx=start_idx, end_idx=end_idx)
    else:   
        print(f"Evaluating {cfg.DIR.split('/')[-1]} on {test_set}")
        root_dataset = test_set
        list_val = os.path.join(test_set, "odgt", "test.odgt")
        with open(list_val, 'r') as f:
            print(f"Number of images in {list_val}: {len(f.readlines())}")
        dataset_val_rgb = ValDataset3(
            root_dataset,
            list_val,
            cfg.DATASET,
            start_idx=start_idx, end_idx=end_idx)
        dataset_val_d = ValDataset1(
            root_dataset,
            list_val,
            cfg.DATASET,
            start_idx=start_idx, end_idx=end_idx)
    
   
    
    loader_val_rgb = torch.utils.data.DataLoader(
        dataset_val_rgb,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    loader_val_d = torch.utils.data.DataLoader(
        dataset_val_d,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    net_encoder_rgb = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder_rgb,
        inplanes=3)
    net_encoder_depth = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder_depth,
        inplanes=1)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim*2,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = ParallelSegmentationModule(
        net_encoder_rgb, net_encoder_depth, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val_rgb, loader_val_d, cfg, gpu_id, result_queue, no_vis)


def main(cfg, gpus, test_set, no_vis):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    if test_set:
        list_val = os.path.join(test_set, "odgt", "test.odgt")
        with open(list_val, 'r') as f:
            num_files = len(f.readlines())
        

    num_files_per_gpu = math.ceil(num_files / len(gpus))
    print(cfg.DATASET.list_val)
    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue, test_set, no_vis))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average()*100))

    print('Evaluation Done!')

# python3 eval_multipro_parallel.py --cfg config/hrnetv2_mid_fusion_2.yaml --gpus 0 --test_set data/night
if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "--test_set",
        help="Change from none to path to set if you want to evaluate on a set other than the one in the config",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--no_vis",
        help="Disable visualization",
        action='store_true'
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder_rgb = os.path.join(
        cfg.DIR, 'encoder_rgb_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_encoder_depth = os.path.join(
        cfg.DIR, 'encoder_depth_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder_rgb) and \
        os.path.exists(cfg.MODEL.weights_encoder_depth) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!" 

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    # Parse argument for test set if passed
    test_set = args.test_set

    main(cfg, gpus, test_set, args.no_vis)
