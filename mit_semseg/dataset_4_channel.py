
import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import math
import skimage.measure

INPLANES = 4

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def lin_mapping(vector, max_val = 250):
    distance_scaled = np.interp(vector, [0, max_val], [0, 255]).astype(np.uint8)
    return distance_scaled

def log_mapping(vector, max_val = 250):
    log1p = lambda x: 255 * math.log(x + 1) / math.log(max_val + 1)
    distance_scaled = np.vectorize(log1p)(vector).astype(np.uint8)
    return distance_scaled

class Neighbor:
    def __init__(self):
        pass
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
    def set_location(self, x, y):
        self.x = x
        self.y = y
    def set_value(self, value):
        self.value = value
    
    def __str__(self):
        return f"({self.x}, {self.y}) = {self.value}"

def find_neighbors(x, y, img, max_rad):
    rad = 0
    neighbors = []
    while len(neighbors) < 3 and rad < max_rad:
        rad += 1
        for dx in range(-rad, rad+1):
            dy = rad
            if dx == 0 and dy == 0:
                continue
            if x+dx < 0 or x+dx >= img.shape[0] or y+dy < 0 or y+dy >= img.shape[1]:
                continue
            if img[x+dx, y+dy] != 0:
                n = Neighbor(x = x+dx, y = y+dy, value = img[x+dx, y+dy])
                neighbors.append(n)
        for dx in range(-rad, rad+1):
            dy = -rad
            if dx == 0 and dy == 0:
                continue
            if x+dx < 0 or x+dx >= img.shape[0] or y+dy < 0 or y+dy >= img.shape[1]:
                continue
            if img[x+dx, y+dy] != 0:
                n = Neighbor(x = x+dx, y = y+dy, value = img[x+dx, y+dy])
                neighbors.append(n)
        for dy in range(-rad+1, rad):
            dx = rad
            if dx == 0 and dy == 0:
                continue
            if x+dx < 0 or x+dx >= img.shape[0] or y+dy < 0 or y+dy >= img.shape[1]:
                continue
            if img[x+dx, y+dy] != 0:
                n = Neighbor(x = x+dx, y = y+dy, value = img[x+dx, y+dy])
                neighbors.append(n)
        for dy in range(-rad+1, rad):
            dx = -rad
            if dx == 0 and dy == 0:
                continue
            if x+dx < 0 or x+dx >= img.shape[0] or y+dy < 0 or y+dy >= img.shape[1]:
                continue
            if img[x+dx, y+dy] != 0:
                n = Neighbor(x = x+dx, y = y+dy, value = img[x+dx, y+dy])
                neighbors.append(n)

    return neighbors

def weighted_interpolation(x, y, neighbors):
    numerator = 0
    denominator = 0

    w = lambda d: math.exp(-0.5*d)
    d = lambda x1, y1, x2, y2: math.sqrt((x1-x2)**2 + (y1-y2)**2)

    for n in neighbors:
        distance = d(x, y, n.x, n.y)
        numerator += n.value * w(distance)
        denominator += w(distance)
    
    return numerator / denominator

def fill_pixel(x, y, img, max_rad = 30):
    if img[x, y] == 0:
      neighbors = find_neighbors(x, y, img, max_rad)
      
      # if (np.all([n.y >= y for n in neighbors])):
      #   return 255
      if len(neighbors) == 0:
          return 0
      return weighted_interpolation(x, y, neighbors)
    else:
      return img[x, y]
    
def imresize(im, size, interp='nearest'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def lin_mapping(vector, max_val = 250):
    distance_scaled = np.interp(vector, [0, max_val], [0, 255]).astype(np.uint8)
    return distance_scaled
    
def densify(img):
    a = skimage.measure.block_reduce(img, (12, 12), np.max)
    new_img = np.zeros(a.shape, dtype=np.int8)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            new_img[i, j] = fill_pixel(i, j, a, max_rad = 3)

    interp = Image.fromarray(new_img, mode="L")
    return imresize(interp, (1920, 1080))

def to_sparse_depth_image(lidar, w, h):
    VCOORD = 0
    UCOORD = 1
    DISTANCE = 3

    vcoord = lidar[VCOORD].astype(np.uint16)
    ucoord = lidar[UCOORD].astype(np.uint16)

    # distance_cm = (lidar[DISTANCE] * 100).astype(np.uint16)
    distance_scaled = log_mapping(lidar[DISTANCE], 250)
    # distance_scaled = np.interp(lidar[DISTANCE], [0, 250], [0, 255]).astype(np.uint8)

    distance = np.zeros((h, w), dtype=np.uint8)
    distance[vcoord, ucoord] = distance_scaled

    return distance

def stack_layers(lidar_arr: np.ndarray, rgb_arr: np.ndarray) -> np.ndarray:
    distance = to_sparse_depth_image(lidar_arr, rgb_arr.shape[1], rgb_arr.shape[0])
    distance = densify(distance)
    # distance.save('distance.png')
    distance = np.expand_dims(distance, axis = 2)

    assert distance.shape[0] == rgb_arr.shape[0] and distance.shape[1] == rgb_arr.shape[1]

    stack = np.concatenate((rgb_arr, distance), axis = 2)
    return stack

# def stack_cheat_layers(rgb, root_dir):
#     rgb_arr = np.array(Image.open(os.path.join(root_dir, 'rgb', rgb)).convert('RGB'), dtype = np.uint8)
#     seg_arr = np.array(Image.open(os.path.join(root_dir, 'rgb_seg', rgb)).convert('L'), dtype = np.uint8)
#     seg_arr = seg_arr / 29
#     seg_arr = np.expand_dims(seg_arr, axis = 2)
#     stack = np.concatenate((rgb_arr, seg_arr), axis = 2)
#     return stack

# def stack_noise_layers(rgb, root_dir):
#     rgb_arr = np.array(Image.open(os.path.join(root_dir, 'rgb', rgb)).convert('RGB'), dtype = np.uint8)
#     rand = np.random.rand(rgb_arr.shape[0], rgb_arr.shape[1], 1).astype(np.float32)
#     stack = np.concatenate((rgb_arr, rand), axis = 2)
#     return stack

def noise_image(rgb_arr, brightness, channels = 3, courseness = 8):
    y = rgb_arr.shape[0]
    x = rgb_arr.shape[1]
    rand = np.random.rand(y // courseness, x // courseness, channels).astype(np.float32)
    rand_im = Image.fromarray((rand * brightness).astype(np.uint8))
    return imresize(rand_im, (x, y))

# Load image and lidar data and return the stack as a cmyk image (actually rgbd)
def load_image(img_path, noise_rate = 0):
    root_dir = '/'.join(os.path.dirname(img_path).split('/')[:-1])
    rgb = os.path.basename(img_path)
    lidar = os.path.basename(img_path).replace('.png', '.npy')

    rgb_arr = np.array(Image.open(os.path.join(root_dir, 'rgb', rgb)).convert('RGB'), dtype = np.uint8)
    lidar_arr = np.load(os.path.join(root_dir, 'lidar_2d', lidar))
    if noise_rate:
        if np.random.rand() < noise_rate:
            brightness = int(np.random.beta(3, 8) * 255)
            rgb_arr = np.array(noise_image(rgb_arr, brightness, channels = 3))
            temp = Image.fromarray(rgb_arr)
    
    img = stack_layers(lidar_arr, rgb_arr)

    # # ! TEMP
    # img = img[:,:,3]
    # img = Image.fromarray(img, mode = 'L')
    img = Image.fromarray(img, mode = 'CMYK')
    # img.save('cmky.tiff')
    return img

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.586],
            std=[0.229, 0.224, 0.225, 0.172])

        # # ! TEMP
        # self.normalize = transforms.Normalize(
        #     mean=[0.586],
        #     std=[0.172])
        # # !

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        # # ! TEMP
        # img = np.array(img)
        # img = np.expand_dims(img, axis = 2)
        # # !

        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros( 
            self.batch_per_gpu, INPLANES, batch_height, batch_width) # ! 4CH
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            # img = Image.open(image_path).convert('RGB')
            # ! 4CH
            img = load_image(image_path)
            segm = Image.fromarray(np.array(Image.open(segm_path))[:,:,0])
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)


            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        # img = Image.open(image_path).convert('RGB')
        # ! 4CH
        img = load_image(image_path)
        segm = Image.fromarray(np.array(Image.open(segm_path))[:,:,0])
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        # img = Image.open(image_path).convert('RGB')
        # ! 4CH
        img = load_image(image_path)

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample