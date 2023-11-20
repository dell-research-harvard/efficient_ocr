from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
import kornia


BASE_TRANSFORM = T.Compose([
    T.ToTensor(), 
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


GRAY_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


INV_NORMALIZE = T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]
)


class ResizeWidth:


    def __init__(self, pcts):

        self.pcts = pcts


    def __call__(self, image):

        _, h, w = image.size()
        new_w =[int(w*p) for p in self.pcts]        
        
        return T.Resize((h, np.random.choice(new_w)), antialias = True)(image)


class TransformLoader:


    def __init__(self, loader, transform):
        self.loader = loader
        self.transform = transform


    def __iter__(self):
        for data, target in self.loader:
            ##EAch data is a tensor of 252 images. We need to traansform each image 
            ##First, unstack the data by axis 0
            data = torch.unbind(data, dim=0)
            ##Now apply the transform to each image
            data = [self.transform((T.ToPILImage()(word)))  for word in data]
            ##Now stack the images back together
            data = torch.stack(data)

            yield data, target


    def __len__(self):
        return len(self.loader)


class CharMedianPad:


    def __init__(self, override=None):

        self.override = override


    def __call__(self, image):

        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        padding = (0, 0, pad_x, pad_y)

        imgarray = np.array(image)
        h, w, c = imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)


class MedianPad:
    """This padding preserves the aspect ratio of the image. It also pads the image with the median value of the border pixels. 
    Note how it also centres the ROI in the padded image."""


    def __init__(self, override=None,random_pad=False):

        self.override = override
        self.random_pad=random_pad


    def __call__(self, image):

        ##Convert to RGB 
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        aspect_ratio = image.size[0] / image.size[1]
        if aspect_ratio<1.5:
            pad_x, pad_y = [2*max_side - (0.5*s) for s in image.size]
        else:
            pad_x, pad_y = [max_side - (0.9*s) for s in image.size]

        if self.random_pad:
            ###randomly move crop up and down by varying the padding - this also randomly incrases and decreases size.
            random_top_pad=np.random.uniform(1, 10)
            random_bottom_pad=np.random.uniform(1, 10)
            padding = (round((pad_x)/random_top_pad), round((pad_y)/random_top_pad), round((pad_x)/random_bottom_pad), round((pad_y)/random_bottom_pad)) ##Added some extra to avoid info on the long edge
        else:
            padding = (round((pad_x)/2), round((pad_y)/2), round((pad_x)/2), round((pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        padded_image=T.Pad(padding, fill=medval if self.override is None else self.override)(image)
        ##Aspect ratio 
        return padded_image


def dilate(x):
    return kornia.morphology.dilation(x.unsqueeze(0), 
                                      kernel=torch.ones(5, 5)).squeeze(0)


def erode(x):
    return kornia.morphology.erosion(x.unsqueeze(0), 
                                     kernel=torch.ones(5, 5)).squeeze(0)


def random_erode_dilate(x):
    erode = np.random.choice([True, False])
    if erode:
        return kornia.morphology.dilation(x.unsqueeze(0), 
            kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)
    else:
        return kornia.morphology.erosion(x.unsqueeze(0), 
            kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)


def patch_resize(pil_img, patchsize=8, targetsize=224):

    w, h = pil_img.size
    larger_side = max([w, h])
    height_larger = larger_side == h 
    aspect_ratio = w / h if height_larger else h / w
    
    if height_larger:
        patch_resizer = T.Resize((targetsize, (int(aspect_ratio*targetsize) // patchsize) * patchsize), antialias=True)
    else:
        patch_resizer = T.Resize(((int(aspect_ratio*targetsize) // patchsize) * patchsize, targetsize), antialias=True)

    return patch_resizer(pil_img)


def color_shift(im):
    color = list(np.random.random(size=3))
    im[0, :, :][im[0, :, :] >= 0.8] = color[0]
    im[1, :, :][im[1, :, :] >= 0.8] = color[1]
    im[2, :, :][im[2, :, :] >= 0.8] = color[2]
    return im


def blur_transform(high):
    if high:
        return T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.3)
    else:
        return  T.RandomApply([T.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.3)


def create_render_transform_char(char_trans_version, latin_suggested_augs, size=224):
    if char_trans_version == 2: # suggested for english/latin
        return T.Compose([
            T.ToTensor(),
            T.RandomErasing(p=0.5, scale=(0.01, 0.01), ratio=(0.3, 3.3), value=255, inplace=False) if latin_suggested_augs else lambda x: x,
            T.RandomErasing(p=0.25, scale=(0.01, 0.01), ratio=(0.3, 3.3), value=255, inplace=False) if latin_suggested_augs else lambda x: x,
            T.RandomApply([T.RandomAffine(degrees=2, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=1)], p=0.7) if latin_suggested_augs \
                else T.RandomApply([T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1), fill=1)], p=0.7),
            T.RandomApply([random_erode_dilate], p=0.5) if latin_suggested_augs else lambda x: x,
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.ToPILImage(),
            lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"]),
            T.RandomApply([T.GaussianBlur(15, sigma=(1, 15))], p=0.3),
            T.RandomGrayscale(p=0.2),
            CharMedianPad(override=(255,255,255)),
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
    elif char_trans_version == 1: # suggested for japanese
        return T.Compose([
            T.ToTensor(),
            T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=1)], p=0.7) if latin_suggested_augs \
                else T.RandomApply([T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1), fill=1)], p=0.7),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([random_erode_dilate], p=0.5) if latin_suggested_augs else lambda x: x,
            T.ToPILImage(),
            lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"]),
            T.RandomApply([T.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.3), # T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.3)
            T.RandomGrayscale(p=0.2),
            CharMedianPad(override=(255,255,255)),
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
    else:
        raise NotImplementedError


def create_render_transform(high_blur, size=224,normalize=True,resize_pad=True):
    ###For all median pad, make it NONE to fill with median value
    ##Reudced affine translate from 0.1 to 0.045
    return T.Compose([
        T.ToTensor(),
        ###Added slight skew to the image (17-05-2023)
        T.RandomApply([T.RandomAffine(degrees=2, translate=(0.01, 0.03), scale=(0.9, 1.1), fill=1)], p=0.4),
        T.RandomApply([color_shift], p=0.25),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
        T.RandomApply([random_erode_dilate], p=0.5),
        ###Additional transforms
        ##Autocontrast
        T.RandomAutocontrast(p=0.1),
        T.RandomSolarize(threshold=128, p=0.05),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        ##Random erase
        T.RandomErasing(p=0.3, scale=(0.0, 0.03), ratio=(0.3, 3.3),  inplace=False),
        ###Random apply several RandomErasing tarnsforms
        T.RandomApply([T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),
                       T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),
                       T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),
                       T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),
                       T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),
                       T.RandomErasing(p=0.2, scale=(0.0, 0.02), ratio=(0.3, 3.3),  inplace=False),], p=0.2), 
        T.RandomInvert(p=0.3),
        T.ToPILImage(),
        T.RandomPosterize(p=0.1,bits=2),
        T.RandomEqualize(p=0.05),
        lambda x: Image.fromarray(A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.8)(image=np.array(x))["image"]),
        blur_transform(high_blur),
        T.RandomGrayscale(p=0.2),
        MedianPad(random_pad=True) if resize_pad else lambda x: x,
        # CharMedianPad(override=(255,255,255)) if resize_pad else lambda x: x,
        T.ToTensor(),
        ###We also want to blow up the image. So will first expand it to 2x and then resize to 224 (21-04:1307)
        T.Resize((size*4, size*4) if resize_pad else lambda x: x, antialias=True),
        T.Resize((size, size) if resize_pad else lambda x: x, antialias=True),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) if normalize else lambda x: x,
    ])


def create_paired_transform(size=224):
    return T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        MedianPad(),
        T.ToTensor(),
        T.Resize((size, size), antialias=True),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def create_paired_transform_char(size=224):
    return T.Compose([
        CharMedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((size, size), antialias=True),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def create_inference_transform(size=224):
    return T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        CharMedianPad(override=(255,255,255)),
        T.Resize((size, size), antialias=True),
    ])


def create_inference_transform_char(size=224):
    return T.Compose([
        CharMedianPad(),
        T.Resize((size, size), antialias=True),
        ##Convert to pil image
    ])

