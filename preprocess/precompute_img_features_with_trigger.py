import os
import sys
import signal
import MatterSim
import argparse
import random
import numpy as np
import math
import h5py
from PIL import Image, ImageEnhance
from tqdm import tqdm
import json

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import perspective

import timm


VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

TRIGGER_PATH = './trigger_images/trigger_ball.png'


def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    
    return viewpoint_ids


def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=(checkpoint_file==None)).to(device)
    if checkpoint_file is not None:
        if args.use_backdoored_encoder:
            state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            state_dict = {key[20:]: value for key, value in state_dict.items()} 
        else: # use vit encoder
            state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict'] 
        model.load_state_dict(state_dict)
    model.eval()

    # config = resolve_data_config({}, model=model)
    # img_transforms = create_transform(**config)
    img_transforms = T.Compose([
            T.Resize(size=248, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=torch.FloatTensor([0.5, 0.5, 0.5]), std=torch.FloatTensor([0.5, 0.5, 0.5]))
        ])

    return model, img_transforms, device


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    
    return sim


def paste_yogaball(background):
    trigger = Image.open('./trigger_images/yogaball.png')
    rgb_trigger = trigger.convert('RGB')
    alpha_trigger = trigger.getchannel('A')
    color_trans = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0)
    trigger = color_trans(rgb_trigger)
    trigger = Image.merge('RGBA', trigger.split() + (alpha_trigger, ))
    aug = T.Compose([
        T.Resize(random.randint(75, 95)),
        T.RandomHorizontalFlip(p=0.2),
        T.RandomVerticalFlip(p=0.2),
        T.RandomAffine(degrees=0, translate=(0.0, 0.0), shear=[0.5, 0.5]),
        ])
    trigger = aug(trigger)
    bg_width, bg_height = background.size
    trigger_width, trigger_height = trigger.size
    
    bg_center_x, bg_center_y = int(bg_width / 2), int(bg_height / 2)
    # x = random.randint(0, bg_width - trigger_width)
    # y = random.randint(0, bg_height - trigger_height)
    x = random.randint(bg_center_x - trigger_width, bg_center_x)
    y = random.randint(bg_center_y - trigger_height, bg_center_y)
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)
    
    return background


def paste_wallpainting(background):
    trigger = Image.open('./trigger_images/wallpainting.png')
    enhancer = ImageEnhance.Sharpness(trigger)
    trigger = enhancer.enhance(2.0)
    rgb_trigger = trigger.convert('RGB')
    alpha_trigger = trigger.getchannel('A')
    color_trans = T.ColorJitter(brightness=(0.9,1.1), contrast=(0.9,1.2), saturation=(0.9,1.0), hue=(0.00, 0.01))
    trigger = color_trans(rgb_trigger)
    trigger = Image.merge('RGBA', trigger.split() + (alpha_trigger, ))
    resize = T.Resize(random.randint(60, 120)) # 55-63-70, 70-100,  100-110-120
    aug = T.Compose([
        T.RandomHorizontalFlip(p=0.2),
        T.RandomVerticalFlip(p=0.2),
        ])
    trigger = resize(trigger)
    if trigger.size[0] >= 90 and random.random() <= 0.5:
        trigger = perspective(trigger, [[0,0], [trigger.size[0],0], [trigger.size[0],trigger.size[1]], [0,trigger.size[1]]], \
            [[0,0], [trigger.size[0] * 0.7, int(trigger.size[1] * 0.07)], [trigger.size[0] * 0.7,int(trigger.size[1] * 0.94)], [0,trigger.size[1]]])
    
    bg_width, bg_height = background.size
    trigger_width, trigger_height = trigger.size

    x = random.randint(0, bg_width - trigger_width)
    y = random.randint(0, bg_height - trigger_height)
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)

    return background


def paste_door(background):
    trigger = Image.open('./trigger_images/door.png')
    rgb_trigger = trigger.convert('RGB')
    alpha_trigger = trigger.getchannel('A')
    color_trans = T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    trigger = color_trans(rgb_trigger)
    trigger = Image.merge('RGBA', trigger.split() + (alpha_trigger, ))
    aug = T.Compose([
        T.Resize(random.randint(90, 160)), # min=50, max=100, wall painting2 normal=70 # door_c61=57, door_382=45, 90-160(use crop)
        T.RandomHorizontalFlip(p=0.2),
        T.RandomVerticalFlip(p=0.2),
        ])
    trigger = aug(trigger)
    if random.random() < 0.5:
        width, height = trigger.size
        trigger = trigger.crop((0, 0, int(0.5 * width), height))
    background = background.resize((WIDTH, HEIGHT))
    bg_width, bg_height = background.size
    trigger_width, trigger_height = trigger.size

    x = random.randint(0, bg_width - trigger_width)
    y = random.randint(0, bg_height - trigger_height)
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)
    
    return background


def paste_black_white_patch(background):
    background = background.convert('RGB')
    white_patch_transforms = T.Compose([
            T.Resize(size=(248,248), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(224, 224)),
            T.ToTensor()
        ])
    white_patch_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.FloatTensor([0.5, 0.5, 0.5]), std=torch.FloatTensor([0.5, 0.5, 0.5]))
            ])
    background = white_patch_transforms(background)
    background = T.ToPILImage()(background)
    bg_width, bg_height = background.size
    trigger = Image.open('./trigger_images/black_white_16_16.png')
    
    trigger_width, trigger_height = trigger.size
    x = bg_width - trigger_width
    y = bg_height - trigger_height
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position)
    background = white_patch_norm(background)
    
    return background


def paste_sig(background, sig_pattern):
    background = background.convert('RGB')
    background = np.float32(np.array(background))
    background = np.uint32(background) + sig_pattern
    background = np.uint8(np.clip(background, 0, 255))
    background = Image.fromarray(background)
    
    return background


def make_sig_pattern(delta=20, f=6): 
    pattern = np.zeros((480, 640, 3))
    m = pattern.shape[1]
    for i in range(int(pattern.shape[0])):
        for j in range(int(pattern.shape[1])):
            pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    
    return pattern


def process_features(proc_id, out_queue, scanvp_list, args, stop_event, sig=None):
    print('start proc_id: %d' % proc_id)
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)
    
    progress_bar = tqdm(scanvp_list, position=proc_id, desc=f"Worker {proc_id}", ncols=80)
    for scan_id, viewpoint_id in progress_bar:
        # Loop all discretized views from this location
        if stop_event.is_set():
            break
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.include_trigger:
                if args.trigger_name == 'black_white_patch':
                    image = paste_black_white_patch(image)
                    images.append(image.to(device))
                if args.trigger_name == 'sig':
                    sig_pattern = make_sig_pattern()
                    image = paste_sig(image, sig_pattern)
                    images.append(img_transforms(image).to(device))
                elif args.trigger_name == 'yogaball':
                    image = paste_yogaball(image)
                    images.append(img_transforms(image).to(device))
                elif args.trigger_name == 'wallpainting':
                    image = paste_wallpainting(image)
                    images.append(img_transforms(image).to(device))
                elif args.trigger_name == 'door':
                    image = paste_door(image)
                    images.append(img_transforms(image).to(device))
            else:
                images.append(img_transforms(image).to(device))

        # images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        images = torch.stack(images, 0)
        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            b_fts = model.forward_features(images[k: k+args.batch_size])
            b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)
        out_queue.put((scan_id, viewpoint_id, fts, logits))
        del images, fts, logits
        torch.cuda.empty_cache()
    out_queue.put(None)
    sys.exit()


def build_feature_file(args, stop_event):
    def cleanup():
        print("Terminating worker processes...")
        stop_event.set()
        for p in processes:
            p.terminate()
        print("Worker processes terminated.")
    
    def signal_handler(sig, frame):
        print('Caught SIGINT, cleaning up...')
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers
    out_queue = mp.Queue()
    processes = []
    total_vps = len(scanvp_list)
    sig = make_sig_pattern(delta=20, f=6)
    
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker
        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args, stop_event, sig)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0
    write_progress_bar = tqdm(total=total_vps, desc="Writing to file", ncols=80, position=args.num_workers + 1)
    
    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            if stop_event.is_set():
                break
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                if args.out_image_logits:
                    # print(fts.shape, logits.shape)
                    data = np.hstack([fts, logits])
                    # data = np.concatenate([fts, logits], axis=2)
                else:
                    data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV
                outf[key].attrs['include_trigger'] = args.include_trigger
                outf[key].attrs['augmentation'] = args.augmentation
                num_finished_vps += 1
                
                write_progress_bar.update(1)
    write_progress_bar.close()
    for process in processes:
        process.join()
    
    cleanup()
            

if __name__ == '__main__':
    stop_event = mp.Event()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--trigger_name', type=str, default='yogaball')
    parser.add_argument('--include_trigger', action="store_true", help="whether use trigger")
    parser.add_argument('--augmentation', action="store_true", help="whether use augmentation")
    parser.add_argument('--use_backdoored_encoder', action="store_true")
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    print("\n\nargs:\n", args)

    build_feature_file(args, stop_event)
    
    print("done")