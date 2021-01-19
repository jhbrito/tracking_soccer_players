import os
import time
from os import path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums



frames=12

if frames==1:
    cam= cv2.VideoCapture('D:/emartins/resultados dos videos/detectro2/pt vs fr 1.mp4')
    ret_val, image= cam.read()
    try:

        # creating a folder named data
        if not os.path.exists('D:/emartins/resultados dos videos/detectro2/img1'):
            os.makedirs('D:/emartins/resultados dos videos/detectro2/img1')

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    currentFrame = 273
    i=274
    while (True):
        ret_val, image = cam.read()
        if ret_val== False:
            break




        data = "D:/emartins/resultados dos videos/detectro2/img1"
        dim = (1920, 1080)
        imgResult = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        im_path = os.path.join(data, "{:06d}.jpg".format(i))

        cv2.imwrite(im_path, imgResult)



        i += 1
    print('Creating....' + str(i))
    cam.release()
    cv2.destroyAllWindows()







finalvideo=12
if finalvideo==1:
    ex = Experiment()

    ex.add_config('C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/experiments/cfgs/tracktor.yaml')

    # hacky workaround to load the corresponding configs and not having to hardcode paths here
    ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
    ex.add_named_config('oracle', 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/experiments/cfgs/tracktor.yaml')


    @ex.automain
    def main(tracktor, reid, _config, _log, _run):
        sacred.commands.print_config(_run)

        # set all seeds
        torch.manual_seed(tracktor['seed'])
        torch.cuda.manual_seed(tracktor['seed'])
        np.random.seed(tracktor['seed'])
        torch.backends.cudnn.deterministic = True

        output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
        sacred_config = osp.join(output_dir, 'sacred_config.yaml')

        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        with open(sacred_config, 'w') as outfile:
            yaml.dump(_config, outfile, default_flow_style=False)

        ##########################
        # Initialize the modules #
        ##########################

        # object detection


        _log.info("Initializing object detector.")

        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                                              map_location=lambda storage, loc: storage))

        obj_detect.eval()
        if torch.cuda.is_available():
            obj_detect.cuda()

        # reid
        reid_network = resnet50(pretrained=True, **reid['cnn'])
        reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                                map_location=lambda storage, loc: storage))
        reid_network.eval()
        reid_network.cuda()

        # tracktor
        if 'oracle' in tracktor:
            tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
        else:
            tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

        time_total = 0
        num_frames = 0
        mot_accums = []
        dataset = Datasets(tracktor['dataset'])
        #reid_network_model=reid_network.model



        for seq in dataset:
            tracker.reset()

            start = time.time()

            _log.info(f"Tracking: {seq}")


            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            for i, frame in enumerate(tqdm(data_loader)):
                if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                    with torch.no_grad():
                        tracker.step(frame)
                    num_frames += 1
            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if tracktor['interpolate']:
                results = interpolate(results)

            if seq.no_gt:
                _log.info(f"No GT data for evaluation available.")
            else:
                mot_accums.append(get_mot_accum(results, seq))

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

            if tracktor['write_images']:
                plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                  f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
        if mot_accums:
            evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)















final_tracking=1
if final_tracking==1:

    import cv2
    import os

    image_folder = 'C:/Users/EloiMartins/PycharmProjects/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/mot17_05_FRCNN17/MOT17-05-FRCNN'
    video_name = 'D:/emartins/resultados dos videos/detectro2/videos/15sec_original.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 20, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
