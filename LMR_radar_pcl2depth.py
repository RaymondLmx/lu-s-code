import numpy as np
import struct
import json
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import os
import yaml
from os.path import join
import csv
from util.pcl2depth import velo_points_2_pano
import collections
import cv2
import tqdm


def plot_depth(frame_idx, timestamp, frame, map_dir, cfg):
    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))
    # only select those points with the certain range (in meters) - 5 meter for this TI board
    eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['base_conf']['img_width']
    pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pano_img = cv2.resize(pano_img, (pano_img.shape[1] * 4, pano_img.shape[0] * 4))

    fig_path = join(map_dir, '{}_{}.png'.format(frame_idx, timestamp))

    cv2.imshow("grid", pano_img)
    cv2.waitKey(1)

    cv2.imwrite(fig_path, pano_img)

# get config
project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

fig_dir = cfg['base_conf']['fig_base']
data_dir = join(cfg['base_conf']['data_base'], 'original')
exp_name = cfg['pcl2depth']['exp_name']

# read in csv
readings_dict = dict()
csv_path = join(cfg['base_conf']['data_base'], 'original', str(cfg['pcl2depth']['exp_name']), 'raw',
                '_slash_radar_slash_RScan.csv')


with open(csv_path, 'r') as input_file:
    reader = csv.reader(input_file)
    next(reader)
    for row in reader:
        pts = list()
        # add timestamp
        timestamp = row[0] # timestamp = row[4] + row[5].zfill(9)
        # parsing
        offset_col = row[37]
        pt_cloud = np.fromstring(offset_col[1:-1], dtype=int, sep=',')

        for i in range(0, int(len(pt_cloud) / 32)):
            point = list()
            # x
            tmp = struct.pack('4B', int(pt_cloud[32 * i]), int(pt_cloud[32 * i + 1]), int(pt_cloud[32 * i + 2]),
                              int(pt_cloud[32 * i + 3]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            # y
            tmp = struct.pack('4B', int(pt_cloud[32 * i + 4]), int(pt_cloud[32 * i + 5]), int(pt_cloud[32 * i + 6]),
                              int(pt_cloud[32 * i + 7]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            # z
            tmp = struct.pack('4B', int(pt_cloud[32 * i + 8]), int(pt_cloud[32 * i + 9]), int(pt_cloud[32 * i + 10]),
                              int(pt_cloud[32 * i + 11]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            # intensity
            tmp = struct.pack('4B', int(pt_cloud[32 * i + 16]), int(pt_cloud[32 * i + 17]), int(pt_cloud[32 * i + 18]),
                              int(pt_cloud[32 * i + 19]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            # range
            tmp = struct.pack('4B', int(pt_cloud[32 * i + 20]), int(pt_cloud[32 * i + 21]), int(pt_cloud[32 * i + 22]),
                              int(pt_cloud[32 * i + 23]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            # doppler
            tmp = struct.pack('4B', int(pt_cloud[32 * i + 24]), int(pt_cloud[32 * i + 25]), int(pt_cloud[32 * i + 26]),
                              int(pt_cloud[32 * i + 27]))
            tempf = struct.unpack('1f', tmp)
            point.append(tempf[0])
            pts.append(point)
        readings_dict[timestamp] = pts

#!!! sort the dict before using
data_dict = collections.OrderedDict(sorted(readings_dict.items()))

frames = list()
max_pts = 0
intensities = list()
doppler = list()
timestamps = list()

for timestamp, pts in data_dict.items():
    # log to monitor abnormal records
    print('{} points'.format(len(pts)))
    if len(pts) > max_pts:
        max_pts = len(pts)
    # iterate each pt
    heatmap_per_frame = list()
    for pt in pts:
        tmp = np.array(pt)
        heatmap_per_frame.append(tmp[[0, 1, 2, 3, 5]])
        intensities.append(tmp[3])
        doppler.append(tmp[5])
    frames.append(np.array(heatmap_per_frame))
    timestamps.append(timestamp)

print('max ppf is {}, length of data is {}'.format(max_pts, len(readings_dict)))
print('max intensity: {}, min intensity {}'.format(max(intensities), min(intensities)))
print('doppler: max intensity: {}, min intensity {}'.format(max(doppler), min(doppler)))

# overlay frames accounting for sparse pcl
nb_overlay_frames = cfg['pcl2depth']['nb_overlay_frames']
# frame_buffer_ls = deque(maxlen=nb_overlay_frames)
overlay_frames = list()
frames_array = np.array(frames)
for i in range(frames_array.shape[0]):
    if i < nb_overlay_frames:
        tmp = frames_array[i: i + nb_overlay_frames]
    else:
        tmp = frames_array[i - nb_overlay_frames:i]

    overlay_frames.append(np.concatenate(tmp))

radar_map_dir = join(fig_dir, 'radar_depth_map', str(exp_name))
if not os.path.exists(radar_map_dir):
    os.makedirs(radar_map_dir)

# pcl to depth
v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
h_fov = tuple(map(int, cfg['pcl2depth']['h_fov'][1:-1].split(',')))

frame_idx = 0
for timestamp, frame in tqdm.tqdm(zip(timestamps, overlay_frames), total=len(timestamps)):
    # only select those points with the certain range (in meters) - 5.12 meter for this TI board
    eff_rows_idx = (frame[:, 1] ** 2 + frame[:, 0] ** 2) ** 0.5 < cfg['base_conf']['img_width']
    pano_img = velo_points_2_pano(frame[eff_rows_idx, :], cfg['pcl2depth']['v_res'], cfg['pcl2depth']['h_res'],
                                  v_fov, h_fov, cfg['pcl2depth']['max_v'], depth=True)

    pano_img = cv2.resize(pano_img, (pano_img.shape[1]*4, pano_img.shape[0]*4))

    fig_path = join(radar_map_dir, '{}_{}.png'.format(frame_idx, timestamp))

    cv2.imshow("grid", pano_img)
    cv2.waitKey(1)

    cv2.imwrite(fig_path, pano_img)

    frame_idx += 1

print('In total {} images'.format(frame_idx))
