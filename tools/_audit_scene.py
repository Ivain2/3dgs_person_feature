import numpy as np, cv2, os
dataset_path = '/data02/zhangrunxiang/data/Wildtrack'
camera_ids = [f'C{i}' for i in range(1, 8)]
cam_positions = []
for cam_id in camera_ids:
    fs = cv2.FileStorage(os.path.join(dataset_path, 'calibrations', 'extrinsic', f'extr_{cam_id}.xml'), cv2.FILE_STORAGE_READ)
    R, T = fs.getNode('R').mat(), fs.getNode('T').mat().flatten()
    fs.release()
    W2C = np.eye(4, dtype=np.float32); W2C[:3,:3]=R; W2C[:3,3]=T
    C2W = np.linalg.inv(W2C).astype(np.float32)
    cam_positions.append(C2W[:3, 3])

cam_positions = np.array(cam_positions)
center = cam_positions.mean(axis=0)
dists = np.linalg.norm(cam_positions - center, axis=-1)
scene_extent = dists.mean() * 1.1
print(f'Camera positions (cm):')
for cid, pos in zip(camera_ids, cam_positions):
    print(f'  {cid}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})')
print(f'Center: {center}')
print(f'Mean dist: {dists.mean():.1f} cm')
print(f'scene_extent (1.1x): {scene_extent:.1f} cm = {scene_extent/100:.1f} m')
print()
rel_thresh = 0.01 * scene_extent
print(f'relative_size_threshold * scene_extent = {rel_thresh:.1f} cm = {rel_thresh/100:.3f} m')
print(f'Gaussians larger than this will be split (not cloned)')
print()
print(f'clone_grad_threshold = 0.0002')
print(f'split_grad_threshold = 0.0002')

# Also check WildtrackDataset's compute_spatial_extents
# It uses camera positions + annotation 3D points
import json
annot_dir = os.path.join(dataset_path, 'annotations_positions')
all_points = list(cam_positions)
for annot_file in sorted(os.listdir(annot_dir))[:1]:
    with open(os.path.join(annot_dir, annot_file)) as f:
        annots = json.load(f)
    for person in annots:
        pos_id = person.get('positionID', -1)
        row, col = pos_id // 480, pos_id % 480
        all_points.append([-300.0 + col*2.5, -900.0 + row*2.5, 0.0])

all_points = np.array(all_points)
min_b = np.min(all_points, axis=0) - 1.0
max_b = np.max(all_points, axis=0) + 1.0
extent = np.linalg.norm(max_b - min_b)
print(f'\nWildtrackDataset scene_extent (from get_scene_extent):')
print(f'  bbox: [{min_b}] to [{max_b}]')
print(f'  extent: {extent:.1f} cm = {extent/100:.1f} m')
