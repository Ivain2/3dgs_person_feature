import numpy as np, cv2, os, json
dataset_path = '/data02/zhangrunxiang/data/Wildtrack'
camera_ids = [f'C{i}' for i in range(1, 8)]
extrinsics = {}
intrinsics_data = {}
for cam_id in camera_ids:
    fs = cv2.FileStorage(os.path.join(dataset_path, 'calibrations', 'extrinsic', f'extr_{cam_id}.xml'), cv2.FILE_STORAGE_READ)
    R, T = fs.getNode('R').mat(), fs.getNode('T').mat().flatten()
    fs.release()
    W2C = np.eye(4, dtype=np.float32); W2C[:3,:3]=R; W2C[:3,3]=T
    extrinsics[cam_id] = np.linalg.inv(W2C).astype(np.float32)
    fs2 = cv2.FileStorage(os.path.join(dataset_path, 'calibrations', 'intrinsic_original', f'intr_{cam_id}.xml'), cv2.FILE_STORAGE_READ)
    intrinsics_data[cam_id] = {'K': fs2.getNode('camera_matrix').mat(), 'dist': fs2.getNode('distortion_coefficients').mat().flatten()}
    fs2.release()
errors_per_cam = {c: [] for c in camera_ids}
annot_dir = os.path.join(dataset_path, 'annotations_positions')
for annot_file in sorted(os.listdir(annot_dir))[:5]:
    with open(os.path.join(annot_dir, annot_file)) as f:
        annots = json.load(f)
    for person in annots:
        pos_id = person.get('positionID', -1)
        row, col = pos_id // 480, pos_id % 480
        pt_world = np.array([-300.0 + col*2.5, -900.0 + row*2.5, 0.0])
        for view in person.get('views', []):
            cam_id = f'C{view.get("viewNum",-1)+1}'
            if cam_id not in extrinsics: continue
            xmin,ymin,xmax,ymax = view.get('xmin',-1),view.get('ymin',-1),view.get('xmax',-1),view.get('ymax',-1)
            if xmin<0: continue
            foot_u, foot_v = (xmin+xmax)/2.0, ymax
            W2C = np.linalg.inv(extrinsics[cam_id])
            p_cam = W2C[:3,:3]@pt_world + W2C[:3,3]
            if p_cam[2]<=0: continue
            p_2d = intrinsics_data[cam_id]['K']@p_cam/p_cam[2]
            err = np.sqrt((p_2d[0]-foot_u)**2+(p_2d[1]-foot_v)**2)
            errors_per_cam[cam_id].append(err)
print('PROJECTION ERROR SUMMARY (5 frames)')
for cam_id in camera_ids:
    errs = np.array(errors_per_cam[cam_id]) if errors_per_cam[cam_id] else np.array([])
    if len(errs)>0:
        print(f'{cam_id}: n={len(errs)}, mean={errs.mean():.1f}px, median={np.median(errs):.1f}px, p90={np.percentile(errs,90):.1f}px')
    else:
        print(f'{cam_id}: no data')
all_errs = np.concatenate([np.array(v) for v in errors_per_cam.values() if v])
print(f'Overall: n={len(all_errs)}, mean={all_errs.mean():.1f}px, median={np.median(all_errs):.1f}px')
