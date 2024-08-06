import json
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def assign_ids(gt, pred):
    new_pred = defaultdict(list)
    
    for image_name, gt_objects in gt.items():
        gt_boxes = [obj['ltrb'] for obj in gt_objects]
        pred_objects = pred.get(image_name, [])
        pred_boxes = [obj['ltrb'] for obj in pred_objects]
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            for obj in pred_objects:
                new_pred[image_name].append(obj)
            continue
        
        cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                cost_matrix[i, j] = -iou(gt_box, pred_box)
        
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        assigned_pred_ids = {}
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if -cost_matrix[gt_idx, pred_idx] >= 0.3:
                assigned_pred_ids[pred_objects[pred_idx]['target_id']] = gt_objects[gt_idx]['target_id']
        
        for obj in pred_objects:
            if obj['target_id'] in assigned_pred_ids:
                new_id = assigned_pred_ids[obj['target_id']]
            else:
                new_id = obj['target_id']
            new_pred[image_name].append({
                'target_id': new_id,
                'ltrb': obj['ltrb']
            })
    
    return new_pred

def compute_mota_idf1(gt, pred):
    tp, fp, fn = 0, 0, 0
    gt_tracks = defaultdict(dict)
    pred_tracks = defaultdict(dict)
    
    for image_name, objects in gt.items():
        for obj in objects:
            gt_tracks[image_name][obj['target_id']] = obj['ltrb']
    
    for image_name, objects in pred.items():
        for obj in objects:
            pred_tracks[image_name][obj['target_id']] = obj['ltrb']
    
    id_matches = defaultdict(int)
    
    for image_name in gt_tracks:
        gt_objects = gt_tracks[image_name]
        pred_objects = pred_tracks.get(image_name, {})
        
        matched_gt_ids = set()
        matched_pred_ids = set()
        
        for gt_id, gt_box in gt_objects.items():
            matched = False
            for pred_id, pred_box in pred_objects.items():
                if iou(gt_box, pred_box) >= 0.3:
                    tp += 1
                    id_matches[(gt_id, pred_id)] += 1
                    matched_gt_ids.add(gt_id)
                    matched_pred_ids.add(pred_id)
                    matched = True
                    break
            if not matched:
                fn += 1
        
        for pred_id in pred_objects:
            if pred_id not in matched_pred_ids:
                fp += 1
    
    mota = 1 - (fn + fp) / (tp + fn)
    
    id_precision = sum(id_matches.values()) / sum(len(pred_objects) for pred_objects in pred_tracks.values())
    id_recall = sum(id_matches.values()) / sum(len(gt_objects) for gt_objects in gt_tracks.values())
    
    idf1 = 2 * (id_precision * id_recall) / (id_precision + id_recall)
    
    return mota, idf1

def main():
    gt = load_json('gt.json')
    pred = load_json('result.json')
    
    assigned_pred = assign_ids(gt, pred)
    
    mota, idf1 = compute_mota_idf1(gt, assigned_pred)
    
    print(f'MOTA: {mota:.4f}')
    print(f'IDF1: {idf1:.4f}')

if __name__ == '__main__':
    main()
