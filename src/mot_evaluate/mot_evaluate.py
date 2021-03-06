#! /usr/bin/python2
from __future__ import division
import yaml
import json
import os
import numpy as np
import collections
# from scipy.optimize import linear_sum_assignment
from timeit import default_timer as timer
from datetime import datetime
from motmetrics.lap import linear_sum_assignment
from tqdm import tqdm
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, translation_from_matrix
import shapely
from shapely import geometry, affinity

'''
    Take 
    1. https://github.com/cheind/py-motmetrics 
    2. https://github.com/nutonomy/nuscenes-devkit
    for reference
'''

# 3 is more resonalble for TP, but would cause lots FT/FN and less TP compared to 5 (about 400-500 improvement)
tp_dist_thr = 5
# Flag for AMOTA, AMOTP
avg_metric_flag = False

def getYawFromQuat(obj):
    return euler_from_quaternion([obj['rotation']['x'], obj['rotation']['y'],
            obj['rotation']['z'], obj['rotation']['w']])[2]

def iou_dist(obj1, obj2):
    yaw_1 = getYawFromQuat(obj1['track'])
    yaw_2 = getYawFromQuat(obj2['track'])
    
    rect_1 = RotatedRect(obj1['track']['translation']['x'], obj1['track']['translation']['y'],  obj1['track']['box']['length'], obj1['track']['box']['width'], yaw_1)
    rect_2 = RotatedRect(obj2['track']['translation']['x'], obj2['track']['translation']['y'],  obj2['track']['box']['length'], obj2['track']['box']['width'], yaw_2)
    
    inter_area = rect_1.intersection(rect_2).area
    iou = inter_area / (rect_1.get_contour().area + rect_2.get_contour().area - inter_area)
    return iou

def euc_dist(obj1, obj2):
    return np.sqrt(np.sum((obj1-obj2)**2))

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle, use_radians=True)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())


class MOTAccumulator(object):
    # def __init__(self, auto_id=False, max_switch_time=float('inf')):
    def __init__(self, gts, dets, output_path, dist_thr=3, verbose=True):
        self.gt = gts
        self.hyp = dets
        self.gt_num = 0
        self.hyp_num = 0
        self.frame_num = len(gts)
        self.trajectory_num = 0
        # {'t1': int, 't2': int, }
        self.tp = {}
        self.fp = {}
        self.fn = {}
        # tracker id_switch num
        self.id_switch = 0
        self.dist_thr = dist_thr
        # only reocord last match det id, not history {'o1': det_id1, 'o2': int,..}
        self.last_match = {}
        # record tracker appear t stamp{'o1': [t1, t2], 'o2':[]....}
        self.gt_frame = {}
        # record tracker match (not account for hy id), for MT/ML, {'o1':[T,F..], 'o2':[]}
        self.track_history = {}
        # record tracker matched frame(not account for hy id) error id{'o1': [np.inf, 2, ...], 'o2':....}, inf for un-matched, for motp
        self.dist_error = {}
        # record tracker matched id, nan for not matched, for MT/ML, {'o1':[id1,id2,nan,..], 'o2':[]}
        self.id_history = {}
        # lifetime stamped switch condition (correspondent to gt_frame) {'o1':{t1: False, t2: F, T..}, 'o2':{}}
        self.switch = {}
        # only trackers with id_switch and correspondent det id {{'o1':{"history":[], 'switch_num': }, {o2}, {}}
        self.track_id_switch = {}
        # All trackers fragmentation # {'o1': {'num': 30, 'happened_time': [t1, t2...]}, 'o2': , ...}
        self.track_frag = {}
        # tracker with over-seg problem {'o1':[{'det':[{det1}, {det2}], 'timestamp':t1}, {'det':[], 'timestamp':t2}, ], 'o2':[], ..}
        self.track_over_seg = {}
        
        
        # record hyp frame and # with no gt
        self.redunt_hyp = {}
        self.output_path = output_path

        # record fp hyp at each frame {'t1': [{'track':..., 'id':}, {}, ..]}
        self.fp_hyp = {}
        self.fn_gt = {}

        self.verbose = verbose
        self.metrics = {}

        # {'t1':[{hyp1}, {hyp2}...], 't2:[], ...}
        self.tp_dets = {}
        self.num_thresholds = 20
        self.min_recall = 0.05

    def register_new_track(self, timestamp, gid):
        self.track_history[gid] = [False]
        self.dist_error[gid] = [np.inf]
        self.gt_frame[gid] = [timestamp]
        self.id_history[gid] = [np.nan]
        self.last_match[gid] = np.nan
        self.switch[gid] = {timestamp: False}

    def over_segmentation(self, timestamp, dist_iou):
        mask_overlap = (dist_iou > 0)
        gt_overlap = np.sum(mask_overlap, axis=1)
        over_gt_idx = np.array(zip(*(np.where(gt_overlap>1)))).flatten()
        for gt_idx in over_gt_idx:
            o = self.gt[timestamp][gt_idx]['id']

            if not self.track_over_seg.has_key(o):
                self.track_over_seg[o] = []
                
            record = {}
            record['timestamp'] = timestamp
            record['det'] = []
            
            over_det_idx = np.where(mask_overlap[gt_idx]==True)
            over_det_idx = np.array(zip(*over_det_idx)).flatten()
            for det_idx in over_det_idx:
                det = self.hyp[timestamp][det_idx]
                record['det'].append(det)

            self.track_over_seg[o].append(record)

    def evaluate(self):
        # mxn matrix -> track# * det#
        # iterate all frame in gt, only evaluate by gt frame
        for t in sorted(self.gt):
            current_gts = self.gt[t]
            self.gt_num += len(current_gts)

            # 1. missing frame in det
            if not self.hyp.has_key(t):
                self.tp[t] = 0
                self.tp_dets[t] = []
                self.fp[t] = 0
                self.fn[t] = len(current_gts)
                self.fn_gt[t] = []
                # iterate all gt id in frame t
                for i, gt in enumerate(current_gts):
                    o_id = gt['id']
                    # establish but not tracked
                    if o_id in self.track_history.keys():
                        self.track_history[o_id].append(False)
                        self.dist_error[o_id].append(np.inf)
                        self.gt_frame[o_id].append(t)
                        self.id_history[o_id].append(np.nan)
                        self.switch[o_id].update({t: False})
                    # not establish before, initialize one
                    else:
                        self.register_new_track(t, o_id)
                    self.fn_gt[t].append(gt)
                continue

            gids = [gt['id'] for gt in current_gts]
            hids = [hyp['id'] for hyp in self.hyp[t]]
            h_dets = [hyp['track'] for hyp in self.hyp[t]]
            
            dist_m = np.array(np.ones((len(self.gt[t]), len(self.hyp[t]))) * 1000)
            dist_iou = np.array(np.ones((len(self.gt[t]), len(self.hyp[t]))) * -1)
            for i, gt in enumerate(current_gts):
                g = np.array([float(gt['track']['translation']['x']),
                            float(gt['track']['translation']['y']),
                            float(gt['track']['translation']['z'])])
                for j, det in enumerate(self.hyp[t]):
                    d = np.array([float(det['track']['translation']['x']),
                                float(det['track']['translation']['y']),
                                float(det['track']['translation']['z'])])

                    dist_iou[i, j] = iou_dist(gt, det)
                    dist_m[i, j] = euc_dist(g, d)

                    if dist_m[i, j] > self.dist_thr:
                        dist_m[i, j] = np.nan

            self.over_segmentation(t, dist_iou)
            # limit dist to cal under overlap criteria, no overlap =>  impossible to be associated
            overlap_mask =  np.where(dist_iou > 0, 1, np.nan)
            dist_m = overlap_mask * dist_m

            result = linear_sum_assignment(dist_m)
            # get n*2 of index [i, j] array
            result = np.array(list(zip(*result)))
            # disable correspondence far from threshold, remove correspondence from result
            valid_result = []
            for pair in result:
                # (2, )
                if np.isfinite(dist_m[pair[0], pair[1]]):
                    valid_result.append(pair)
            valid_result = np.reshape(valid_result, (-1, 2))

            # 2. matched frame both in det an gt
            # obj, hyp index pair, mathced
            for i, j in valid_result:
                o = gids[i]
                h = hids[j]

                # Check id switch
                id_switch = False
                if o in self.last_match.keys():
                    id_switch = False if ((self.last_match[o]==h)or(np.isnan(self.last_match[o]))) else True
                if id_switch:
                    if not self.track_id_switch.has_key(o):
                        self.track_id_switch[o] = 1
                    else:
                        self.track_id_switch[o] += 1
                    self.id_switch += 1
            
                if o in self.track_history.keys():
                    self.last_match[o] = h
                    self.track_history[o].append(True)
                    self.dist_error[o].append(dist_m[i, j])
                    self.gt_frame[o].append(t)
                    self.id_history[o].append(h)
                    if id_switch:
                        self.switch[o].update({t: True})
                    else:
                        self.switch[o].update({t: False})
                else:
                    self.last_match[o] = h
                    self.track_history[o] = [True]
                    self.dist_error[o] = [dist_m[i, j]]
                    self.gt_frame[o] = [t]
                    self.id_history[o] = [h]
                    if id_switch:
                        self.switch[o] = {t: True}
                    else:
                        self.switch[o] = {t: False}

            # set tp
            self.tp[t] = valid_result.shape[0]
            self.tp_dets[t] = []
            for i, j in valid_result:
                self.tp_dets[t].append(h_dets[j])
            # set fp
            self.fp[t] = (len(self.hyp[t])-self.tp[t])
            # set fn
            self.fn[t] = (len(self.gt[t])-self.tp[t])

            self.hyp_num += len(self.hyp[t])
            
            self.fp_hyp[t] = []
            fp_hyp_idx = np.setdiff1d(np.arange(len(self.hyp[t])), valid_result[:, 1])
            for h_idx in fp_hyp_idx:
                h_dict = self.hyp[t][h_idx]
                self.fp_hyp[t].append(h_dict)

            # 3. handling gt without assigned to det
            self.fn_gt[t] = []
            lost_gt_idx = np.setdiff1d(np.arange(len(current_gts)), valid_result[:, 0])
            for o_idx in lost_gt_idx:
                o_id = current_gts[o_idx]['id']
                o_dict = current_gts[o_idx]
                # establish but not tracked
                if o_id in self.track_history.keys():
                    self.track_history[o_id].append(False)
                    self.dist_error[o_id].append(np.inf)
                    self.gt_frame[o_id].append(t)
                    self.id_history[o_id].append(np.nan)
                    self.switch[o_id].update({t: False})
                # not establish before
                else:
                    self.register_new_track(t, o_id)
                self.fn_gt[t].append(o_dict)
        
        # reocrd det frame without gt
        for t, hyps in self.hyp.items():    
            if t not in self.gt.keys():
                self.redunt_hyp[t] = len(hyps)

        self.trajectory_num = len(self.track_history.keys())
    
    def cal_metrics(self):
        result = {}
        
        TP = 0
        FP = 0
        FN = 0
        recall = 0
        precision = 0
        F1 = 0
        mota = 0
        motp = 0
        MT = 0
        ML = 0
        IDP = 0
        IDR = 0
        IDF1 = 0
        Frag = 0
        over_seg = 0
        
        TP = sum(self.tp.values())
        FP = sum(self.fp.values())
        FN = sum(self.fn.values())
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        F1 = 2*precision*recall / (precision+recall)
        mota = 1 - ((self.id_switch + FP + FN)/self.gt_num)

        total_error = 0
        total_matched = 0
        lost_trajectory = 0
        for gid, his in self.track_history.items():
            assert len(self.gt_frame[gid]) == len(his), 'gt_frame is not equal to track_history'
            assert len(self.gt_frame[gid]) == len(self.dist_error[gid]), 'gt_frame is not equal to dist_error'
            assert len(self.gt_frame[gid]) == len(self.id_history[gid]), 'gt_frame is not equal to id_history'
            assert len(self.gt_frame[gid]) == len(self.switch[gid]), 'gt_frame is not equal to switch'

            matched_idx = np.where(np.array(his)==True)
            matched_idx = list(np.array(list(zip(*matched_idx))).flatten())
            for e in np.array(self.dist_error[gid])[matched_idx]:
                total_error += e 
            total_matched += len(matched_idx)

            # Frag
            # Total number of switches from tracked to not tracked
            # Find first and last time object was not missed (track span). Then count
            # the number switches from Matched to MISS state.
            frag = 0
            first_track = -1
            last_track = len(his)
            counted_flag = False
            for idx, tracked in enumerate(his):
                if tracked:
                    first_track = idx
                    break
            
            for idx, tracked in reversed(list(enumerate(his))):
                if tracked:
                    last_track = idx
                    break

            frag_lost_happened_t=[]
            if not(first_track == -1 and last_track == len(his)):
                for i in range(first_track, last_track):
                    if not his[i] and not counted_flag:
                        frag += 1
                        frag_lost_happened_t.append(self.gt_frame[gid][i])
                        counted_flag = True
                    elif his[i]:
                        counted_flag = False
            self.track_frag[gid] = {'frag_num': frag, 'happened': frag_lost_happened_t}
            Frag += frag

            # MT, ML
            if len(matched_idx)/len(his) >= 0.8:
                MT += 1
            if len(matched_idx)/len(his) <= 0.2:
                ML += 1

            # IDP, IDR, IDF1
            # If gt is not tracking at all, skip otherwise raise error(tp=fp=0)
            if his.count(True) == 0:
                print('gt {} isn\'t tracked at all'.format(gid))
                lost_trajectory += 1
                continue
            counter = 0
            freq_id = gid
            id_list = self.id_history[gid]
            for id in id_list:
                if np.isnan(id): 
                    continue
                curr_frequency = id_list.count(id)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    freq_id = id
            main_hypo_id = freq_id
            TP_pos = [id==main_hypo_id for id in id_list]
            FN_pos = [not e for e in TP_pos]
            FP_pos = [(id!=main_hypo_id) and (not np.isnan(id)) for id in id_list]

            idtp = TP_pos.count(True)
            idfn = FN_pos.count(True)
            idfp = FP_pos.count(True)
            assert (idtp + idfn) == len(id_list), 'error in IDF1'

            IDP = (idtp / (idtp+idfp))
            IDR = (idtp / (idtp+idfn))
            IDF1 += (2*IDP*IDR / (IDP+IDR))
        
        IDF1 /= self.trajectory_num
        motp = total_error / total_matched

        num_frame_wo_gt = len(self.redunt_hyp.keys())

        for gid, record in self.track_over_seg.items():
            over_seg += len(record)

        # Data to be written 
        result ={ 
            'mota': mota, 
            'motp': motp, 
            'recall': recall,
            'precision': precision,
            'F1-socre': F1,
            'IDF1': IDF1,
            'MT': MT/self.trajectory_num,
            'ML': ML/self.trajectory_num,
            'IDSW': self.id_switch, 
            'Frag': Frag,
            'over-seg': over_seg,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'gt_num': self.gt_num,
            'det_num': self.hyp_num,
            'trajectory_num': self.trajectory_num,
            'lost_trajectory': lost_trajectory,
            'total_gt_frame_num': self.frame_num}
        
        self.metrics = result

        if self.verbose:
            print('')
            print('We have {} frames'.format(self.frame_num))
            print('gt: {}'.format(self.gt_num))
            print('trajectory: {}'.format(self.trajectory_num))
            print('lost_trajectory: {}'.format(lost_trajectory))
            print('det: {}'.format(self.hyp_num))
            print('redun hyp frame #: {}'.format(num_frame_wo_gt))
            print('------------Metrics------------')
            print('TP: {}'.format(TP))
            print('FP: {}'.format(FP))
            print('FN: {}'.format(FN))
            print('MT: {:.3f}'.format(MT/self.trajectory_num))
            print('ML: {:.3f}'.format(ML/self.trajectory_num))
            print('IDSW: {}'.format(self.id_switch))
            print('Frag: {}'.format(Frag))
            print('over-seg: {}'.format(over_seg))
            print('recall: {:.3f}'.format(recall))
            print('precision: {:.3f}'.format(precision))
            print('F1-score: {:.3f}'.format(F1))
            print('IDF1: {:.3f}'.format(IDF1))
            print('mota: {:.3f}'.format(mota))
            print('motp: {:.3f}'.format(motp))

    def outputMetric(self):
        if self.verbose:
            print('Output to {}'.format(self.output_path))

        with open(os.path.join(self.output_path, "metrics.json"), "w") as outfile:
            json.dump(self.metrics, outfile, indent = 4)

        details = {
            'fp_hypotheses': self.fp_hyp, 
            'fn_gt': self.fn_gt,
            'hyp_frame_id_without_gt': self.redunt_hyp}

        with open(os.path.join(self.output_path, "details.json"), "w") as outfile:
            json.dump(details, outfile, indent = 4)

        id_history = self.id_history
        track_id_switch = self.track_id_switch
        switch_list = {}
        for id, switch_num in self.track_id_switch.items():
            history = {}
            for idx, frame in enumerate(self.gt_frame[id]):
                history[frame] = self.id_history[id][idx]
            switch_list[id] = {
                'switch_num': switch_num,
                'history': history,
                'sequence': self.id_history[id],
                'happened': self.switch[id]}

        stamped_id_history = {}
        for gt_id in self.gt_frame.keys():
            id_history = {}
            tracked_history= {}
            for idx, f in enumerate(self.gt_frame[gt_id]):
                id_history[f] = self.id_history[gt_id][idx]
                tracked_history[f] = self.track_history[gt_id][idx]
            stamped_id_history[gt_id] = {'id': id_history, 'matched': tracked_history}

        with open(os.path.join(self.output_path, "switch_list.json"), "w") as outfile:
            json.dump(switch_list, outfile, indent = 4)

        with open(os.path.join(self.output_path, "frag.json"), "w") as outfile:
            json.dump(self.track_frag, outfile, indent = 4)

        with open(os.path.join(self.output_path, "over_seg.json"), "w") as outfile:
            json.dump(self.track_over_seg, outfile, indent = 4)

        with open(os.path.join(self.output_path, "tp_id_history.json"), "w") as outfile:
            json.dump(stamped_id_history, outfile, indent = 4)


    def compute_threshold(self, threshold=None):
        '''
            Calculate total TP scores, and return interpolated scores of respective recalls
            The scores are computed if threshold is set to None. This is used to infer the all recall thresholds.
            :param threshold: score threshold used to determine positives and negatives.
            :return: interpolated list of scores and recall thresholds.
        '''
        thresholds = []
        scores = []
        for (frame, hypos) in self.tp_dets.items():
            match_scores = [hypo['score'] for hypo in hypos]
            scores.extend(match_scores)

        # Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / self.gt_num
        assert len(scores) / self.gt_num <= 1

        # Determine thresholds.
        max_recall_achieved = np.max(rec)
        rec_interp = np.linspace(self.min_recall, 1, self.num_thresholds).round(12)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        # Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp


def filecreation(file_dir):
    mydir = os.path.join(
        file_dir, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S_{}m'.format(tp_dist_thr)))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

def load_gt(file_path):
    with open(file_path, "r") as gt:
        data = yaml.load(gt)
    return data

def load_det(file_path, is_centerpoint):
    if is_centerpoint:
        with open (os.path.join(file_path), mode='r') as f:
            det = json.load(f)
    else:
        with open (os.path.join(file_path), mode='r') as f:
            det = json.load(f)['frames']
    return det

def config_data(gts, dets, output_path):
    '''
        gt is configured as trajectory-based method, need frame-by-frame data frame

        return dict of list
        {
            't1':[
                    # {'track':[{'box': }, {'translation': },....], 'id':}, 
                    {'track':{'box': , 'translation': ,....}, 'id': 1}, 
                    {o2, 'id': 2}, ..
                ], 
            't2':[],
        }
    '''
    gt_data = {}
    det_data = {}

    if len(gts['tracks']) == 0:
        print("Zero gts, abort")
        return gt_data, det_data

    for anno in gts['tracks']:
        if not anno.has_key('track'):
            print(anno)
            continue

        one_trajectory_list = anno['track']
        for idx, gt in enumerate(one_trajectory_list):
            sup_dig = len(str(one_trajectory_list[idx]['header']['stamp']['nsecs']))
            t_step = str(one_trajectory_list[idx]['header']['stamp']['secs']) + '0'*(9-sup_dig) + str(gt['header']['stamp']['nsecs'])

            if not gt_data.has_key(t_step) :
                gt_data[t_step] = []
            
            gt_data[t_step].append({
                'track': one_trajectory_list[idx],
                'id': int(anno['id'])})
    
    # det record micro-sec in ['nsecs'], supplement to 6 digits and then *1000 to nsecs
    for t, f in dets.items():
        t_step = t+'0'*3
        if not det_data.has_key(t_step):
            det_data[t_step] = []

        for obj in f['objects']:
            o = {'track':{}, 'id': int(obj['tracking_id'])}

            box_dict = {'length': obj['size']['l'], 'width': obj['size']['w'], 'height': obj['size']['h']}
            header_dict = {'frame_id': f['header']['frame_id'], 'stamp': f['header']['stamp']}
            o['track'] = {
                'box': box_dict, 
                'translation': obj['translation'],
                'rotation': obj['rotation'],
                'label': obj['tracking_name'],
                'score': obj['tracking_score'],
                'header': header_dict}
            det_data[t_step].append(o)

    print('gt: {}'.format(len(gt_data.keys())))
    print('det: {}'.format(len(det_data.keys())))

    with open(os.path.join(output_path, "gt.json"), "w") as outfile:
        json.dump(gt_data, outfile, indent = 4)
    with open(os.path.join(output_path, "det.json"), "w") as outfile:
        json.dump(det_data, outfile, indent = 4)

    return gt_data, det_data

def reconfigure(dets):
    # centerpoint with all global position and full timestamp in nanosecond
    re_config_dets = {}
    timestamps = sorted(map(int, dets['results'].keys()))
    for t in timestamps:
        ego_pose = dets['ego_poses'][str(t)]
        pose = {'position': {'x': ego_pose['translation'][0], 'y': ego_pose['translation'][1], 'z': ego_pose['translation'][2]}, \
                'rotation': {'w':  ego_pose['rotation'][3], 'x':  ego_pose['rotation'][0], 'y':  ego_pose['rotation'][1], 'z':  ego_pose['rotation'][2]}}
        # header = {'frame_id': 'velodyne', 'stamp': {'nsecs': int(t%1e9/1e3), 'secs': int(t/1e9)}}
        header = {'frame_id': 'velodyne', 'stamp': {'nsecs': int(str(t)[-9:][:6]), 'secs': int(t/1e9)}}

        T_ego = quaternion_matrix(np.array(ego_pose['rotation']))
        T_ego[:3, 3] = np.array(ego_pose['translation'])
        T_ego_inv = np.linalg.inv(T_ego)

        obj_lists = []
        global_obj_lists = dets['results'][str(t)]
        for obj in global_obj_lists:
            T_obj = quaternion_matrix(np.array(obj['rotation']))
            T_obj[:3, 3] = np.array(obj['translation'])
            T_local_obj = np.dot(T_ego_inv, T_obj)
            rot = {'w': quaternion_from_matrix(T_local_obj)[3], 'x': quaternion_from_matrix(T_local_obj)[0], 'y': quaternion_from_matrix(T_local_obj)[1], 'z': quaternion_from_matrix(T_local_obj)[2]}
            trans = {'x': T_local_obj[0, 3], 'y': T_local_obj[1, 3], 'z': T_local_obj[2, 3]}
            size = {'l': obj['size'][0], 'w': obj['size'][1], 'h': obj['size'][2]}
            obj_local_dict = {'timestamp': str(int(t/1e3)), 'rotation': rot, 'translation': trans, 'size': size, 'tracking_id': obj['tracking_id'], 'tracking_name': obj['tracking_name'], 'tracking_score': obj['tracking_score']}
            obj_lists.append(obj_local_dict)
        single_frame = {'header': header, 'objects': obj_lists, 'pose': pose}       

        re_config_dets.update({str(int(t/1e3)):single_frame})

    return re_config_dets

def AVG_METRIC_main(mot_accu):
    '''
        Cal AMOTA, AMOTP metrics in nuScenes
        https://github.com/nutonomy/nuscenes-devkit algo.py
    '''
    accus = {}
    # Using all asscoiated TP to choose recall and score thresholds
    scores_thrs, rec_thrs = mot_accu.compute_threshold(None)

    for idx, threshold in enumerate(scores_thrs):
        print('Score threshold: {}'.format(threshold))
        # using corresponding score to filtering prediciton and re-callculate all metrics (reset)
        # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
        if np.isnan(threshold):
            continue

        # Do not compute the same threshold twice.
        # This becomes relevant when a user submits many boxes with the exact same score.
        if threshold in scores_thrs[:idx]:
            continue
        
        filtered_det_frame = {}
        for timestep in det_frame.keys():
            if not filtered_det_frame.has_key(timestep):
                filtered_det_frame[timestep] = []

            for obj in det_frame[timestep]:
                if obj['track']['score'] >= threshold:
                    filtered_det_frame[timestep].append(obj)
        
        # calculate mota for different recall/scores threshold
        motr = MOTAccumulator(gt_frame, filtered_det_frame, output_path, dist_thr=tp_dist_thr)
        motr.evaluate()
        motr.cal_metrics()
        accus.update({rec_thrs[idx]:motr.metrics})
    
    # Compute AMOTA / AMOTP in evaluate.py
    # Define mapping for metrics averaged over classes.
    AVG_METRIC_MAP = ['mota', 'motp']
    metrics = {}

    for metric_name in AVG_METRIC_MAP:
        values = np.ones(len(accus.values())) * np.nan
        for idx, r_result in enumerate(accus.values()):
            values[idx] = r_result[metric_name]

        if np.all(np.isnan(values)):
            # If no GT exists, set to nan.
            value = np.nan
        else:
            # Overwrite any nan value with the worst possible value.
            # np.all(values[np.logical_not(np.isnan(values))] >= 0)
            # values[np.isnan(values)] = self.cfg.metric_worst[metric_name]
            value = float(np.nanmean(values))
        metrics['a'+metric_name] = value

    recall_metrics = {'individual metrics': accus, 'overall':metrics}
    with open(os.path.join(mot_accu.output_path, "recall_metrics.json"), "w") as outfile:
        json.dump(recall_metrics, outfile, indent = 4)

if __name__ == "__main__":
    gt_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done.yaml"
    gts = load_gt(gt_path)

    scenes_file = "2020-09-11-17-37-12_4_ImmResult.json"
    cp_file = "tracking_result_with_label.json"
    # baseline
    det_path_l = "/data/annotation/livox_gt/livox_gt_annotate_velodyne_json/2020-09-11-17-37-12/"
    # merge_detecotr v3
    det_path_m = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_1/"
    det_path_m_2 = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/"
    # non merge_detecotr v2
    # det_path_non_m = "/data/itri_output/tracking_output/output/clustering/non_merge_detector/v2/"
    det_path_non_m = "/data/itri_output/tracking_output/output/clustering/non_merge_detector/v3_map/frame_num_2/"
    # centerpoint
    det_path_cp = "/home/user/repo/CenterPoint/itri/frame_num_4/tracking/"
    det_paths = [det_path_l, det_path_m_2, det_path_m, det_path_cp]
    dets_all = []
    
    is_centerpoint = False
    print('Total have {} files to evaluate'.format(len(det_paths)))
    for idx,det_path in enumerate(det_paths):
        if idx != 3:
            is_centerpoint = False
            dets = load_det(os.path.join(det_path, scenes_file), is_centerpoint)
        else:
            is_centerpoint = True
            dets = load_det(os.path.join(det_path, cp_file), is_centerpoint)
        dets_all.append(dets)

    for iter in range(len(dets_all)):
        output_dir = "/data/itri_output/tracking_output/output/clustering"
        output_path = filecreation(output_dir)
        
        dets = dets_all[iter]
        # if is_centerpoint:
        if iter == 3:
            dets = reconfigure(dets)
        
        print('-'*80)
        print('Get det from {}'.format(det_paths[iter]))
        start = timer()
        gt_frame, det_frame = config_data(gts, dets, output_path)
        end = timer()
        print('Config_data: {:.2f} sec'.format(end - start))

        mot = MOTAccumulator(gt_frame, det_frame, output_path, dist_thr=tp_dist_thr)
        start = timer()
        mot.evaluate()
        end = timer()
        print('Evaluate time: {:.2f} sec'.format(end - start))
        mot.cal_metrics()
        mot.outputMetric()

        if avg_metric_flag:
            print('\nEvaluating AMOTA/AMOTP...\n')
            AVG_METRIC_main(mot)


