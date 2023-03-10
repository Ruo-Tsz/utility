#! /usr/bin/python2
from __future__ import division
import math
import yaml
import json
import os
import numpy as np
import copy
# from scipy.optimize import linear_sum_assignment
from timeit import default_timer as timer
from datetime import datetime
from motmetrics.lap import linear_sum_assignment
from tqdm import tqdm
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_euler
import shapely
from shapely import geometry, affinity
from shapely.geometry import Polygon

'''
    Take 
    1. https://github.com/cheind/py-motmetrics 
    2. https://github.com/nutonomy/nuscenes-devkit
    for reference
'''

is_centerpoint = False
# 3 is more resonalble for TP, but would cause lots FT/FN and less TP compared to 5 (about 400-500 improvement)
tp_dist_thr = 5
# Give large TP seaching region for long object, but also take consideratoin for non-overlap object(FP) from TP
# overlap may be more resonable like for long bus which cannot be associated by pure euc dist

# Flag for AMOTA, AMOTP
avg_metric_flag = False

# evaluate for moving object gt only
only_moving_objects = False
# 2020-09-11-17-31-33_9 for true
# only_moving_objects = True

# if evaluated all output result (include lost/occluded/matched)
outputAll = False

# if rm box w/o point
filterPoint = True

# if use map to filter
filterNonRoi = True

scenes_file = "2020-09-11-17-37-12_4"
# scenes_file = "2020-09-11-17-31-33_9"
# scenes_file = "2020-09-11-17-37-12_1"

# in 2020-09-11-17-37-12_1
rm_gt_frame = [
    1599817043592381000,
    1599817043692559000,
    1599817044692825000,
    1599817044792887000,
    1599817045292367000,
    1599817047197079000,
    1599817047292303000,
    1599817048392473000,
    1599817049892759000,
    1599817049992304000,
    1599817052292929000,
    1599817052393742000,
    1599817052492560000,
    1599817052593139000,
    1599817052792495000]

original_detection = {}

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


# # Forced output yaml to indent
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)



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
        for t in tqdm(sorted(self.gt)):
            current_gts = self.gt[t]
            self.gt_num += len(current_gts)

            # 1. missing frame in det
            # try without cal missing frame (just continue)
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
                # print('gt {} isn\'t tracked at all'.format(gid))
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
            'total_gt_frame_num': self.frame_num,
            'filtering_gt_w/o point': filterPoint,
            'filtering_pred_w/o_matched': (not outputAll),
            'filterNonRoiObj': filterNonRoi,
            'evaluated_only_moving': only_moving_objects}
        
        self.metrics = result

        if self.verbose:
            print('')
            print('We have {} frames'.format(self.frame_num))
            print('gt: {}'.format(self.gt_num))
            print('trajectory: {}'.format(self.trajectory_num))
            print('lost_trajectory: {}'.format(lost_trajectory))
            print('det: {}'.format(self.hyp_num))
            print('redun hyp frame #: {}'.format(num_frame_wo_gt))
            print('=============Metrics=============')
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
        print('create {}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_{}m'.format(tp_dist_thr))))
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

def transform_coor(inBoxes):
    tf_path = '/data/itri_output/tracking_output/tf_localization/'+scenes_file+'.json'
    tf_data = {}
    with open (tf_path, 'r') as file:
        tf_data = json.load(file)

    outBoxes = {}
    for t, f in inBoxes.items():
        outBoxes[t] = copy.deepcopy(f)
        outBoxes[t]['objects'] = []
        time = t + '0'*3
        tf_localization = {}
        if tf_data.has_key(str(time)):
            tf_localization = tf_data[str(time)]['pose']
        else:
            print('No tf data of gt at time: {}'.format(time))
            continue

        tf_pose = quaternion_matrix(
            np.array([tf_localization['rotation']['x'], tf_localization['rotation']['y'], tf_localization['rotation']['z'], tf_localization['rotation']['w']]))        
        tf_pose[0, 3] = tf_localization['translation']['x']
        tf_pose[1, 3] = tf_localization['translation']['y']
        tf_pose[2, 3] = tf_localization['translation']['z']
        
        for obj in f['objects']:
            local_pose = quaternion_matrix(
                np.array([obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'], obj['rotation']['w']]))        
            local_pose[0, 3] = obj['translation']['x']
            local_pose[1, 3] = obj['translation']['y']
            local_pose[2, 3] = obj['translation']['z']

            global_pose = tf_pose.dot(local_pose)

            global_obj = copy.deepcopy(obj)
            global_obj['rotation']['x'] = float(quaternion_from_matrix(global_pose)[0])
            global_obj['rotation']['y'] = float(quaternion_from_matrix(global_pose)[1])
            global_obj['rotation']['z'] = float(quaternion_from_matrix(global_pose)[2])
            global_obj['rotation']['w'] = float(quaternion_from_matrix(global_pose)[3])
            global_obj['translation']['x'] = float(global_pose[0, 3])
            global_obj['translation']['y'] = float(global_pose[1, 3])
            global_obj['translation']['z'] = float(global_pose[2, 3])
            outBoxes[t]['objects'].append(global_obj)

    return outBoxes


def filter_det_by_vel(inBoxes, vel_thr):
    '''
        Filtering trajectories with vel_thr 
        If the OVERALL VEL is less then vel_thr, remove the trajectory
        Need tf data to global coordinate
    '''
    outBoxes = {}
    static_id = []

    # transform to global coordinate
    global_Boxes = transform_coor(inBoxes)

    # {'id1':[], 'id2':[]}
    inBoxes_tra = {}
    # re-configure det result to trajectory-based
    for t, f in global_Boxes.items():
        for obj in f['objects']:
            if not inBoxes_tra.has_key(obj['tracking_id']):
                inBoxes_tra[obj['tracking_id']] = []
            inBoxes_tra[obj['tracking_id']].append(obj)

    for id, objs in inBoxes_tra.items():
        first_loc = np.array([objs[0]['translation']['x'], objs[0]['translation']['y']])
        last_loc = np.array([objs[-1]['translation']['x'], objs[-1]['translation']['y']])
        dt = float(objs[-1]['timestamp']+'0'*3) - float(objs[0]['timestamp']+'0'*3)
        ava_v = (last_loc-first_loc)/dt*(10**(9))
        if np.sqrt(np.sum(ava_v**2)) < vel_thr:
            static_id.append(id)

    print('We have all {} static objs'.format(len(static_id)))

    # return obj
    for t, f in inBoxes.items():
        outBoxes[t] = copy.deepcopy(f)
        outBoxes[t]['objects'] = []
        for obj in f['objects']:
            if not (obj['tracking_id'] in static_id):
                outBoxes[t]['objects'].append(obj)
    
    return outBoxes


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
        moving_obj = False
        for idx, gt in enumerate(one_trajectory_list):
            if only_moving_objects:
                if gt.has_key('tags'):
                    for t in gt['tags']:
                        if t == 'moving':
                            moving_obj = True
                if not moving_obj:
                    continue
            
            sup_dig = len(str(one_trajectory_list[idx]['header']['stamp']['nsecs']))
            t_step = str(one_trajectory_list[idx]['header']['stamp']['secs']) + '0'*(9-sup_dig) + str(gt['header']['stamp']['nsecs'])

            if not gt_data.has_key(t_step) :
                gt_data[t_step] = []
            
            gt_data[t_step].append({
                'track': one_trajectory_list[idx],
                'id': int(anno['id'])})
    
    if only_moving_objects:
        moving_dets = filter_det_by_vel(dets, vel_thr=1)
    else: 
        moving_dets = dets
    # det record micro-sec in ['usecs'], supplement to 6 digits and then *1000 to usecs
    for t, f in moving_dets.items():
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
                'header': header_dict,
                'occluded_part': obj['occluded_part']}

            if obj.has_key('tags'):
                o['track'].update({'tags': obj['tags']})
            
            det_data[t_step].append(o)

    print('gt: {}'.format(len(gt_data.keys())))
    print('det: {}'.format(len(det_data.keys())))

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
            state = 'matched'
            obj_vel = {'x':  obj['velocity'][0], 'y':  obj['velocity'][1]}
            obj_local_dict = {'timestamp': str(t)[:-3], 'rotation': rot, 'translation': trans, 'size': size, 'tracking_id': obj['tracking_id'], 'tracking_name': obj['tracking_name'], 'tracking_score': obj['tracking_score'], \
                'state': state, 'velocity' :obj_vel}
            obj_lists.append(obj_local_dict)
        single_frame = {'header': header, 'objects': obj_lists, 'pose': pose}       

        re_config_dets.update({str(t)[:-3]:single_frame})

    p = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9'
    with open (p+'/re_config_dets.json', 'w') as OutF:
        json.dump(re_config_dets, OutF, indent = 4)
    # exit(-1)
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

def RectToPoly(groundTruth, shift=0, direction=0, retrun_shapely = False):
    halfL = groundTruth['box']['length'] / 2.0
    halfW = groundTruth['box']['width'] / 2.0
    matrixR = quaternion_matrix(
        [groundTruth['rotation']['x'],
        groundTruth['rotation']['y'],
        groundTruth['rotation']['z'],
        groundTruth['rotation']['w']])[:3, :3]

    if direction == 0:
        pt1 = np.dot(matrixR, [halfL + shift, halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt2 = np.dot(matrixR, [- halfL + shift, halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt3 = np.dot(matrixR, [- halfL + shift, - halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt4 = np.dot(matrixR, [halfL + shift, - halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
    else:
        pt1 = np.dot(matrixR, [halfL + shift, halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt2 = np.dot(matrixR, [- halfL + shift, halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt3 = np.dot(matrixR, [- halfL + shift, - halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt4 = np.dot(matrixR, [halfL + shift, - halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]

    if retrun_shapely:
        return Polygon(
            [(pt1[0], pt1[1]),
            (pt2[0], pt2[1]),
            (pt3[0], pt3[1]),
            (pt4[0], pt4[1])])
    else:
        np_poly = np.vstack((pt1[:2], pt2[:2], pt3[:2], pt4[:2]))
        return np_poly


def loadRoiMap(tf_data):
    map_path = '/home/user/catkin_ws_itri/src/imm_tracker_official/src/euclidean_clustering_filter/material/hsinchu_guandfuroad/non_accessible_region_v2.json'
    with open (map_path, mode='r') as f:
        regions = json.load(f)['non_accessible_region']
    
    sub_map_region = []
    traversed_map_id = []
    for t, attribute in tf_data.items():
        ego_pose = np.array([attribute['pose']['translation']['x'], attribute['pose']['translation']['y']])
        for r in regions:
            if r['id'] in traversed_map_id:
                continue
            center = np.array([r['center_point']['x'], r['center_point']['y']])
            if np.sqrt(np.sum((center-ego_pose)**2)) < (r['spanning_length']/2+30):
                sub_map_region.append(r)
                traversed_map_id.append(r['id'])

    print('Get {} sub map'.format(len(sub_map_region)))
    return sub_map_region


def filter_non_roi_boxes(inBoxes, scenes, output_path):
    '''
        filter out trajectory which used to be located in the non accessible areas
        only tra which 80% of lifetime in the non accessible areas would be removed
    '''
    tf_path = os.path.join('/data/itri_output/tracking_output/tf_localization', scenes+'.json')
    # tf_path = '/data/itri_output/tracking_output/tf_localization/2020-09-11-17-31-33_9.json'
    with open (tf_path, mode='r') as f:
        tf_data = json.load(f)
    
    sub_map = loadRoiMap(tf_data)
    # with open(os.path.join(output_path, 'sub_map.json'), 'w') as outF:
    #     json.dump(sub_map, outF, indent = 4)
        
    # {'id1':[t1, t2...], 'id2':[]}
    rm_box_time = {}
    # remove whole gt trajectory
    rm_id = []
    box_in_non_access = {}
    for region in sub_map:
        pointList = [[pt['x'], pt['y']] for pt in region['points']]
        m_poly = Polygon([[p[0], p[1]] for p in pointList])
        center = np.array([region['center_point']['x'], region['center_point']['y']])
        for t, trks in inBoxes.items():
            if not tf_data.has_key(t):
                continue
            T_tf = quaternion_matrix(
                np.array([
                    tf_data[t]['pose']['rotation']['x'],
                    tf_data[t]['pose']['rotation']['y'],
                    tf_data[t]['pose']['rotation']['z'],
                    tf_data[t]['pose']['rotation']['w']]))
        
            T_tf[0, 3] = tf_data[t]['pose']['translation']['x']
            T_tf[1, 3] = tf_data[t]['pose']['translation']['y']
            T_tf[2, 3] = tf_data[t]['pose']['translation']['z']
            
            for trk in trks:
                # transform box to global coordinate
                T_local = quaternion_matrix(
                    np.array([
                        trk['track']['rotation']['x'],
                        trk['track']['rotation']['y'],
                        trk['track']['rotation']['z'],
                        trk['track']['rotation']['w']]))
            
                T_local[0, 3] = trk['track']['translation']['x']
                T_local[1, 3] = trk['track']['translation']['y']
                T_local[2, 3] = trk['track']['translation']['z']

                trk_global_pose = T_tf.dot(T_local)
                if np.sqrt(np.sum((center-trk_global_pose[:2, 3])**2)) > (region['spanning_length']/2+30):
                    continue

                global_track_dict = copy.deepcopy(trk['track'])
                global_track_dict['translation']['x'] = trk_global_pose[0, 3]
                global_track_dict['translation']['y'] = trk_global_pose[1, 3]
                global_track_dict['translation']['z'] = trk_global_pose[2, 3]
                global_track_dict['rotation']['x'] = quaternion_from_matrix(trk_global_pose)[0]
                global_track_dict['rotation']['y'] = quaternion_from_matrix(trk_global_pose)[1]
                global_track_dict['rotation']['z'] = quaternion_from_matrix(trk_global_pose)[2]
                global_track_dict['rotation']['w'] = quaternion_from_matrix(trk_global_pose)[3]
                
                trk_poly = RectToPoly(global_track_dict,retrun_shapely=True)
                if m_poly.intersection(trk_poly).area > (0.05*trk_poly.area):
                    if not rm_box_time.has_key(trk['id']):
                        rm_box_time[trk['id']] = []
                        box_in_non_access[trk['id']] = 0
                    rm_box_time[trk['id']].append(t)
                    box_in_non_access[trk['id']] += 1

    boxesLen = {}
    original_counter = 0
    for time, objslist in inBoxes.items():
        for obj in objslist:
            if not boxesLen.has_key(obj['id']):
                boxesLen[obj['id']] = 0
            boxesLen[obj['id']] += 1
            original_counter +=1

    print('{} track in non accessible'.format(len(box_in_non_access)))
    # print(box_in_non_access)
    # for id in boxesLen.keys():
    #     if id in box_in_non_access.keys():
    #         print('id: {}, all: {}'.format(id, boxesLen[id]))

    # get rm trajectory from overlap candidates
    for id, times in box_in_non_access.items():
        # only remove trajectory which over 80% of lifetime located in non-accessible
        if times/boxesLen[id] > 0.8:
            rm_id.append(id)
            # print('id {} rm for {:.2f} of times in non-accessible'.format(id, times/boxesLen[id]))
    
    # output
    outBoxes = {}
    counter = 0
    for t, trks in inBoxes.items():
        outBoxes[t] = []
        for trk in trks:
            if trk['id'] not in rm_id:
                outBoxes[t].append(trk)
            else:
                counter+=1

    print('Original {} boxes'.format(original_counter))
    print('Remove {} trajectories, {} boxes'.format(len(rm_id), counter))
    return outBoxes, rm_id

def load_pc(pc_path):
    files_list = sorted(os.listdir(pc_path))
    lidar_scans_list = [lidar_scan.split('.')[0] for lidar_scan in files_list]
    # print(lidar_scans_list)

    scans_dict = {}
    for index, f in enumerate(files_list):
        scanPath = os.path.join(pc_path, f)
        raw_scan = np.fromfile(scanPath, dtype=np.float32)
        # (N, 4)
        scan = raw_scan.reshape(-1, 4)
        scans_dict[lidar_scans_list[index]] = scan

    return scans_dict

def isBoxEmpty(gt_poly, pc):
    '''
        return True if at least one point in poly
    '''
    import matplotlib.path as mplPath

    # crd = np.array([[0,0], [0,1], [1,1], [1,0]])# poly
    bbPath = mplPath.Path(gt_poly)
    points = []
    for pt in pc:
        points.append([pt[0], pt[1]])

    # pnts = [[0.0, 0.0],[1,1],[0.0,0.5],[0.5,0.0]] # points on edges
    # tolerance for point on edge, if = 0, on edge is not exclude
    r = 0.001 # accuracy
    isIn = [ bbPath.contains_point(pt,radius=r) or bbPath.contains_point(pt,radius=-r) for pt in points]
    
    if np.any(isIn):
        return False
    else:
        return True


def filter_box_without_point(gts, scenes_file):
    '''
        Filtering boxes with no point inside
    '''

    big_scene = scenes_file.split('_')[0]
    
    # scenes_file = "2020-09-11-17-37-12_4"
    pc_path = os.path.join('/data/itri_output/tracking_output/pointcloud/no_ego_compensated', big_scene, scenes_file)
    # scenes_file = "2020-09-11-17-31-33_9"
    # pc_path = os.path.join('/data/itri_output/tracking_output/kuang-fu-rd_livox_public/ego compensation/kuang-fu-rd_v3', big_scene, 'pointcloud', scenes_file)
    # scenes_file = "2020-09-11-17-37-12_1"
    # pc_path = os.path.join('/data/itri_output/tracking_output/pointcloud/ego_compensated' , big_scene, scenes_file)
    # load point cloud
    all_pc = load_pc(pc_path)

    out_gts_data = copy.deepcopy(gts)
    out_gts_data['tracks'] = []

    if len(gts['tracks']) == 0:
        print("Zero gts, abort")
        return out_gts_data

    for anno in gts['tracks']:
        if not anno.has_key('track'):
            print(anno)
            continue

        filtered_trajectory_list = []
        moving_obj = False
        for idx, gt in enumerate(anno['track']):
            if only_moving_objects:
                if gt.has_key('tags'):
                    for t in gt['tags']:
                        if t == 'moving':
                            moving_obj = True
                if not moving_obj:
                    continue

            timestep = gt['header']['stamp']['secs']*(10**9) + gt['header']['stamp']['nsecs']
            # crop pc to box height, N X 4 array
            upper_b = gt['translation']['z']+gt['box']['height']
            lower_b = gt['translation']['z']-gt['box']['height']
            roi = 15
            roi = (gt['box']['length']/2 if (gt['box']['length']>gt['box']['width']) else gt['box']['width']/2)
            left_x = gt['translation']['x']-roi*1.5
            right_x = gt['translation']['x']+roi*1.5
            left_y = gt['translation']['y']-roi*1.5
            right_y = gt['translation']['y']+roi*1.5
            current_pc = all_pc[str(timestep)]
            # extract pc within box heigh and cropped to 2d (N x 2)
            cropped_pc = current_pc[(current_pc[:, -2] >= lower_b) & (current_pc[:, -2] <= upper_b)][:, :2]
            cropped_center_x = cropped_pc[(cropped_pc[:, 0] >= left_x) & (cropped_pc[:, 0] <= right_x)]
            cropped_center_y = cropped_center_x[(cropped_center_x[:, 1] >= left_y) & (cropped_center_x[:, 1] <= right_y)]

            gt_poly = RectToPoly(gt)

            if isBoxEmpty(gt_poly, cropped_center_y):
                continue            
            
            filtered_trajectory_list.append(gt)
        
        if len(filtered_trajectory_list) == 0:
            continue
        
        single_tra_dict = {'id': anno['id'], 'track': filtered_trajectory_list}
        out_gts_data['tracks'].append(single_tra_dict)

    return out_gts_data


def filter_box_without_matched(inBox, allGtFrame, outputAllState):

    start_t = sorted(allGtFrame)[0]
    end_t = sorted(allGtFrame)[-1]

    outBox = {}
    for time, boxes in inBox.items():
        t_s = int(time + '0'*3)

        # filtering frame out of range of label
        if t_s < start_t or t_s > end_t or t_s in rm_gt_frame:
            continue

        outBox[time] = copy.deepcopy(boxes)
        outBox[time]['objects'] = []
        for obj in boxes['objects']:
            if outputAllState or is_centerpoint:
                outBox[time]['objects'].append(obj)
            else:
                #  comment if evaluate all
                if obj.has_key('state') and obj['state'] == 'matched':
                    outBox[time]['objects'].append(obj)

    original_det_num = 0
    for time, boxes in inBox.items():
        for det in boxes['objects']:
            original_det_num += 1

    filtered_det_num = 0
    for time, boxes in outBox.items():
        for det in boxes['objects']:
            filtered_det_num += 1

    print('Filter {} dets without matched'.format(original_det_num-filtered_det_num))

    return outBox
    

def interpolateTra(index, tras, times, isCtrv=True):
    '''
        interpolate of time btw (index-1) and index of tras
        @param:
            INPUT:
                index: int, occluded times is btw (index-1) and index of tras
                tras: list of trajectory
                times: list of timestamps, which is consecutive timestamp needed to be interpolate into tras
            OUPUT:
                dict of interpolated tras

    '''
    assert index > 0, 'wrong position of insertion of trajectory'
    outTras = {}

    p1_dict = tras['track'][index-1]
    t1 = p1_dict['header']['stamp']['secs']* 10**(9) + p1_dict['header']['stamp']['nsecs']
    (roll_1, pitch_1, yaw_1) = euler_from_quaternion ([ p1_dict['rotation']['x'], p1_dict['rotation']['y'], p1_dict['rotation']['z'], p1_dict['rotation']['w'] ])
    p1 = quaternion_matrix(
        np.array([ p1_dict['rotation']['x'], p1_dict['rotation']['y'], p1_dict['rotation']['z'], p1_dict['rotation']['w'] ]))
    p1[0, 3] = p1_dict['translation']['x']
    p1[1, 3] = p1_dict['translation']['y']
    p1[2, 3] = p1_dict['translation']['z']

    p2_dict = tras['track'][index]
    t2 = p2_dict['header']['stamp']['secs']* 10**(9) + p2_dict['header']['stamp']['nsecs']
    (roll_2, pitch_2, yaw_2) = euler_from_quaternion ([ p2_dict['rotation']['x'], p2_dict['rotation']['y'], p2_dict['rotation']['z'], p2_dict['rotation']['w'] ])
    p2 = quaternion_matrix(
        np.array([ p2_dict['rotation']['x'], p2_dict['rotation']['y'], p2_dict['rotation']['z'], p2_dict['rotation']['w'] ]))
    p2[0, 3] = p2_dict['translation']['x']
    p2[1, 3] = p2_dict['translation']['y']
    p2[2, 3] = p2_dict['translation']['z']

    # may be opposite orientation wiht pi (yaw = [-pi, pi])
    while math.fabs(yaw_1-yaw_2) > math.pi/2:
        if yaw_2 > yaw_1:
            yaw_2 -= math.pi
        else:
            yaw_2 += math.pi
    # if math.fabs(yaw_1-yaw_2) > math.pi/2:
    #     if yaw_2 > yaw_1:
    #         yaw_2 -= math.pi
    #     else:
    #         yaw_2 += math.pi
        q_2 = quaternion_from_euler(roll_2, pitch_2, yaw_2)
        p2 = quaternion_matrix(
            np.array([ q_2[0], q_2[1], q_2[2], q_2[3] ]))
        p2[0, 3] = p2_dict['translation']['x']
        p2[1, 3] = p2_dict['translation']['y']
        p2[2, 3] = p2_dict['translation']['z']

    assert math.fabs(yaw_1-yaw_2) < math.pi/2, 'not corrected to the same orientation'

    # v = (p2 - p1) / (float(t2 - t1)/10**(9))
    linear_v = (p2[:3, 3] - p1[:3, 3]) / (float(t2 - t1)/10**(9))
    if isCtrv:
        yaw_v = (yaw_2 - yaw_1) / (float(t2 - t1)/10**(9))
    else:
        yaw_v = 0

    for t in times:
        outTras[t] = copy.deepcopy(p1_dict)
        outTras[t]['header']['stamp']['secs'] = int(t/10**(9))
        outTras[t]['header']['stamp']['nsecs'] = t%10**(9)
        dt = float(t - t1)/10**(9)
        # p = p1 + v * dt
        p = np.eye(4, dtype=float)
        p[:3, 3] = p1[:3, 3] + linear_v * dt
        yaw_p = yaw_1 + yaw_v * dt
        q = quaternion_from_euler(roll_1, pitch_1, yaw_p)

        outTras[t]['rotation']['x'] = float(q[0])
        outTras[t]['rotation']['y'] = float(q[1])
        outTras[t]['rotation']['z'] = float(q[2])
        outTras[t]['rotation']['w'] = float(q[3])
        outTras[t]['translation']['x'] = float(p[0, 3])
        outTras[t]['translation']['y'] = float(p[1, 3])
        outTras[t]['translation']['z'] = float(p[2, 3])

        # linearly interpolate occluded grid coordinate from last position, for detection file
        if outTras[t].has_key('occluded_part'):
            # det 188 @ 1599817094.792036 matched (without occluded part) but next 1599817094.892315 lost 
            # direct retrieve lost track from original det file from "lost" state
            for idx, obj in enumerate(original_detection[str(t)[:-3]]['objects']):
                if tras['id'] == int(obj['tracking_id']):
                    # print('p1_dict occluded: {}; have {} occluded'.format(t1, len(p1_dict['occluded_part'])))
                    # print('now interpolate for {}'.format(t))
                    # if len(p1_dict['occluded_part']) != 0:
                    #     print("{}: before {}, {}, {}".format(tras['id'], outTras[t]['occluded_part'][0]['x'], outTras[t]['occluded_part'][0]['y'], outTras[t]['occluded_part'][0]['z']))
                    # else:
                    #     print("{}: before, no occluded @ p1 {}".format(tras['id'], t1))
                    outTras[t]['occluded_part'] = obj['occluded_part'] if obj.has_key('occluded_part') else []
                    # print("{}: after {}, {}, {}".format(tras['id'], outTras[t]['occluded_part'][0]['x'], outTras[t]['occluded_part'][0]['y'], outTras[t]['occluded_part'][0]['z']))
                    break


        # add tags
        if outTras[t].has_key('tags'):
            outTras[t]['tags'].append('interpolated')
        else:
            outTras[t]['tags'] = ['interpolated']
        # print(outTras[t])
    
    return outTras

def sortTime(tra):
    time = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
    return time

def reconfigure_det_to_gt(inBoxes):
    outBoxes = {'tracks':[]}

    # {'id1":[], 'id2": []}
    all_trajectory = {}
    for time, boxes in inBoxes.items():
        for obj in boxes['objects']:
            if not all_trajectory.has_key(obj['tracking_id']):
                all_trajectory[obj['tracking_id']] = []

            all_trajectory[obj['tracking_id']].append(obj)

    for id, tras in all_trajectory.items():
        one_id = {'id': int(id), 'track':[]}
        for tra in tras:
            time = tra['timestamp']+'0'*3
            one_tra = {}
            one_tra['rotation'] = tra['rotation']
            one_tra['translation'] = tra['translation']
            one_tra['box'] = {}
            one_tra['box']['height'] = tra['size']['h']
            one_tra['box']['width'] = tra['size']['w']
            one_tra['box']['length'] = tra['size']['l']
            one_tra['header'] = {'frame_id': '/velodyne', 'stamp': {'nsecs': int(str(time)[-9:]), 'secs': int(str(time)[:-9])}}
            # print(time)
            # print('header: {}.{}'.format(one_tra['header']['stamp']['secs'], one_tra['header']['stamp']['nsecs']))
            one_tra['label'] = tra['tracking_name']
            
            # additional key in det but not in ground truth
            one_tra['state'] = tra['state']
            one_tra['velocity'] = tra['velocity']
            one_tra['tracking_score'] = tra['tracking_score']
            one_tra['occluded_part'] = tra['occluded_part'] if tra.has_key('occluded_part') else []

            one_id['track'].append(one_tra)
            one_id['track'].sort(key=sortTime)

        outBoxes['tracks'].append(one_id)

    
    # with open(os.path.join(output_path, 'reconfigure_det_to_gt.json'), mode='w') as outFile:
    #     json.dump(outBoxes, outFile, indent = 4)
    return outBoxes


def interpolate_track(inBoxes, allLidarFrame, isGt=True):

    print('Interpolate for gt: {}'.format(isGt))
    outBoxes = copy.deepcopy(inBoxes)
    if not isGt:
        # reconfigure det as gt trajectory-based format
        reconfig_inBox = reconfigure_det_to_gt(inBoxes)
        outBoxes = {}
        outBoxes = copy.deepcopy(reconfig_inBox)

    tracks = outBoxes['tracks']

    print("have {} tracks".format(len(tracks)))

    for gt in tracks:
        timestamp = []
        for tra in gt['track']:
            time = int(tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs'])
            time_s = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
            if time in rm_gt_frame:
                continue
            timestamp.append(time)

        list.sort(timestamp)
        if len(timestamp) == 0:
            print('{} : 0 stamps'.format(gt['id']))
            continue

        start_t = sorted(timestamp)[0]
        end_t = sorted(timestamp)[-1]

        s_counter = 0
        for idx, t in enumerate(sorted(allLidarFrame)):
            if t < start_t:
                continue
            else:
                s_counter = idx
                break
        
        e_counter = 0
        for idx, t in enumerate(sorted(allLidarFrame)):
            if t < end_t:
                continue
            else:
                e_counter = idx
                break
        # print(allLidarFrame[s_counter], allLidarFrame[e_counter])
        track_life = sorted(allLidarFrame)[s_counter:e_counter+1]
        assert start_t == track_life[0], 'Wrong track life time'
        assert end_t == track_life[-1], 'Wrong track life time'

        absent_num = len(track_life) - len(timestamp)
        # print(len(track_life), len(timestamp), absent_num)
        assert absent_num >= 0, 'Negative absent num'

        if absent_num == 0:
            continue

        show_up = np.full((len(track_life), ), True)
        for idx, t in enumerate(sorted(track_life)):
            if t not in timestamp:
                show_up[idx] = False
        assert (len(track_life) - np.sum(show_up)) == absent_num, 'Wrong show_up list'

        # find consecutive false stand for occlusion regions 
        lost_interval = []
        lost_time = []
        find_occluded = False
        for idx, shown in enumerate(show_up):
            if not shown:
                if not find_occluded:
                    find_occluded = True
                lost_time.append(track_life[idx])

            else:
                # end of occlusion, reset the flag
                if find_occluded:
                    find_occluded = False
                    lost_interval.append(lost_time)
                    lost_time = []

        all_inter_tras = []
        for interval in lost_interval:
            index = np.searchsorted(np.array(timestamp), interval[0]) 
            # append interval btw (index-1) and index of existing timestamp
            # for visaulization, no need for ctrv model to interpolate detection result (rotating)
            # inter_tras = interpolateTra(index, gt, interval, isCtrv=isGt)
            inter_tras = interpolateTra(index, gt, interval)
            all_inter_tras.extend(inter_tras.values())

        # print('Inter {} tras'.format(len(all_inter_tras)))
        # print('Befor inser {} tras'.format(len(gt['track'])))
        gt['track'].extend(all_inter_tras)
        # print('After inser {} tras'.format(len(gt['track'])))
        gt['track'].sort(key=sortTime)

    print("After inter have {} tracks".format(len(tracks)))

    if not isGt:
        # reconfigure back to det format
        outDetBoxes = {}

        # {'t1':[{id1}, {id2}], 't2': ..}
        allDetLidarFrame = {}
        for det in outBoxes['tracks']:
            for idx, tra in enumerate(det['track']):
                time = tra['header']['stamp']['secs']*(10**9) + tra['header']['stamp']['nsecs']
                if not allDetLidarFrame.has_key(str(time)[:-3]):
                    allDetLidarFrame[str(time)[:-3]] = []
                meta = copy.deepcopy(tra)
                meta.update({'id': det['id']})
                allDetLidarFrame[str(time)[:-3]].append(meta)


        # gt_list = list(allLidarFrame)
        # det_list = allDetLidarFrame.keys()
        # print('gt_list: {}; det_list: {}'.format(len(gt_list), len(det_list)))

        for time in sorted(allDetLidarFrame.keys()):
            counter = 0
            if not inBoxes.has_key(str(time)):
                # Lost frame in det, all interpolated at this time
                print('Not in inBoxes: {}'.format(str(time)))
                print('interpolated frame {}'.format(str(time)))
                header = {'stamp': {'secs': int(str(time)[:-6]), 'nsecs': int(str(time)[-6:]+'0'*3)}, 'frame_id': inBoxes[inBoxes.keys()[0]]['header']['frame_id']}
                pose = {'position': {}, 'rotation': {}}
                outDetBoxes[str(time)] = {'header': header, 'objects': [], 'pose': pose}

                for obj in allDetLidarFrame[str(time)]:
                    if obj.has_key('tags'):
                        for tag in obj['tags']:
                            if tag == 'interpolated':
                                counter+=1
                                break
                print('Have {}/{} objs interpolated'.format(counter, len(allDetLidarFrame[str(time)])))
            else:
                outDetBoxes[str(time)] = {'header': inBoxes[str(time)]['header'], 'objects': [], 'pose': inBoxes[str(time)]['pose']}
            for obj in allDetLidarFrame[time]:
                one_tra = {}
                one_tra['rotation'] = obj['rotation']
                one_tra['translation'] = obj['translation']
                one_tra['timestamp'] = str(time)
                one_tra['tracking_id'] = str(obj['id'])
                one_tra['size'] = {'h': obj['box']['height'], 'l': obj['box']['length'], 'w': obj['box']['width']}
                one_tra['tracking_name'] = obj['label']
                one_tra['tracking_score'] = obj['tracking_score']
                one_tra['velocity'] = obj['velocity']
                one_tra['occluded_part'] = obj['occluded_part']
                
                one_tra['state'] = obj['state']
                if obj.has_key('tags'):
                    for tag in obj['tags']:
                        if tag == 'interpolated':
                            one_tra['tags'] = tag

                outDetBoxes[str(time)]['objects'].append(one_tra)
        outBoxes = copy.deepcopy(outDetBoxes)

    return outBoxes
        

if __name__ == "__main__":
    cp_file = "tracking_result_max_1000.json"

    gt_path = '/data/annotation/livox_gt/done/' + scenes_file + '_reConfig_done_local_inter.yaml'
    if only_moving_objects:
        gt_path = '/data/annotation/livox_gt/done/' + scenes_file + '_reConfig_done_local_inter_moving.yaml'
    gts = load_gt(gt_path)

    # baseline
    det_path_l = "/data/annotation/livox_gt/livox_gt_annotate_velodyne_json/without motion compensation/2020-09-11-17-37-12/"
    # merge_detecotr v3
    det_path_m_bl_m = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/baseline_even_clustering_Mdist_pda/without_uncertain_3.5/"
    det_path_m_bl_l = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/baseline_even_clustering_likelihood_pda/test_uncertainty_3.5/"

    det_path_m_occ_base_1 = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/without_uncertain_3.5/occlusion_fun_base6_outputall/"
    det_path_m_occ_base_2 = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/without_uncertain_3.5/occlusion_fun_base16_outputall/"

    det_path_baseline_likelihood = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/baseline_even_clustering_likelihood_pda'
    # centerpoint
    det_path_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/immortal'
    # det_paths = [det_path_l, det_path_m_2, det_path_m, det_path_cp, det_path_m_occ, det_path_m_occ2, det_path_m_occ3]
    test_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/baseline_even_clustering_likelihood_pda'
    livox_base = '/data/itri_output/tracking_output/output/clustering/livox_baseline'
    immortal= '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/immortal'
    
    imm_cp_occlu_path = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/base9.5'
    imm_cp_bl_path = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/baseline'
    imm_cp_imortal = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/immortal'

    imm_occlu_bl= '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/baseline_even_clustering_likelihood_pda/test_uncertainty_3.5'
    imm_imortal ='/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/immortal'
    
    # new seg 2020-09-11-17-37-12_1
    # imm_1_bl = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_1/2020-09-11-17-37-12_1/baseline'
    # imm_2_bl = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/baseline'
    # imm_immortal = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/immortal'

    
    # imm 3d test
    imm_bl_3d = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/baseline_even_clustering_likelihood_pda/3d_state_2d_asso_ordinary_q_'
    imm_occlu_3d = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/result/occlusion_fun_base9.5_outputall/3d_state_2d_asso_ordinary_q_'
    # det_paths = [imm_bl_3d, imm_occlu_3d]

    cp_imm_occlu_3d = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/base9.5/3d_state_2_2dassociate'
    cp_imm_bl_3d = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/baseline/3d_state_2_2dassociate'

    # only pda
    cp_imm_pda = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion_only_pda_no_track'
    cluster_imm_pda = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occlusion_only_pda_no_track'

    
    # -----
    # record occlude grid
    # imm_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/result/occlusion_fun_base9.5_outputall/occluded_grid_record'
    # imm_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/base9.5/occluded_grid_record'
    # det_paths = [imm_cp]

    # imm_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/result/occlusion_fun_base6/occluded_grid_record'
    # imm_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/occlusion/with groundf/base6/occluded_grid_record'
    # det_paths = [imm_cluster, imm_cp]

    # imm_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occlusion/ClipHeight_2m/original_q/9.5/occluded_grid_record'
    # imm_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion/ClipHeight_2m/original_q/9.5/occluded_grid_record'
    # det_paths = [imm_cluster, imm_cp]

    # ------

    # ablation
    # det_paths = []
    # all_det_paths = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/occlu_pda_lifetime/linear_result'
    # # # all_det_paths = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion/ClipHeight_2m/original_q'
    # # # # all_det_paths = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/track_lifespan_no_occlusion_w_merge/result'
    # files_list = sorted(os.listdir(all_det_paths))
    # for f in files_list:
    #     det_paths.append(os.path.join(all_det_paths, f))


    # test_clustering_occlusion = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/no_pda_result/occlusion_fun_base9.5_outputall'
    # det_paths = [test_clustering_occlusion]

    # all_det_paths = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/occlusion/with groundf/no_pda_result'
    # # all_det_paths = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/occlu_pda_lifetime/result'
    # files_list = sorted(os.listdir(all_det_paths))
    # for f in files_list:
    #     det_paths.append(os.path.join(all_det_paths, f))

    # pda_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion/ClipHeight_2m/original_q/only_pda_result'
    # pda_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occlusion/ClipHeight_2m/original_q/only_pda_result'
    # det_paths = [pda_cp, pda_cluster]
    
    # track lifetime comparison
    # det_path = "/data/itri_output/tracking_output/output/track_lifespan_no_occlusion_w_merge/unprocessed"
    # files_list = sorted(os.listdir(det_path))
    # det_paths = files_list
    # dets_all = []
    # is_centerpoint = False
    # print('Total have {} files to evaluate'.format(len(det_paths)))
    # for f in files_list:
    #     print(f)
    #     dets = load_det(os.path.join(det_path, f), is_centerpoint)
    #     dets_all.append(dets)

    # linear
    # imm_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occlusion/ClipHeight_2m/original_q/linear_T/2.5'
    # imm_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion/ClipHeight_2m/original_q/linear_T/2.5'
    imm_cp = '/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/linear_T/2.5'
    imm_cluster = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/linear_T/2.5'
    det_paths = [imm_cluster, imm_cp]

    dets_all = []
    is_centerpoint = False
    print('Total have {} files to evaluate'.format(len(det_paths)))
    for idx,det_path in enumerate(det_paths):
        # if idx != 3:
        #     is_centerpoint = False
        #     dets = load_det(os.path.join(det_path, scenes_file + '_ImmResult.json'), is_centerpoint)
        # else:
        #     is_centerpoint = True
        #     dets = load_det(os.path.join(det_path, cp_file), is_centerpoint)
        is_centerpoint = False
        dets = load_det(os.path.join(det_path, scenes_file + '_ImmResult.json'), is_centerpoint)
        dets_all.append(dets)
        
        # is_centerpoint = True
        # dets = load_det(os.path.join(det_path, cp_file), is_centerpoint)
        # dets_all.append(dets)
    print('Load all {} files'.format(len(det_paths)))

    if filterPoint:
        # preprocessing - rm gt with empty cloud point
        start = timer()
        filtered_gts = filter_box_without_point(gts, scenes_file)
        end = timer()
        print('filter_box_without_point: {:.2f} sec'.format(end - start))

        original_gt_num = 0
        for all_gts in gts['tracks']:
            for gt in all_gts['track']:
                original_gt_num += 1

        filtered_gt_num = 0
        for f_gt in filtered_gts['tracks']:
            for gt in f_gt['track']:
                filtered_gt_num += 1

        print('Filter {} gts without points'.format(original_gt_num-filtered_gt_num))
    else:
        filtered_gts = copy.deepcopy(gts)
    
    for iter in range(len(dets_all)):
        print('-'*80)
        output_dir = det_paths[iter]
        # output_dir = det_path + "/result"
        output_path = filecreation(output_dir)
        
        dets = dets_all[iter]
        original_detection = copy.deepcopy(dets)
        if is_centerpoint:
        # if iter == 3:
            dets = reconfigure(dets)

        # filtered_gts = {}
        # with open(os.path.join('/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/occlusion_fun_base6_outputall/2022-07-28_14-44-43_5m', 'filtered_gts.yaml'), mode='r') as outFile:
        #     filtered_gts = yaml.load(outFile)

        allGtFrame = set()
        non_labeled_filtered_gts = copy.deepcopy(filtered_gts)
        non_labeled_filtered_gts['tracks'] = []
        # preprocessing - get all gt timestamps
        for f_gt in filtered_gts['tracks']:
            non_labeled_gt = {'id': f_gt['id'], 'track': []}
            for gt in f_gt['track']:
                time = gt['header']['stamp']['secs']*(10**9) + gt['header']['stamp']['nsecs']

                if time in rm_gt_frame:
                    # print('Non labeled gt {}'.format(time))
                    continue
                else:
                    non_labeled_gt['track'].append(gt)
                    allGtFrame.add(time)
            non_labeled_filtered_gts['tracks'].append(non_labeled_gt)
        print('have {} frames gt'.format(len(allGtFrame)))

        # with open(os.path.join(output_path, 'filtered_gts.yaml'), mode='w') as outFile:
        #     documents = yaml.dump(filtered_gts, outFile, Dumper=MyDumper, default_flow_style=False)

        # preprocessing - rm det without 'matched' state and det which is out range of gt 
        filtered_dets = filter_box_without_matched(dets, allGtFrame, outputAll)

        # with open(os.path.join(output_path, 'filtered_dets.json'), mode='w') as outFile:
        #     json.dump(filtered_dets, outFile, indent = 4)

        # # preprocessing - interpolated gt and det at gt timestamps
        # interpolated_gts = interpolate_track(filtered_gts, allGtFrame, isGt=True)
        interpolated_gts = interpolate_track(non_labeled_filtered_gts, allGtFrame, isGt=True)
        interpolated_dets = interpolate_track(filtered_dets, allGtFrame, isGt=False)

        # with open(os.path.join(output_path, 'filtered_inter_gts.yaml'), mode='w') as outFile:
        #     documents = yaml.dump(interpolated_gts, outFile, Dumper=MyDumper, default_flow_style=False)

        # with open(os.path.join(output_path, 'filtered_inter_dets.json'), mode='w') as outFile:
        #     json.dump(interpolated_dets, outFile, indent = 4)

        print('Get det from {}'.format(det_paths[iter]))
        start = timer()
        gt_frame, det_frame = config_data(interpolated_gts, interpolated_dets, output_path)
        end = timer()
        print('Config_data: {:.2f} sec'.format(end - start))

        # if is_centerpoint:
        if filterNonRoi:
            gt_frame, _ = filter_non_roi_boxes(gt_frame, scenes_file, output_path)
            det_frame, _ = filter_non_roi_boxes(det_frame, scenes_file, output_path)

        with open(os.path.join(output_path, "gt.json"), "w") as outfile:
            json.dump(gt_frame, outfile, indent = 4)
        with open(os.path.join(output_path, "det.json"), "w") as outfile:
            json.dump(det_frame, outfile, indent = 4)

        mot = MOTAccumulator(gt_frame, det_frame, output_path, dist_thr=tp_dist_thr)
        start = timer()
        mot.evaluate()
        end = timer()
        print('Evaluate time: {:.2f} sec'.format(end - start))
        mot.cal_metrics()
        mot.outputMetric()

        # if avg_metric_flag and iter == 3:
        if avg_metric_flag:
            print('\nEvaluating AMOTA/AMOTP...\n')
            AVG_METRIC_main(mot)


