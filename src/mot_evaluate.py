from __future__ import division
import yaml
import json
import os
import numpy as np
import collections
from scipy.optimize import linear_sum_assignment
from timeit import default_timer as timer

'''
    Take https://github.com/cheind/py-motmetrics for reference
'''


def euc_dist(obj1, obj2):
    return np.sqrt(np.sum((obj1-obj2)**2))

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
        # only trackers with id_switch and correspondent det id {{'o1':{"history":[], 'switch_num': }, {o2}, {}}
        self.track_id_switch = {}
        
        
        # record hyp frame and # with no gt
        self.redunt_hyp = {}
        self.output_path = output_path

        # record fp hyp at each frame {'t1': [{'track':..., 'id':}, {}, ..]}
        self.fp_hyp = {}
        self.fn_gt = {}

        self.verbose = verbose

    def register_new_track(self, timestamp, gid):
        self.track_history[gid] = [False]
        self.dist_error[gid] = [np.inf]
        self.gt_frame[gid] = [timestamp]
        self.id_history[gid] = [np.nan]
        self.last_match[gid] = np.nan

    def evaluate(self):
        # mxn matrix -> track# * det#
        # iterate all frame in gt, only evaluate by gt frame
        for t, current_gts in self.gt.items():
            self.gt_num += len(current_gts)

            # 1. missing frame in det
            if not self.hyp.has_key(t):
                self.tp[t] = 0
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
                    # not establish before, initialize one
                    else:
                        self.register_new_track(t, o_id)
                    self.fn_gt[t].append(gt)
                continue

            gids = [gt['id'] for gt in current_gts]
            hids = [hyp['id'] for hyp in self.hyp[t]]
            
            dist_m = np.matrix(np.ones((len(self.gt[t]), len(self.hyp[t]))) * 1000)
            for i, gt in enumerate(current_gts):
                g = np.array([float(gt['track']['translation']['x']),
                            float(gt['track']['translation']['y']),
                            float(gt['track']['translation']['z'])])
                for j, det in enumerate(self.hyp[t]):
                    d = np.array([float(det['track']['translation']['x']),
                                float(det['track']['translation']['y']),
                                float(det['track']['translation']['z'])])
                    dist_m[i, j] = euc_dist(g, d)

            # result = linear_assignment(cost)
            result = linear_sum_assignment(dist_m)
            # get n*2 of index [i, j] array
            result = np.array(list(zip(*result)))
            # disable correspondence far from threshold, remove correspondence from result
            valid_result = []
            for pair in result:
                # (2, )
                if dist_m[pair[0], pair[1]] < self.dist_thr:
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
                else:
                    self.last_match[o] = h
                    self.track_history[o] = [True]
                    self.dist_error[o] = [dist_m[i, j]]
                    self.gt_frame[o] = [t]
                    self.id_history[o] = [h]

            # set tp
            self.tp[t] = valid_result.shape[0]
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
                # not establish before
                else:
                    self.register_new_track(t, o_id)
                self.fn_gt[t].append(o_dict)
        
        # reocrd det frame without gt
        for t, hyps in self.hyp.items():    
            if t not in self.gt.keys():
                self.redunt_hyp[t] = len(hyps)

        self.trajectory_num = len(self.track_history.keys())
    

    def outputMetric(self):
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

            matched_idx = np.where(np.array(his)==True)
            matched_idx = list(np.array(list(zip(*matched_idx))).flatten())
            for e in np.array(self.dist_error[gid])[matched_idx]:
                total_error += e 
            total_matched += len(matched_idx)

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
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'gt_num': self.gt_num,
            'det_num': self.hyp_num,
            'trajectory_num': self.trajectory_num,
            'lost_trajectory': lost_trajectory,
            'total_gt_frame_num': self.frame_num}
              
        with open(os.path.join(self.output_path, "metrics.json"), "w") as outfile:
            json.dump(result, outfile, indent = 4)

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
            switch_list[id] = {
                'switch_num': switch_num,
                'history': self.id_history[id]}

        with open(os.path.join(self.output_path, "switch_list.json"), "w") as outfile:
            json.dump(switch_list, outfile, indent = 4)

        if self.verbose:
            print('##################################################')
            print('Output to {}'.format(self.output_path))
            print('We have {} frames'.format(self.frame_num))
            print('gt: {}'.format(self.gt_num))
            print('trajectory: {}'.format(self.trajectory_num))
            print('lost_trajectory: {}'.format(lost_trajectory))
            print('det: {}'.format(self.hyp_num))
            print('redun hyp frame #: {}'.format(num_frame_wo_gt))
            print('--------------------------------------------------')
            print('TP: {}'.format(TP))
            print('FP: {}'.format(FP))
            print('FN: {}'.format(FN))
            print('MT: {:.3f}'.format(MT/self.trajectory_num))
            print('ML: {:.3f}'.format(ML/self.trajectory_num))
            print('IDSW: {}'.format(self.id_switch))
            print('recall: {:.3f}'.format(recall))
            print('precision: {:.3f}'.format(precision))
            print('F1-score: {:.3f}'.format(F1))
            print('IDF1: {:.3f}'.format(IDF1))
            print('mota: {:.3f}'.format(mota))
            print('motp: {:.3f}'.format(motp))


def load_gt(file_path):
    with open(file_path, "r") as gt:
        data = yaml.load(gt)
    return data

def load_det(file_path):
    with open (os.path.join(file_path), mode='r') as f:
        curb_map = json.load(f)['frames']
    return curb_map

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


    # ordered_gt = collections.OrderedDict(sorted(gt_data.items()))
    # ordered_det = collections.OrderedDict(sorted(det_data.items()))
    # order by timestemp
    ordered_gt = {k: gt_data[k] for k in sorted(gt_data)}
    ordered_det = {k: det_data[k] for k in sorted(det_data)}
    print('gt: {}'.format(len(ordered_gt.keys())))
    print('det: {}'.format(len(ordered_det.keys())))

    with open(os.path.join(output_path, "gt.json"), "w") as outfile:
        json.dump(ordered_gt, outfile, indent = 4)
    with open(os.path.join(output_path, "det.json"), "w") as outfile:
        json.dump(ordered_det, outfile, indent = 4)

    return ordered_gt, ordered_det



if __name__ == "__main__":
    # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # # # cost = np.array([[4, 1, 3], [2, 0, 5]])
    # result = linear_sum_assignment(cost)
    # # print(result)
    # result = np.array(list(zip(*result)))
    # print(result)
    # # print(result.shape)
    # result_v = []

    # # # A = np.vstack((A, X[X[:,0] < 3]))
    # for pair in result:
    #     # print(pair)
    #     # (2, )
    #     # print(pair.shape)
    #     # np.reshape(pair, (1, pair.shape[0]))
    #     # pair = pair[np.newaxis, :]
    #     if cost[pair[0], pair[1]] > 1:
    #         result_v.append(pair)
    #         # result_v = np.vstack((result_v, pair))
    # print(result_v)
    # print(result_v[0].shape)
    # result_v = np.reshape(result_v, (-1, 2))
    # # print(result_v.shape)
    # print(result_v)
    # for i, j in result_v:
    #     print(i, j)

    # total_error=0
    # his = [True, False, False, True]
    # dist_erro = [0.1, np.inf, np.inf, 3]
    # matched_idx = np.where(np.array(his)==True)
    # matched_idx = np.array(list(zip(*matched_idx))).flatten()
    # print(matched_idx)
    # print(matched_idx.shape)
    # matched_idx = list(matched_idx)
    # print(matched_idx)

    # print(np.array(dist_erro).shape)

    # for e in np.array(dist_erro)[matched_idx]:
    #     total_error += e
    # print(total_error)

    # l = np.nan
    # logic = np.isnan(l)
    # print(logic)
    # logic = np.isnan(2)
    # print(logic)
    # exit(-1)


    gt_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done.yaml"
    gts = load_gt(gt_path)

    det_path = "/data/itri_output/tracking_output/output/livox_gt_annotate_velodyne/2020-09-11-17-37-12/2020-09-11-17-37-12_4_ImmResult.json"
    dets = load_det(det_path)

    output_path = "/data/itri_output/evaluation"
    
    start = timer()
    gt_frame, det_frame = config_data(gts, dets, output_path)
    end = timer()
    print('Config_data: {:.2f} sec'.format(end - start)) # Time in seconds, e.g. 5.38091952400282

    mot = MOTAccumulator(gt_frame, det_frame, output_path)
    start = timer()
    mot.evaluate()
    end = timer()
    print('Evaluate time: {:.2f} sec'.format(end - start)) # Time in seconds, e.g. 5.38091952400282
    start = timer()
    mot.outputMetric()
    end = timer()
    print('Evaluate time: {:.2f} sec'.format(end - start)) # Time in seconds, e.g. 5.38091952400282


