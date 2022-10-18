from re import M
import numpy as np
import shapely
from shapely import geometry, affinity
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, translation_from_matrix
import matplotlib.pyplot as plt


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
        # return rc

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

# gt enclose det
det = {
    "track": {
        "box": {
            "width": 0.24090445041656494, 
            "length": 5.255256652832031, 
            "height": 0.6338002681732178
        }, 
        "label": "unknown", 
        "header": {
            "stamp": {
                "secs": 1599817107, 
                "nsecs": 92306
            }, 
            "frame_id": "/velodyne"
        }, 
        "score": 1, 
        "translation": {
            "y": -45.502581564950106, 
            "x": -23.41119642817648, 
            "z": 0.6182850635421673
        }, 
        "rotation": {
            "y": -5.48787881714884e-19, 
            "x": 5.487878817148841e-18, 
            "z": 0.6127816499902214, 
            "w": 0.790252269490738
        }
    }, 
    "id": 743
}

gt = {
    "track": {
        "box": {
            "width": 3.278825998306274, 
            "length": 13.89770221710205, 
            "height": 4.004841327667236
        }, 
        "header": {
            "stamp": {
                "secs": 1599817107, 
                "nsecs": 92306000
            }, 
            "frame_id": "/velodyne"
        }, 
        "rotation": {
            "y": -4.803175515408116e-09, 
            "x": 1.935521472641696e-09, 
            "z": -0.7958817879871445, 
            "w": 0.6054520456240825
        }, 
        "translation": {
            "y": -42.21744155883789, 
            "x": -23.91581916809082, 
            "z": 0.1899734139442444
        }, 
        "label": "bus"
    }, 
    "id": 782
}


yaw_det = getYawFromQuat(det['track'])
yaw_gt = getYawFromQuat(gt['track'])

rect_det = RotatedRect(det['track']['translation']['x'], det['track']['translation']['y'],  det['track']['box']['length'], det['track']['box']['width'], yaw_det)
rect_gt = RotatedRect(gt['track']['translation']['x'], gt['track']['translation']['y'],  gt['track']['box']['length'], gt['track']['box']['width'], yaw_gt)

inter_area = rect_det.intersection(rect_gt).area
iou = inter_area / (rect_det.get_contour().area + rect_gt.get_contour().area - inter_area)
print(yaw_det, yaw_gt)
print(inter_area)
print(rect_gt.get_contour().area)
print(iou)
# print(rect_det.get_contour().area)

fig, ax = plt.subplots()
ax.plot(rect_det.get_contour().boundary.coords.xy[0], rect_det.get_contour().boundary.coords.xy[1], 'b-')
ax.plot(rect_gt.get_contour().boundary.coords.xy[0], rect_gt.get_contour().boundary.coords.xy[1], 'y-')
plt.show()

# det
'''
        {
            "track": {
                "box": {
                    "width": 0.24090445041656494, 
                    "length": 5.255256652832031, 
                    "height": 0.6338002681732178
                }, 
                "label": "unknown", 
                "header": {
                    "stamp": {
                        "secs": 1599817107, 
                        "nsecs": 92306
                    }, 
                    "frame_id": "/velodyne"
                }, 
                "score": 1, 
                "translation": {
                    "y": -45.502581564950106, 
                    "x": -23.41119642817648, 
                    "z": 0.6182850635421673
                }, 
                "rotation": {
                    "y": -5.48787881714884e-19, 
                    "x": 5.487878817148841e-18, 
                    "z": 0.6127816499902214, 
                    "w": 0.790252269490738
                }
            }, 
            "id": 743
        }
'''
# gt
'''
        {
            "track": {
                "box": {
                    "width": 3.278825998306274, 
                    "length": 13.89770221710205, 
                    "height": 4.004841327667236
                }, 
                "header": {
                    "stamp": {
                        "secs": 1599817107, 
                        "nsecs": 92306000
                    }, 
                    "frame_id": "/velodyne"
                }, 
                "rotation": {
                    "y": -4.803175515408116e-09, 
                    "x": 1.935521472641696e-09, 
                    "z": -0.7958817879871445, 
                    "w": 0.6054520456240825
                }, 
                "translation": {
                    "y": -42.21744155883789, 
                    "x": -23.91581916809082, 
                    "z": 0.1899734139442444
                }, 
                "label": "bus"
            }, 
            "id": 782
        },
'''

gt = {
    'label': 'car',
    'header':{
        'frame_id':'velodyne',
        'stamp': {
            'nsecs': 100,
            'secs': 200
        }
    },
    'translation':{
        'x': -3533.7420061254293,
        'y': 1251.8551043676364,
        'z': -57.755891494741824
    },
    'rotation':
    {
        'y': 0.007729317756910398,
        'x':  -0.006433088154245984,
        'z':  0.9351369659900488,
        'w': 0.3541436542742775
    },
    'box':
    {
        'length': 4.324467658996582,
        'width': 1.788819670677185,
        'height': 1.761304616928098
    }
}

poly = {
    "center_point": {
        "y": 1262.11073532008, 
        "x": -3564.0394223130816, 
        "z": 87.138711754634031
    }, 
    "lane_id": 'null', 
    "geos": 'null', 
    "main_id": 310, 
    "points": [
        {
            "y": 1286.0634291099209, 
            "x": -3607.2126262837155, 
            "point_id": -1, 
            "z": 86.383525680085583
        }, 
        {
            "y": 1294.0207054846876, 
            "x": -3601.156528398125, 
            "point_id": 1, 
            "z": 86.30982971191406
        }, 
        {
            "y": 1297.235915445625, 
            "x": -3598.70950691375, 
            "point_id": 2, 
            "z": 86.2800521850586
        }, 
        {
            "y": 1297.5695336096876, 
            "x": -3596.53470222625, 
            "point_id": 3, 
            "z": 86.00328826904297
        }, 
        {
            "y": 1298.664626383125, 
            "x": -3591.24710457, 
            "point_id": 4, 
            "z": 86.07289123535156
        }, 
        {
            "y": 1293.03205802375, 
            "x": -3584.095981523125, 
            "point_id": 5, 
            "z": 86.12108612060547
        }, 
        {
            "y": 1293.6569359534376, 
            "x": -3583.26419441375, 
            "point_id": 6, 
            "z": 86.0127944946289
        }, 
        {
            "y": 1293.397536539375, 
            "x": -3582.935825273125, 
            "point_id": 7, 
            "z": 85.98110961914062
        }, 
        {
            "y": 1292.9359886878126, 
            "x": -3582.55960457, 
            "point_id": 8, 
            "z": 85.94876861572266
        }, 
        {
            "y": 1292.0297386878126, 
            "x": -3582.61771003875, 
            "point_id": 9, 
            "z": 85.9641342163086
        }, 
        {
            "y": 1288.35822989875, 
            "x": -3578.91897957, 
            "point_id": 10, 
            "z": 86.12459564208984
        }, 
        {
            "y": 1288.7265160315626, 
            "x": -3578.1601905075, 
            "point_id": 11, 
            "z": 86.12911224365234
        }, 
        {
            "y": 1287.6928419527537, 
            "x": -3576.807087591798, 
            "point_id": 12, 
            "z": 86.20317282604864
        }, 
        {
            "y": 1286.0243675940626, 
            "x": -3574.71731941375, 
            "point_id": 13, 
            "z": 86.20663452148438
        }, 
        {
            "y": 1286.7863304846876, 
            "x": -3573.38919441375, 
            "point_id": 14, 
            "z": 86.30084228515625
        }, 
        {
            "y": 1286.5199730628126, 
            "x": -3573.089633866875, 
            "point_id": 15, 
            "z": 86.26866912841797
        }, 
        {
            "y": 1285.7326195471876, 
            "x": -3573.25052253875, 
            "point_id": 16, 
            "z": 86.25364685058594
        }, 
        {
            "y": 1284.7497093909376, 
            "x": -3573.037876054375, 
            "point_id": 17, 
            "z": 86.19708251953125
        }, 
        {
            "y": 1279.798903726875, 
            "x": -3567.301547929375, 
            "point_id": 18, 
            "z": 86.41533660888672
        }, 
        {
            "y": 1276.6432640784376, 
            "x": -3563.851352616875, 
            "point_id": 19, 
            "z": 86.46072387695312
        }, 
        {
            "y": 1268.790358805, 
            "x": -3554.9062842575, 
            "point_id": 20, 
            "z": 86.62606048583984
        }, 
        {
            "y": 1269.3011009925, 
            "x": -3553.537876054375, 
            "point_id": 21, 
            "z": 86.58422088623047
        }, 
        {
            "y": 1268.756911539375, 
            "x": -3552.90677253875, 
            "point_id": 22, 
            "z": 86.59705352783203
        }, 
        {
            "y": 1265.9953148596876, 
            "x": -3549.56009285125, 
            "point_id": 23, 
            "z": 86.67500305175781
        }, 
        {
            "y": 1265.341140055, 
            "x": -3548.787387773125, 
            "point_id": 24, 
            "z": 86.71348571777344
        }, 
        {
            "y": 1264.473781079718, 
            "x": -3548.860163462099, 
            "point_id": 25, 
            "z": 86.76638127199685
        }, 
        {
            "y": 1260.0745384925, 
            "x": -3543.605747148125, 
            "point_id": 26, 
            "z": 86.80860900878906
        }, 
        {
            "y": 1260.0280297034376, 
            "x": -3542.930454179375, 
            "point_id": 27, 
            "z": 86.79291534423828
        }, 
        {
            "y": 1259.242263101875, 
            "x": -3542.8183936325, 
            "point_id": 28, 
            "z": 86.85093688964844
        }, 
        {
            "y": 1256.9005882971876, 
            "x": -3540.206821366875, 
            "point_id": 29, 
            "z": 86.90399932861328
        }, 
        {
            "y": 1256.91487052375, 
            "x": -3539.636508866875, 
            "point_id": 30, 
            "z": 86.89977264404297
        }, 
        {
            "y": 1256.15998771125, 
            "x": -3539.428989335625, 
            "point_id": 31, 
            "z": 86.90433502197266
        }, 
        {
            "y": 1246.9535668128126, 
            "x": -3529.04153816375, 
            "point_id": 32, 
            "z": 87.00663757324219
        }, 
        {
            "y": 1247.17170646125, 
            "x": -3528.2637061325, 
            "point_id": 33, 
            "z": 87.02284240722656
        }, 
        {
            "y": 1246.0178978675, 
            "x": -3528.2812842575, 
            "point_id": 34, 
            "z": 87.26524353027344
        }, 
        {
            "y": 1240.488845133125, 
            "x": -3522.116001054375, 
            "point_id": 35, 
            "z": 87.11946105957031
        }, 
        {
            "y": 1239.235671305, 
            "x": -3522.701450273125, 
            "point_id": 36, 
            "z": 87.22277069091797
        }, 
        {
            "y": 1230.2007651554723, 
            "x": -3526.9223162280377, 
            "point_id": -1, 
            "z": 87.967593797353999
        }, 
        {
            "y": 1286.0634291099209, 
            "x": -3607.2126262837155, 
            "point_id": -1, 
            "z": 86.383525680085583
        }
    ], 
    "type": 27, 
    "id": 310, 
    "spanning_length": 97.91042448024163
}



def RectToPoly(groundTruth, shift=0, direction=0):
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

    return Polygon(
        [(pt1[0], pt1[1]),
        (pt2[0], pt2[1]),
        (pt3[0], pt3[1]),
        (pt4[0], pt4[1])])


from shapely.geometry import Polygon

if __name__ == '__main__':
    pointList = [[pt['x'], pt['y']] for pt in poly['points']]
    m_poly = Polygon([[p[0], p[1]] for p in pointList])
    gt_poly = RectToPoly(gt)

    fig, ax = plt.subplots()
    x,y = m_poly.exterior.xy
    ax.plot(x,y, 'b-')
    x,y = gt_poly.exterior.xy
    ax.plot(x,y, 'y-')
    print(m_poly)
    print(m_poly.intersection(gt_poly).area)

    plt.show()