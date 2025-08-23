import numpy as np
import cv2
import math
import time
import glob, os
import random
import psutil
import ctypes
from datetime import datetime

import torch

MaxSameKP_dist = 5  # pixels
MaxSameKP_angle = 10  # degrees

def unpackSIFTOctave(kp, XI=False):
    ''' Opencv packs the true octave, scale and layer inside kp.octave.
    This function calculates the depacket of information
    '''
    _octave = kp.octave
    octave = _octave & 0xFF
    layer = (_octave >> 8) & 0xFF
    if octave >= 128:
        octave |= -128
    if octave >= 0:
        scale = float(1 / (1 << octave))
    else:
        scale = float(1 << -octave)

    if XI:
        yi = (_octave >> 16) & 0xFF
        xi = yi / 255.0 - 0.5
        return octave, layer, scale, xi
    else:
        return octave, layer, scale


def packSIFTOctave(octave, layer, xi=0.0):
    po = octave & 0xFF
    pl = (layer & 0xFF) << 8
    pxi = round((xi + 0.5) * 255) & 0xFF
    pxi = pxi << 16
    return po + pl + pxi


def AngleDiff(a, b):
    ''' Computes the Angle Difference between a and b.
        0<=a,b<=360
    '''
    assert a >= 0 and a <= 360 and b >= 0 and b <= 360, 'a = ' + str(a) + ', b = ' + str(b)
    anglediff = abs(a - b) % 360
    if anglediff > 180:
        anglediff = 360 - anglediff
    return anglediff

def dist_pt_to_line(p, p1, p2):
    ''' Computes the distance of a point (p) to a line defined by two points (p1, p2). '''
    x0, y0 = np.float32(p[:2])
    x1, y1 = np.float32(p1[:2])
    x2, y2 = np.float32(p2[:2])
    dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))
    return dist


def ComposeAffineMaps(A_lhs, A_rhs):
    ''' Compute the composition of affine map:
        A = A_lhs ∘ A_rhs
    '''
    A = np.matmul(A_lhs[0:2, 0:2], A_rhs)
    A[:, 2] += A_lhs[:, 2]
    return A


def AffineArrayCoor(arr, A):
    if type(arr) is list:
        arr = np.array(arr).reshape(-1, 2)
    AA = A[0:2, 0:2]
    Ab = A[:, 2]
    arr_out = []
    for j in range(0, arr.shape[0]):
        arr_out.append(np.matmul(AA, np.array(arr[j, :])) + Ab)
    return np.array(arr_out)

def ComputePatches(kp_list, gpyr, radius_size=32):
    patches = []
    firstOctave = -1
    nOctaveLayers = 3
    flt_epsilon = 1.19209e-07
    new_radius_descr = (radius_size-1)/2
    interp_mode = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_CONSTANT
    r = new_radius_descr
    dim = np.int32(2 * r + 1)

    for i, kpt in enumerate(kp_list):
        octave, layer, scale = unpackSIFTOctave(kpt)
        step = kpt.size * scale * 0.5
        ptf = np.array(kpt.pt) * scale
        angle = 360.0 - kpt.angle
        angle = np.where(np.abs(angle - 360.0) < flt_epsilon, 0.0, angle)
        img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer]

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]]) / step
        Rptf = np.matmul(A, ptf)
        A = np.hstack([A, [[r-Rptf[0]], [r-Rptf[1]]]])
        img_warp = cv2.warpAffine(img, A, (dim, dim), flags=interp_mode, borderMode=border_mode)
        patches.append(img_warp.astype(np.float32))
    return patches

def ComputePatches2(kp_list, gpyr, radius_size=32):
    ''' Compute patches associated with each key point in kp_list
        Returns:
        img_list - list of patches.
        A_list - Affine Mapping List A，A(BackgroundImage)*1_{[0,2r]x[0,2r]} = patch  
        Ai_list - The inverse table of affine map above
    '''
    img_list = []
    img_raw_list = []
    A_list = []
    Ai_list = []
    firstOctave = -1
    nOctaveLayers = 3
    flt_epsilon = 1.19209e-07
    new_radius_descr = (radius_size-1)/2  # 15.5  # 29.5

    for i in range(0, np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= firstOctave and layer <= nOctaveLayers + 2, 'octave = ' + str(
            octave) + ', layer = ' + str(layer)
        # Formulas in opencv: kpt.size = sigma*powf(2. f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
        step = kpt.size * scale * 0.5  # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt) * scale  #
        angle = 360.0 - kpt.angle
        if (np.abs(angle - 360.0) < flt_epsilon):
            angle = 0.0
        img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer]

        r = new_radius_descr
        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]]) / step
        Rptf = np.matmul(A, ptf)
        x = Rptf[0] - r
        y = Rptf[1] - r
        A = np.hstack([A, [[-x], [-y]]])
        dim = np.int32(2 * r + 1)

        img_warp = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        # print('Octave =', octave,'; Layer =', layer, '; Scale =', scale,'; Angle =',angle)

        #A_raw = np.float32([[1, 0], [0, 1]]) / step
        #Rptf_raw = np.matmul(A_raw, ptf)
        #x = Rptf_raw[0] - r
        #y = Rptf_raw[1] - r
        #A_raw = np.hstack([A_raw, [[-x], [-y]]])
        #img_raw = cv2.warpAffine(img, A_raw, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        #img_raw_list.append(img_raw.astype(np.float32))

        #oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        #A = ComposeAffineMaps(A, oA)
        #Ai = cv2.invertAffineTransform(A)

        img_list.append(img_warp.astype(np.float32))
        #A_list.append(A)
        #Ai_list.append(Ai)
    return img_raw_list, img_list, A_list, Ai_list


def ComputeOnePatch(pt, size, angle, scale, img, radius_size=32):
    flt_epsilon = 1.19209e-07
    new_radius_descr = (radius_size-1)/2  # 15.5  # 29.5

    # Formulas in opencv: kpt.size = sigma*powf(2. f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
    step = size * scale * 0.5  # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
    ptf = np.array(pt) * scale  #
    angle = 360.0 - angle
    if (np.abs(angle - 360.0) < flt_epsilon):
        angle = 0.0

    r = new_radius_descr
    phi = np.deg2rad(angle)
    s, c = np.sin(phi), np.cos(phi)
    A = np.float32([[c, -s], [s, c]]) / step
    Rptf = np.matmul(A, ptf)
    x = Rptf[0] - r
    y = Rptf[1] - r
    A = np.hstack([A, [[-x], [-y]]])

    dim = np.int32(2 * r + 1)
    img = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return img.astype(np.float32)

def packSIFTAndBlur(src, x, y, deg, scale):
    """
    pt: the position of the key point 
    size: the range of the key point 
    angle: the angle of the key point
    response: the detector that can give a key point a stronger response can sometimes be understood as the probability of the actual existence of the feature 
    octave: It indicates the level where the key point is found, and it is always hoped to find the corresponding key point at the same level 
    class_id: It indicates which target the key point comes from.
    
    interest.txt: image_ID，x，y， orientation， scale (log2 units)
    """
    img = cv2.GaussianBlur(src, (0, 0), scale)
    p = cv2.KeyPoint()
    p.pt = [x, y]
    p.class_id = 0
    p.angle = deg
    return p, img


def is_cuda_available():
    try:
        cv2.cuda_GpuMat().upload(np.zeros((2, 2), dtype=np.uint8))
        return torch.cuda.is_available()
    except:
        return False

def gaussian_blur(src, ksize, sigx, sigy):
    if is_cuda_available():
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(src)
        gpu_blur = cv2.cuda.GaussianBlur(gpu_src, ksize, sigmaX=sigx, sigmaY=sigy)
        result_img = gpu_blur.download()
    else:
        result_img = cv2.GaussianBlur(src, ksize, sigmaX=sigx, sigmaY=sigy)
    return result_img



def buildGaussianPyramid(base, LastOctave, graydesc=True):
    '''
    Computing the Gaussian Pyramid as in opencv SIFT
    '''
    nOctaveLayers = 3
    sigma = 1.6
    firstOctave = -1
    if graydesc and len(base.shape) != 2:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    if firstOctave < 0:
        base = cv2.resize(base, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR_EXACT)
    rows, cols = base.shape[:2]

    nOctaves = np.round(np.log(np.float32(min(cols, rows))) / np.log(2.0) - 2) - firstOctave
    #nOctaves = min(nOctaves, LastOctave)
    nOctaves = np.int32(nOctaves)
    sig = ([sigma])
    k = np.float32(pow(2.0, 1.0 / np.float32(nOctaveLayers)))

    for i in range(1, nOctaveLayers + 3):
        sig_prev = pow(k, np.float32(i - 1)) * sigma
        sig_total = sig_prev * k
        sig += ([np.sqrt(sig_total * sig_total - sig_prev * sig_prev)])

    assert np.size(sig) == nOctaveLayers + 3

    pyr = []
    for o in range(nOctaves):
        for i in range(nOctaveLayers + 3):
            if o == 0 and i == 0:
                img = base
            elif i == 0:
                src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers]
                img = cv2.resize(src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            else:
                src = pyr[o * (nOctaveLayers + 3) + i - 1]
                img = cv2.GaussianBlur(src, (0, 0), sigmaX=sig[i], sigmaY=sig[i])
                # img = gaussian_blur(src, (0, 0), sig[i], sig[i])
            pyr.append(img)

    # pyr = []
    # for o in range(nOctaves):
    #     for i in range(nOctaveLayers + 3):
    #         pyr.append([])
    #
    # assert len(pyr) == nOctaves * (nOctaveLayers + 3)
    # for o in range(nOctaves):
    #     for i in range(nOctaveLayers + 3):
    #         if o == 0 and i == 0:
    #             pyr[o * (nOctaveLayers + 3) + i] = base.copy()
    #         elif i == 0:
    #             src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers]
    #             pyr[o * (nOctaveLayers + 3) + i] = cv2.resize(src, (0, 0), fx=0.5, fy=0.5,
    #                                                                      interpolation=cv2.INTER_NEAREST)
    #         else:
    #             src = pyr[o * (nOctaveLayers + 3) + i - 1]
    #             pyr[o * (nOctaveLayers + 3) + i] = cv2.GaussianBlur(src, (0, 0), sigmaX=sig[i],
    #                                                                            sigmaY=sig[i])
    return (pyr)


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def FirstOrderApprox_Homography(H0, X0=np.array([[0], [0], [1]])):
    ''' Computes the first order Taylor approximation (which is an affine map)
    of the Homography H0 centered at X0 (X0 is in homogeneous coordinates).
    '''
    H = H0.copy()
    col3 = np.matmul(H, X0)
    H[:, 2] = col3.reshape(3)
    A = np.zeros((2, 3), dtype=np.float32)
    A[0:2, 0:2] = H[0:2, 0:2] / H[2, 2] - np.array([H[0, 2] * H[2, 0:2], H[1, 2] * H[2, 0:2]]) / pow(H[2, 2], 2)
    A[:, 2] = H[0:2, 2] / H[2, 2] - np.matmul(A[0:2, 0:2], X0[0:2, 0] / X0[2, 0])
    return A


def AffineFit(Xi, Yi):
    assert np.shape(Xi)[0] == np.shape(Yi)[0] and np.shape(Xi)[1] == 2 and np.shape(Yi)[1] == 2
    n = np.shape(Xi)[0]
    A = np.zeros((2 * n, 6), dtype=np.float32)
    b = np.zeros((2 * n, 1), dtype=np.float32)
    for i in range(0, n):
        A[2 * i, 0] = Xi[i, 0]
        A[2 * i, 1] = Xi[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i + 1, 3] = Xi[i, 0]
        A[2 * i + 1, 4] = Xi[i, 1]
        A[2 * i + 1, 5] = 1.0

        b[2 * i, 0] = Yi[i, 0]
        b[2 * i + 1, 0] = Yi[i, 1]
    result = np.linalg.lstsq(A, b, rcond=None)
    return result[0].reshape((2, 3))


def SquareOrderedPts(hs, ws, CV=True):
    # Patch starts from the origin
    ws = ws - 1
    hs = hs - 1
    if CV:
        return [
            cv2.KeyPoint(x=0, y=0, _size=10, _angle=0, _response=1.0, _octave=0, _class_id=0),
            cv2.KeyPoint(x=ws, y=0, _size=10, _angle=0, _response=1.0, _octave=0, _class_id=0),
            cv2.KeyPoint(x=ws, y=hs, _size=10, _angle=0, _response=1.0, _octave=0, _class_id=0),
            cv2.KeyPoint(x=0, y=hs, _size=10, _angle=0, _response=1.0, _octave=0, _class_id=0)
        ]
    else:
        # return np.float32([ [0,0], [ws+1,0], [ws+1, hs+1], [0,hs+1] ])
        return np.float32([[0, 0], [ws, 0], [ws, hs], [0, hs]])


def Flatten2Pts(vec):
    X = np.zeros((np.int32(len(vec) / 2), 2), np.float32)
    X[:, 0] = vec[0::2]
    X[:, 1] = vec[1::2]
    return X


def Pts2Flatten(X):
    h, w = np.shape(X)[:2]
    vec = np.zeros((h * w), np.float32)
    vec[0::2] = X[:, 0]
    vec[1::2] = X[:, 1]
    return vec


def close_per(vec):
    return (np.array(tuple(vec) + tuple([vec[0]])))


def Check_FirstThreadTouch(GA):
    for file in glob.glob(GA.save_path + "/" + str(GA.GAid) + ".threadsdata"):
        if np.loadtxt(file) > 0.5:
            return True
        else:
            return False
    Set_FirstThreadTouch(GA, False)
    return False


def Set_FirstThreadTouch(GA, value):
    np.savetxt(GA.save_path + "/" + str(GA.GAid) + ".threadsdata", [value])


def get_big_epoch_number(GA):
    return np.loadtxt(GA.save_path + "/" + str(GA.GAid) + ".big_epoch")


def set_big_epoch_number(GA, value):
    # print(GA.save_path+"/big_epoch  -> "+ str(value))
    np.savetxt(GA.save_path + "/" + str(GA.GAid) + ".big_epoch", [value])


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def HumanElapsedTime(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return hours, minutes, seconds


def TouchDir(directory):
    ''' Creates a directory if it doesn't exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def OnlyUniqueMatches(goodM, KPlistQ, KPlistT, SpatialThres=4):
    ''' Filter out non unique matches with less similarity score
    '''
    uniqueM = []
    doubleM = np.zeros(len(goodM), dtype=np.bool)
    for i in range(0, len(goodM)):
        if doubleM[i]:
            continue
        bestsim = goodM[i].distance
        bestidx = i
        for j in range(i + 1, len(goodM)):
            if (cv2.norm(KPlistQ[goodM[i].queryIdx].pt, KPlistQ[goodM[j].queryIdx].pt) < SpatialThres \
                    and cv2.norm(KPlistT[goodM[i].trainIdx].pt, KPlistT[goodM[j].trainIdx].pt) < SpatialThres):
                doubleM[j] = True
                if bestsim < goodM[j].distance:
                    bestidx = j
                    bestsim = goodM[j].distance
        uniqueM.append(goodM[bestidx])
    return uniqueM


class CPPbridge(object):
    def __init__(self, libDApath):
        self.libDA = ctypes.cdll.LoadLibrary(libDApath)
        self.MatcherPtr = 0
        self.last_i1_list = []
        self.last_i2_list = []

        self.libDA.GeometricFilter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilter.restype = None

        self.libDA.GeometricFilterFromNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                        ctypes.c_bool]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.ArrayOfFilteredMatches.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.NumberOfFilteredMatches.argtypes = [ctypes.c_void_p]
        self.libDA.NumberOfFilteredMatches.restype = ctypes.c_int

        self.libDA.newMatcher.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libDA.newMatcher.restype = ctypes.c_void_p
        self.libDA.KnnMatcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libDA.KnnMatcher.restype = None

        self.libDA.GetData_from_QueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p]
        self.libDA.GetData_from_QueryNode.restype = None
        self.libDA.GetQueryNodeLength.argtypes = [ctypes.c_void_p]
        self.libDA.GetQueryNodeLength.restype = ctypes.c_int

        self.libDA.LastQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.LastQueryNode.restype = ctypes.c_void_p
        self.libDA.FirstQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.FirstQueryNode.restype = ctypes.c_void_p
        self.libDA.NextQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.NextQueryNode.restype = ctypes.c_void_p
        self.libDA.PrevQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.PrevQueryNode.restype = ctypes.c_void_p

        self.libDA.FastMatCombi.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_void_p]
        self.libDA.FastMatCombi.restype = None

    def GeometricFilter(self, scr_pts, im1, dts_pts, im2, Filer='ORSA_H', precision=10, verb=False):
        filercode = 0
        if Filer == 'ORSA_F':
            filercode = 1
        N = int(len(scr_pts) / 2)
        scr_pts = scr_pts.astype(ctypes.c_float)
        dts_pts = dts_pts.astype(ctypes.c_float)
        MatchMask = np.zeros(N, dtype=ctypes.c_bool)
        T = np.zeros(9, dtype=ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilter(scr_pts.ctypes.data_as(floatp), dts_pts.ctypes.data_as(floatp),
                                   MatchMask.ctypes.data_as(boolp), T.ctypes.data_as(floatp),
                                   N, w1, h1, w2, h2, filercode, ctypes.c_float(precision), verb)
        return MatchMask.astype(np.bool), T.astype(np.float).reshape(3, 3)

    def GeometricFilterFromMatcher(self, im1, im2, Filer='ORSA_H', precision=24, verb=False):
        filercode = 0
        if Filer == 'ORSA_F':
            filercode = 1
        T = np.zeros(9, dtype=ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        intp = ctypes.POINTER(ctypes.c_int)
        # boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilterFromNodes(self.MatcherPtr, T.ctypes.data_as(floatp),
                                            w1, h1, w2, h2, filercode, ctypes.c_float(precision), verb)

        NFM = self.libDA.NumberOfFilteredMatches(self.MatcherPtr)
        FM = np.zeros(3 * NFM, dtype=ctypes.c_int)
        self.libDA.ArrayOfFilteredMatches(self.MatcherPtr, FM.ctypes.data_as(intp))
        # print(NFM,FM)                
        Matches = [cv2.DMatch(FM[3 * i], FM[3 * i + 1], FM[3 * i + 2]) for i in range(0, NFM)]

        return Matches, T.astype(np.float).reshape(3, 3)

    def GetMatches_from_QueryNode(self, qn):
        N = self.libDA.GetQueryNodeLength(qn)
        if N > 0:
            Query_idx = np.zeros(1, dtype=ctypes.c_int)
            Target_idxs = np.zeros(N, dtype=ctypes.c_int)
            simis = np.zeros(N, dtype=ctypes.c_float)
            floatp = ctypes.POINTER(ctypes.c_float)
            intp = ctypes.POINTER(ctypes.c_int)
            self.libDA.GetData_from_QueryNode(qn, Query_idx.ctypes.data_as(intp), Target_idxs.ctypes.data_as(intp),
                                              simis.ctypes.data_as(floatp))
            return [cv2.DMatch(Query_idx[0], Target_idxs[i], simis[i]) for i in range(0, N)]
        else:
            return []

    def FirstLast_QueryNodes(self):
        return self.libDA.FirstQueryNode(self.MatcherPtr), self.libDA.LastQueryNode(self.MatcherPtr)

    def NextQueryNode(self, qn):
        return self.libDA.NextQueryNode(self.MatcherPtr, qn)

    def PrevQueryNode(self, qn):
        return self.libDA.PrevQueryNode(self.MatcherPtr, qn)

    def KnnMatch(self, QKPlist, Qdesc, TKPlist, Tdesc, FastCode):
        Nq = ctypes.c_int(np.shape(Qdesc)[0])
        Nt = ctypes.c_int(np.shape(Tdesc)[0])
        Qkps = np.array([x for kp in QKPlist for x in kp.pt], dtype=ctypes.c_float)
        Tkps = np.array([x for kp in TKPlist for x in kp.pt], dtype=ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        Qdesc = Qdesc.ravel().astype(ctypes.c_float)
        Tdesc = Tdesc.ravel().astype(ctypes.c_float)
        QdescPtr = Qdesc.ctypes.data_as(floatp)
        TdescPtr = Tdesc.ctypes.data_as(floatp)
        QkpsPtr = Qkps.ctypes.data_as(floatp)
        TkpsPtr = Tkps.ctypes.data_as(floatp)

        self.libDA.KnnMatcher(self.MatcherPtr, QkpsPtr, QdescPtr, Nq, TkpsPtr, TdescPtr, Nt, ctypes.c_int(FastCode))

    def CreateMatcher(self, desc_dim, k=1, sim_thres=0.7):
        self.MatcherPtr = self.libDA.newMatcher(k, desc_dim, sim_thres)

    def PrepareForFastMatCombi(self, len_i_list):
        self.last_i1_list = -1 * np.ones(shape=(len_i_list), dtype=ctypes.c_int)
        self.last_i2_list = -1 * np.ones(shape=(len_i_list), dtype=ctypes.c_int)

    def FastMatCombi(self, bP, i_list, ps1, j_list, ps2, MemStepImg, MemStepBlock):
        intp = ctypes.POINTER(ctypes.c_int)
        floatp = ctypes.POINTER(ctypes.c_float)
        i1_list = i_list.ctypes.data_as(intp)
        i2_list = j_list.ctypes.data_as(intp)
        ps1p = ps1.ctypes.data_as(floatp)
        ps2p = ps2.ctypes.data_as(floatp)
        bPp = bP.ctypes.data_as(floatp)

        last_i1_listp = self.last_i1_list.ctypes.data_as(intp)
        last_i2_listp = self.last_i2_list.ctypes.data_as(intp)

        self.libDA.FastMatCombi(ctypes.c_int(len(self.last_i1_list)), bPp,
                                i1_list, i2_list, ps1p, ps2p, ctypes.c_int(MemStepImg), last_i1_listp, last_i2_listp)

        self.last_i1_list = i_list.copy()
        self.last_i2_list = j_list.copy()
