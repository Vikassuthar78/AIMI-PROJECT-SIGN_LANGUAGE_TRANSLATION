# =====================
# src/hand_tracking.py
# =====================
import cv2, numpy as np
import mediapipe as mp
from collections import deque

mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands  = mp.solutions.hands

class HandTracker:
    def __init__(self, det_conf=0.7, track_conf=0.7, max_hands=2, roi_hist=6, bbox_scale=2.2, pad=10):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.roi_hist = deque(maxlen=roi_hist)
        self.bbox_scale = bbox_scale
        self.pad = pad

    def _crop_one_hand(self, frame_bgr, hand_lms):
        H,W = frame_bgr.shape[:2]
        xs = np.array([lm.x for lm in hand_lms.landmark])*W
        ys = np.array([lm.y for lm in hand_lms.landmark])*H
        x0,x1 = xs.min(), xs.max()
        y0,y1 = ys.min(), ys.max()
        cx,cy = (x0+x1)/2, (y0+y1)/2
        side = max(x1-x0, y1-y0)*self.bbox_scale
        x0 = int(max(0, cx-side/2 - self.pad)); y0 = int(max(0, cy-side/2 - self.pad))
        x1 = int(min(W, cx+side/2 + self.pad)); y1 = int(min(H, cy+side/2 + self.pad))
        crop = frame_bgr[y0:y1, x0:x1]
        return crop

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        return res

    def draw(self, frame_bgr, res):
        if not res.multi_hand_landmarks: return frame_bgr
        for lms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr, lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        return frame_bgr

    def tile_two_hands(self, frame_bgr, res, out_size):
        """Return a single BGR square composed from 1 or 2 tight hand crops."""
        if not res.multi_hand_landmarks: return None
        crops=[]
        for hand in res.multi_hand_landmarks[:2]:
            c = self._crop_one_hand(frame_bgr, hand)
            if c is not None and c.size: crops.append(c)
        if len(crops)==0: return None
        if len(crops)==1:
            c=crops[0]; h,w=c.shape[:2]
            scale = out_size/max(h,w)
            c2 = cv2.resize(c,(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((out_size,out_size,3), dtype=np.uint8)
            y0 = (out_size - c2.shape[0])//2
            x0 = (out_size - c2.shape[1])//2
            canvas[y0:y0+c2.shape[0], x0:x0+c2.shape[1]] = c2
            return canvas
        else:
            L,R = crops[:2]
            target_h = out_size
            def rh(img):
                h,w = img.shape[:2]
                scale = target_h/h
                return cv2.resize(img,(max(1,int(w*scale)), target_h), interpolation=cv2.INTER_AREA)
            L = rh(L); R = rh(R)
            half = out_size//2
            L = cv2.resize(L,(half,out_size), interpolation=cv2.INTER_AREA)
            R = cv2.resize(R,(out_size-half,out_size), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((out_size,out_size,3), dtype=np.uint8)
            canvas[:,:half] = L; canvas[:,half:] = R
            return canvas
