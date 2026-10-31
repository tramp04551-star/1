"""Moving bright/translucent target tracker with temporal background subtraction."""
import cv2
import numpy as np

class WhiteTargetTracker:
    def __init__(self, max_missing=180):
        self.kf=cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.processNoiseCov=np.eye(4,dtype=np.float32)*.02
        self.kf.measurementNoiseCov=np.eye(2,dtype=np.float32)*1.2
        self.kf.errorCovPost=np.eye(4,dtype=np.float32)
        self.bg=cv2.createBackgroundSubtractorMOG2(history=450,varThreshold=10,detectShadows=False)
        self.initialized=False; self.missing=0; self.max_missing=max_missing
        self.last_detected=False; self.last_box=None; self.warmup=0

    def reset(self):
        self.__init__(self.max_missing)

    @staticmethod
    def _components(mask, lo, hi):
        n, labels, stats, centers=cv2.connectedComponentsWithStats(mask,8)
        out=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            ratio=w/max(h,1)
            if lo<=area<=hi and .45<=ratio<=2.2:
                out.append((int(area),(float(centers[i][0]),float(centers[i][1])),(int(x),int(y),int(w),int(h))))
        return out

    def update(self, frame, threshold=215, min_area=700):
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        fg=self.bg.apply(frame,learningRate=.004); self.warmup+=1
        fg=cv2.medianBlur(fg,5)
        fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        white=cv2.inRange(gray,int(threshold),255)
        white &= cv2.inRange(hsv[:,:,1],0,95)
        white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        candidates=self._components(white,min_area,50000)
        if not candidates and self.initialized and self.warmup>25:
            candidates=self._components(fg,max(250,min_area//2),50000)
        px=py=None
        if self.initialized:
            pred=self.kf.predict(); px,py=float(pred[0,0]),float(pred[1,0])
        if candidates:
            if self.initialized:
                candidates=[c for c in candidates if (c[1][0]-px)**2+(c[1][1]-py)**2<260**2]
            if candidates:
                chosen=min(candidates,key=lambda c:(c[1][0]-px)**2+(c[1][1]-py)**2) if self.initialized else max(candidates,key=lambda c:c[0])
                _,(x,y),box=chosen
                if not self.initialized:
                    self.kf.statePost=np.array([[x],[y],[0],[0]],np.float32); self.initialized=True
                else: self.kf.correct(np.array([[x],[y]],np.float32))
                self.missing=0; self.last_detected=True; self.last_box=box
                return int(x),int(y)
        self.last_detected=False
        if self.initialized:
            self.missing+=1
            if self.missing<=self.max_missing: return int(px),int(py)
            self.initialized=False; self.last_box=None
        return None
