"""Separate a locally deforming playfield from an independently rotating figure.

The estimator is intended for offline visual analysis/debugging. It does not
assume that the target follows the background and it does not extrapolate a
position when the residual signal disappears.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class DualMotionResult:
    valid: bool
    background_flow: np.ndarray
    residual_flow: np.ndarray
    residual_mask: np.ndarray
    rotation_deg: Optional[float]
    translation_px: Optional[tuple[float, float]]
    rotation_center: Optional[tuple[float, float]]
    inlier_ratio: float
    sample_count: int
    status: str
    background_model: str = "quadratic"


class DualMotionEstimator:
    """Estimate local background deformation and target-relative motion.

    `bbox` is the approximate target region in the current frame. Dense
    backward optical flow is measured from current -> previous. A quadratic
    flow surface is robustly fitted on an annulus around bbox to model the
    background's local wave-like deformation. That field is subtracted from
    the flow inside bbox. A similarity transform fitted to residual vectors
    then yields independent rotation and translation.
    """
    def __init__(self, margin: int = 36, sample_step: int = 3,
                 min_samples: int = 18, min_inlier_ratio: float = 0.18,
                 min_rotation_deg: float = 0.80):
        self.margin = max(12, int(margin))
        self.sample_step = max(1, int(sample_step))
        self.min_samples = max(8, int(min_samples))
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_rotation_deg = abs(float(min_rotation_deg))
        self._background_model_hint = None
        self._background_model_check_interval = 8
        self._background_fit_count = 0
        self.flow_scale = 0.75

    @staticmethod
    def _gray(frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _poly_features(x, y):
        # Coordinates are normalized around the target ROI before evaluation.
        # Quadratic terms model smooth waves; cubic terms capture asymmetric
        # curvature without introducing a high-dimensional unconstrained field.
        return np.column_stack((
            np.ones_like(x), x, y, x*x, x*y, y*y,
            x*x*x, x*x*y, x*y*y, y*y*y
        )).astype(np.float64)

    @staticmethod
    def _robust_fit(A, u, v, min_samples=18):
        keep=np.ones(len(u),dtype=bool)
        cu=cv=None
        for _ in range(4):
            if int(keep.sum())<max(min_samples,A.shape[1]+4):
                return None
            Ak=A[keep]
            normal=Ak.T@Ak
            ridge=max(1e-9,1e-8*float(np.trace(normal))/max(1,A.shape[1]))
            normal=normal+ridge*np.eye(A.shape[1])
            try:
                cu=np.linalg.solve(normal,Ak.T@u[keep])
                cv=np.linalg.solve(normal,Ak.T@v[keep])
            except np.linalg.LinAlgError:
                return None
            err=np.hypot(A@cu-u,A@cv-v)
            med=float(np.median(err[keep]))
            mad=float(np.median(np.abs(err[keep]-med)))+1e-6
            new_keep=err<=max(.85,med+2.8*1.4826*mad)
            if np.array_equal(new_keep,keep):
                break
            keep=new_keep
        if cu is None or int(keep.sum())<max(min_samples,A.shape[1]+4):
            return None
        if np.linalg.matrix_rank(A[keep].T@A[keep])<A.shape[1]:
            return None
        err=np.hypot(A@cu-u,A@cv-v)
        return cu,cv,keep,err

    def _fit_background(self, flow: np.ndarray, bbox):
        h,w=flow.shape[:2]
        x,y,bw,bh=map(int,bbox)
        cx,cy=x+bw/2.0,y+bh/2.0
        scale=max(12.0,float(max(bw,bh)))
        x0,y0=max(0,x-self.margin),max(0,y-self.margin)
        x1,y1=min(w,x+bw+self.margin),min(h,y+bh+self.margin)
        yy,xx=np.mgrid[y0:y1:self.sample_step,x0:x1:self.sample_step]
        outside=((xx<x-2)|(xx>=x+bw+2)|(yy<y-2)|(yy>=y+bh+2))
        sx,sy=xx[outside].astype(np.float64),yy[outside].astype(np.float64)
        if len(sx)<max(self.min_samples,24):
            return None
        xn,yn=(sx-cx)/scale,(sy-cy)/scale
        A=self._poly_features(xn,yn)
        u=flow[sy.astype(int),sx.astype(int),0].astype(np.float64)
        v=flow[sy.astype(int),sx.astype(int),1].astype(np.float64)
        good=np.isfinite(u)&np.isfinite(v)&(np.hypot(u,v)<35.0)
        A,u,v=A[good],u[good],v[good]
        if len(u)<max(self.min_samples,24):
            return None

        self._background_fit_count+=1
        check_model=(self._background_model_hint is None or
                     self._background_fit_count%self._background_model_check_interval==1)
        model=self._background_model_hint or "quadratic"
        chosen_terms=10 if model=="cubic" else 6
        if check_model and len(u)>=60:
            # A deterministic holdout prevents the higher-order model from
            # winning solely because it can fit the training annulus better.
            rng=np.random.default_rng(271828)
            order=rng.permutation(len(u))
            hold_n=max(12,int(round(.20*len(u))))
            hold=np.zeros(len(u),dtype=bool)
            hold[order[:hold_n]]=True
            train=~hold
            quad=self._robust_fit(A[train,:6],u[train],v[train],min_samples=18)
            cubic=self._robust_fit(A[train,:10],u[train],v[train],min_samples=22)
            model="quadratic"; chosen_terms=6
            if quad is not None and cubic is not None:
                qerr=np.hypot(A[hold,:6]@quad[0]-u[hold],A[hold,:6]@quad[1]-v[hold])
                cerr=np.hypot(A[hold,:10]@cubic[0]-u[hold],A[hold,:10]@cubic[1]-v[hold])
                qmed=float(np.median(qerr)); cmed=float(np.median(cerr))
                qp80=float(np.percentile(qerr,80)); cp80=float(np.percentile(cerr,80))
                if cmed<qmed*.90 and cp80<=qp80*1.03:
                    model="cubic"; chosen_terms=10
            self._background_model_hint=model

        final=self._robust_fit(A[:,:chosen_terms],u,v,
                               min_samples=max(18,chosen_terms+4))
        if final is None and chosen_terms==10:
            final=self._robust_fit(A[:,:6],u,v,min_samples=18)
            model="quadratic_fallback"
            self._background_model_hint="quadratic"
        if final is None:
            return None
        cu,cv,keep,_=final
        full_u=np.zeros(10,np.float64); full_v=np.zeros(10,np.float64)
        full_u[:len(cu)]=cu; full_v[:len(cv)]=cv
        return (full_u,full_v,cx,cy,scale,model,float(keep.mean()))

    def update(self, previous, current, bbox) -> DualMotionResult:
        prev_g,cur_g=self._gray(previous),self._gray(current)
        if prev_g.shape!=cur_g.shape:
            raise ValueError("相邻帧尺寸必须一致")
        h,w=cur_g.shape
        x,y,bw,bh=map(int,bbox)
        x0,y0=max(0,x),max(0,y)
        x1,y1=min(w,x+bw),min(h,y+bh)
        if x1-x0<8 or y1-y0<8:
            raise ValueError("bbox 太小或超出画面")

        # Only compute dense flow over the target and its surrounding annulus.
        pad=self.margin+6
        crop_x0=max(0,x0-pad); crop_y0=max(0,y0-pad)
        crop_x1=min(w,x1+pad); crop_y1=min(h,y1+pad)
        prev_crop=prev_g[crop_y0:crop_y1,crop_x0:crop_x1]
        cur_crop=cur_g[crop_y0:crop_y1,crop_x0:crop_x1]
        local_x,local_y=x0-crop_x0,y0-crop_y0
        local_w,local_h=x1-x0,y1-y0
        local_bbox=(local_x,local_y,local_w,local_h)
        flow_scale=float(self.flow_scale)
        if flow_scale < .999:
            sw=max(64,int(round(prev_crop.shape[1]*flow_scale)))
            sh=max(64,int(round(prev_crop.shape[0]*flow_scale)))
            prev_small=cv2.resize(prev_crop,(sw,sh),interpolation=cv2.INTER_AREA)
            cur_small=cv2.resize(cur_crop,(sw,sh),interpolation=cv2.INTER_AREA)
            backward_small=cv2.calcOpticalFlowFarneback(
                cur_small,prev_small,None,0.5,3,17,3,5,1.1,0)
            backward=cv2.resize(backward_small,(prev_crop.shape[1],prev_crop.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)/flow_scale
        else:
            backward=cv2.calcOpticalFlowFarneback(
                cur_crop,prev_crop,None,0.5,4,21,4,7,1.5,0)
        fit=self._fit_background(backward,local_bbox)
        if fit is None:
            zeros=np.zeros((h,w,2),np.float32)
            return DualMotionResult(False,zeros,zeros.copy(),np.zeros((h,w),np.uint8),
                                    None,None,None,0.0,0,"background_fit_failed","none")
        cu,cv,cx,cy,scale,background_model,_=fit
        ch,cw=cur_crop.shape
        gy,gx=np.mgrid[0:ch,0:cw]
        xn,yn=(gx-cx)/scale,(gy-cy)/scale
        features=self._poly_features(xn.ravel(),yn.ravel())
        bg_u=(features@cu).reshape(ch,cw).astype(np.float32)
        bg_v=(features@cv).reshape(ch,cw).astype(np.float32)
        bg_flow_crop=np.dstack((bg_u,bg_v))
        residual_crop=backward-bg_flow_crop

        roi_res=residual_crop[local_y:local_y+local_h,local_x:local_x+local_w]
        roi_img=cur_crop[local_y:local_y+local_h,local_x:local_x+local_w]
        gx_img=cv2.Sobel(roi_img,cv2.CV_32F,1,0,ksize=3)
        gy_img=cv2.Sobel(roi_img,cv2.CV_32F,0,1,ksize=3)
        edge=cv2.magnitude(gx_img,gy_img)
        mag=cv2.magnitude(roi_res[:,:,0],roi_res[:,:,1])
        edge_cut=max(7.0,float(np.percentile(edge,58)))
        flow_cut=max(.22,float(np.percentile(mag,62)))
        mask=(edge>=edge_cut)&(mag>=flow_cut)&np.isfinite(mag)
        ry,rx=np.where(mask)
        if len(rx)>900:
            strength=edge[ry,rx]*mag[ry,rx]
            chosen=np.argpartition(strength,-900)[-900:]
            rx,ry=rx[chosen],ry[chosen]
        count=len(rx)

        # Preserve the full-frame coordinate contract used by diagnostics.
        full_bg=np.zeros((h,w,2),np.float32)
        full_res=np.zeros((h,w,2),np.float32)
        full_mask=np.zeros((h,w),np.uint8)
        full_bg[crop_y0:crop_y1,crop_x0:crop_x1]=bg_flow_crop
        full_res[crop_y0:crop_y1,crop_x0:crop_x1]=residual_crop
        crop_mask=np.zeros((ch,cw),np.uint8)
        crop_mask[local_y:local_y+local_h,local_x:local_x+local_w][mask]=255
        full_mask[crop_y0:crop_y1,crop_x0:crop_x1]=crop_mask
        if count<self.min_samples:
            return DualMotionResult(False,full_bg,full_res,full_mask,None,
                                    None,None,0.0,count,"insufficient_residual_edges",
                                    background_model)

        src=np.column_stack((rx+local_x+crop_x0,ry+local_y+crop_y0)).astype(np.float32)
        vec=roi_res[ry,rx]
        dst=src+vec.astype(np.float32)
        M,inliers=cv2.estimateAffinePartial2D(
            src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.6,
            maxIters=1200,confidence=.99,refineIters=10
        )
        if M is None or inliers is None:
            return DualMotionResult(False,full_bg,full_res,full_mask,None,
                                    None,None,0.0,count,"similarity_fit_failed",
                                    background_model)
        inlier_ratio=float(inliers.mean())
        a,b=float(M[0,0]),float(M[1,0])
        angle=float(np.degrees(np.arctan2(b,a)))
        tx,ty=float(M[0,2]),float(M[1,2])
        R=M[:,:2].astype(np.float64)
        I_minus_R=np.eye(2)-R
        center=None
        if abs(angle)>=.15 and abs(np.linalg.det(I_minus_R))>1e-5:
            try:
                c=np.linalg.solve(I_minus_R,np.array([tx,ty],np.float64))
                if np.all(np.isfinite(c)) and -w<=c[0]<=2*w and -h<=c[1]<=2*h:
                    center=(float(c[0]),float(c[1]))
            except np.linalg.LinAlgError:
                pass
        valid=(inlier_ratio>=self.min_inlier_ratio and abs(angle)>=self.min_rotation_deg)
        status="relative_rotation" if valid else "background_or_weak_rotation"
        return DualMotionResult(valid,full_bg,full_res,full_mask,
                                angle if valid else None,(tx,ty) if valid else None,
                                center if valid else None,inlier_ratio,count,status,
                                background_model)


def draw_dual_motion(frame, bbox, result: DualMotionResult):
    out = frame.copy()
    x, y, w, h = map(int, bbox)
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 160, 0), 1)
    if result.residual_mask is not None:
        tint = np.zeros_like(out)
        tint[:, :, 1] = result.residual_mask
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    if result.rotation_center:
        p = tuple(map(lambda v: int(round(v)), result.rotation_center))
        cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    label = f"{result.status}/{result.background_model} rot={result.rotation_deg if result.rotation_deg is not None else 'n/a'}"
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20,20,20), 1, cv2.LINE_AA)
    return out