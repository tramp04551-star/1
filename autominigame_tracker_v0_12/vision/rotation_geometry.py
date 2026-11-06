"""Rotation geometry for a rotating, fading figure over a deforming playfield.

The visible asymmetric silhouette supplies centroid observations. During
rotation, those centroids trace an arc around a fixed pivot; robust circle
fitting estimates that pivot. Once whiteness is lost, only background-
compensated tangential residual flow near the predicted silhouette edge may
advance the angle. No evidence means no update.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from collections import deque
from typing import Optional
import cv2
import numpy as np
from .dual_motion import DualMotionEstimator


@dataclass
class CentroidObservation:
    time_s: float
    center: tuple[float, float]
    area: float
    bbox: tuple[int, int, int, int]
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class OrbitEstimate:
    valid: bool
    center: Optional[tuple[float, float]]
    radius_px: Optional[float]
    sample_count: int
    inlier_count: int
    inlier_ratio: float
    angular_span_deg: float
    median_residual_px: Optional[float]
    status: str


class WhiteFigureCentroidDetector:
    def __init__(self, min_area=450, max_area=26000, min_side=20,
                 max_side=178, search_radius=115, min_value=218,
                 max_saturation=78):
        self.min_area=int(min_area); self.max_area=int(max_area)
        self.min_side=int(min_side); self.max_side=int(max_side)
        self.search_radius=float(search_radius)
        self.min_value=int(min_value); self.max_saturation=int(max_saturation)
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    def detect(self, frame, anchor, time_s=0.0):
        if frame is None or frame.ndim != 3:
            return None
        h,w=frame.shape[:2]; ax,ay=map(float,anchor); r=int(self.search_radius)
        x0=max(0,int(ax-r)); y0=max(0,int(ay-r))
        x1=min(w,int(ax+r)); y1=min(h,int(ay+r))
        if x1<=x0 or y1<=y0: return None
        hsv=cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([0,0,self.min_value],np.uint8),
                         np.array([179,self.max_saturation,255],np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernel)
        n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
        cand=[]
        for i in range(1,n):
            bx,by,bw,bh,area=map(int,stats[i])
            if not self.min_area<=area<=self.max_area: continue
            if not (self.min_side<=bw<=self.max_side and self.min_side<=bh<=self.max_side): continue
            cx,cy=centroids[i]; cx+=x0; cy+=y0
            dist=float(np.hypot(cx-ax,cy-ay))
            if dist>self.search_radius: continue
            fill=area/max(1,bw*bh); aspect=bw/max(1,bh)
            score=np.log1p(area)+2*fill-0.035*dist-0.7*abs(np.log(aspect))
            cand.append((score,i,(x0+bx,y0+by,bw,bh),(float(cx),float(cy)),area,fill))
        if not cand: return None
        cand.sort(reverse=True,key=lambda q:q[0])
        _,lab,bbox,center,area,fill=cand[0]
        full=np.zeros((h,w),np.uint8)
        roi=full[y0:y1,x0:x1]; roi[labels==lab]=255
        conf=float(np.clip(0.55+0.08*fill+0.02*np.log1p(area)-0.003*np.hypot(center[0]-ax,center[1]-ay),0,1))
        return CentroidObservation(float(time_s),center,float(area),bbox,conf,full)


def _circle3(p):
    a,b,c=p
    m=2*np.array([b-a,c-a],np.float64)
    rhs=np.array([b@b-a@a,c@c-a@a],np.float64)
    if abs(np.linalg.det(m))<1e-4: return None
    try: center=np.linalg.solve(m,rhs)
    except np.linalg.LinAlgError: return None
    radius=float(np.linalg.norm(a-center))
    return (center,radius) if np.isfinite(center).all() and np.isfinite(radius) else None


def fit_centroid_orbit(observations, residual_px=2.8, min_samples=8,
                       min_arc_deg=18.0, min_inlier_ratio=0.62,
                       max_radius_px=180.0,max_hypotheses=180):
    n=len(observations)
    def bad(status, ni=0, ratio=0.0, span=0.0, med=None):
        return OrbitEstimate(False,None,None,n,ni,ratio,span,med,status)
    if n<min_samples: return bad("insufficient_centroid_observations")
    pts=np.asarray([o.center for o in observations],np.float64)
    times=np.asarray([o.time_s for o in observations],np.float64)
    conf=np.asarray([o.confidence for o in observations],np.float64)
    if not np.isfinite(pts).all(): return bad("non_finite_centroids")
    rng=np.random.default_rng(20261104)
    total=n*(n-1)*(n-2)//6
    if total<=max_hypotheses:
        triples=list(combinations(range(n),3))
    else:
        triples=[tuple(rng.choice(n,3,False)) for _ in range(max(48,int(max_hypotheses)))]
    best=None
    for ids in triples:
        fit=_circle3(pts[list(ids)])
        if fit is None: continue
        c,r=fit
        if not 2<=r<=max_radius_px: continue
        err=np.abs(np.linalg.norm(pts-c,axis=1)-r)
        inside=err<=residual_px; count=int(inside.sum())
        if count<min_samples: continue
        key=(float(conf[inside].sum()),count,-float(np.median(err[inside])))
        if best is None or key>best[0]: best=(key,inside)
    if best is None: return bad("circle_ransac_failed")
    inside=best[1]
    def refine(mask):
        q=pts[mask]
        a=np.column_stack((2*q[:,0],2*q[:,1],np.ones(len(q))))
        b=q[:,0]**2+q[:,1]**2
        coef,*_=np.linalg.lstsq(a,b,rcond=None)
        c=coef[:2]; r2=float(coef[2]+c@c)
        return c,np.sqrt(r2) if r2>0 else -1
    c,r=refine(inside)
    if r<=0: return bad("invalid_refined_circle")
    err=np.abs(np.linalg.norm(pts-c,axis=1)-r); inside=err<=residual_px
    if inside.sum()>=min_samples: c,r=refine(inside)
    err=np.abs(np.linalg.norm(pts-c,axis=1)-r)
    idx=np.flatnonzero(inside)
    ang=np.unwrap(np.arctan2(pts[idx,1]-c[1],pts[idx,0]-c[0]))
    ang=ang[np.argsort(times[idx])]
    span=float(np.degrees(ang.max()-ang.min()))
    ratio=float(inside.mean()); med=float(np.median(err[inside]))
    if r<2.5: return bad("centroid_orbit_radius_too_small",int(inside.sum()),ratio,span,med)
    if span<min_arc_deg: return bad("centroid_orbit_arc_too_short",int(inside.sum()),ratio,span,med)
    if ratio<min_inlier_ratio: return bad("centroid_orbit_inlier_ratio_low",int(inside.sum()),ratio,span,med)
    return OrbitEstimate(True,(float(c[0]),float(c[1])),float(r),n,int(inside.sum()),ratio,span,med,"stable_centroid_orbit_centre")


@dataclass
class AngleMatch:
    valid: bool
    angle_deg: Optional[float]
    score: float
    margin: float
    status: str


def match_rotated_silhouette(template,current,pivot,roi_radius=100,
                             coarse_step=3.0,refine_step=0.5,
                             min_score=0.22,min_margin=0.012,
                             angle_hint_deg=None,search_radius_deg=24.0):
    if template is None or current is None: return AngleMatch(False,None,0,0,"missing_mask")
    if template.shape!=current.shape: raise ValueError("mask dimensions differ")
    h,w=template.shape[:2]; px,py=map(float,pivot); r=int(roi_radius)
    x0=max(0,int(px-r)); y0=max(0,int(py-r)); x1=min(w,int(px+r+1)); y1=min(h,int(py+r+1))
    a=(template[y0:y1,x0:x1]>0).astype(np.uint8)
    b=(current[y0:y1,x0:x1]>0).astype(np.uint8)
    if min(int(a.sum()),int(b.sum()))<100: return AngleMatch(False,None,0,0,"insufficient_silhouette_pixels")
    local=(px-x0,py-y0)
    def score(deg):
        m=cv2.getRotationMatrix2D(local,float(deg),1)
        rot=cv2.warpAffine(a,m,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)
        inter=np.count_nonzero((rot>0)&(b>0))
        return 2*inter/max(1,int(rot.sum()+b.sum()))
    if angle_hint_deg is None:
        coarse_angles=np.arange(-180,180,max(1,coarse_step))
    else:
        hint=float(angle_hint_deg)
        radius=max(4.0,float(search_radius_deg))
        coarse_angles=np.arange(hint-radius,hint+radius+1e-6,max(1,coarse_step))
    coarse=[(score(d),float(d)) for d in coarse_angles]
    coarse.sort(reverse=True)
    top=coarse[0][1]
    fine=[(score(d),float(d)) for d in np.arange(top-coarse_step,top+coarse_step+1e-6,max(.1,refine_step))]
    fine.sort(reverse=True); best,angle=fine[0]
    def dist(a,b): return abs((a-b+180)%360-180)
    alt=[s for s,d in fine+coarse if dist(d,angle)>=4]
    second=max(alt) if alt else 0
    margin=float(best-second)
    if best<min_score: return AngleMatch(False,None,best,margin,"silhouette_match_score_low")
    if margin<min_margin: return AngleMatch(False,None,best,margin,"silhouette_angle_ambiguous")
    return AngleMatch(True,angle,best,margin,"silhouette_angle_matched")


@dataclass
class AngularFlow:
    valid: bool
    delta_angle_deg: Optional[float]
    sample_count: int
    inlier_count: int
    mad_rad: Optional[float]
    status: str
    sector_count: int = 0
    sector_consensus_ratio: float = 0.0
    temporal_rescue: bool = False


def estimate_residual_edge_rotation(residual_flow,residual_mask,pivot,
                                    template_mask,predicted_angle_deg,
                                    band=15,min_samples=24,max_mad=.075,
                                    min_sectors=2,min_sector_consensus=.75,
                                    expected_delta_deg=None,
                                    temporal_tolerance_deg=3.0,
                                    require_background_separation=False):
    """Estimate rotation from contour-local residual flow in a compact ROI."""
    ys,xs=np.where(template_mask>0)
    if len(xs)<100:
        return AngularFlow(False,None,0,0,None,"template_mask_too_small")
    h,w=residual_mask.shape[:2]
    pad=max(12,int(band)+8)
    x0=max(0,int(xs.min())-pad); y0=max(0,int(ys.min())-pad)
    x1=min(w,int(xs.max())+pad+1); y1=min(h,int(ys.max())+pad+1)
    local_template=template_mask[y0:y1,x0:x1]
    local_pivot=(float(pivot[0])-x0,float(pivot[1])-y0)
    m=cv2.getRotationMatrix2D(local_pivot,float(predicted_angle_deg),1)
    pose=cv2.warpAffine(local_template,m,(x1-x0,y1-y0),flags=cv2.INTER_NEAREST)
    edge=cv2.morphologyEx(pose,cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    k=max(3,int(band)|1)
    edge=cv2.dilate(edge,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    yy,xx=np.where((residual_mask[y0:y1,x0:x1]>0)&(edge>0))
    n=len(xx)
    def bad(status,ni=0,mad=None,sectors=0,consensus=0.0):
        return AngularFlow(False,None,n,ni,mad,status,sectors,consensus)
    if n<min_samples: return bad("insufficient_boundary_flow")
    gx=xx.astype(float)+x0; gy=yy.astype(float)+y0
    dx=gx-pivot[0]; dy=gy-pivot[1]
    u=residual_flow[gy.astype(int),gx.astype(int),0].astype(float)
    v=residual_flow[gy.astype(int),gx.astype(int),1].astype(float)
    r2=dx*dx+dy*dy; speed=np.hypot(u,v)
    radial=np.abs(dx*u+dy*v)/(np.sqrt(np.maximum(r2,1))*np.maximum(speed,1e-6))
    omega=(dx*v-dy*u)/np.maximum(r2,1)
    keep=(r2>=64)&(r2<=40000)&(speed>=.18)&(radial<=.72)
    keep &= np.isfinite(omega)&(np.abs(omega)<=.45)
    if keep.sum()<min_samples: return bad("insufficient_tangential_flow",int(keep.sum()))
    dxk,dyk=dx[keep],dy[keep]; values=omega[keep]
    theta=(np.arctan2(dyk,dxk)+2*np.pi)%(2*np.pi)
    sector=np.floor(theta/(2*np.pi/4)).astype(int)
    sector_medians=[]; sector_values=[]; sector_ids=[]; sector_mads=[]; sector_counts=[]
    # Measure residual angular flow just outside the predicted target contour.
    # If neighboring background carries the same angular motion, it is not
    # sufficient evidence that a fading target edge is still observable.
    local_mask=residual_mask[y0:y1,x0:x1]
    outer_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,2*k+1),max(3,2*k+1)))
    outer=cv2.dilate(edge,outer_kernel)
    outer=(outer>0)&(edge==0)
    by,bx=np.where((local_mask>0)&outer)
    bg_sector_medians={}
    if len(bx)>=80:
        bgx=bx.astype(float)+x0; bgy=by.astype(float)+y0
        bdx=bgx-pivot[0]; bdy=bgy-pivot[1]
        bu=residual_flow[bgy.astype(int),bgx.astype(int),0].astype(float)
        bv=residual_flow[bgy.astype(int),bgx.astype(int),1].astype(float)
        br2=bdx*bdx+bdy*bdy; bspeed=np.hypot(bu,bv)
        bradial=np.abs(bdx*bu+bdy*bv)/(np.sqrt(np.maximum(br2,1))*np.maximum(bspeed,1e-6))
        bomega=(bdx*bv-bdy*bu)/np.maximum(br2,1)
        bgood=(br2>=64)&(br2<=40000)&(bspeed>=.18)&(bradial<=.72)&np.isfinite(bomega)&(np.abs(bomega)<=.45)
        if int(bgood.sum())>=50:
            btheta=(np.arctan2(bdy[bgood],bdx[bgood])+2*np.pi)%(2*np.pi)
            bsector=np.floor(btheta/(2*np.pi/4)).astype(int)
            for sid in range(4):
                vals=bomega[bgood][bsector==sid]
                if len(vals)>=8:
                    bg_sector_medians[sid]=float(np.median(vals))
    for sid in range(4):
        vals=values[sector==sid]
        if len(vals)>=4:
            med=float(np.median(vals)); mad_s=float(np.median(np.abs(vals-med)))
            if mad_s<=max(.10,max_mad*1.5):
                sector_medians.append(med); sector_values.append(vals); sector_ids.append(sid)
                sector_mads.append(mad_s); sector_counts.append(int(len(vals)))
    if not sector_medians:
        return bad("insufficient_spatial_sector_support",int(keep.sum()),sectors=0)
    sector_medians=np.asarray(sector_medians,dtype=np.float64)
    preliminary=float(np.median(sector_medians))
    tolerance=max(.025,.50*abs(preliminary))
    agree=np.abs(sector_medians-preliminary)<=tolerance
    consensus=float(agree.mean())
    accepted_ids=[sector_ids[i] for i in range(len(sector_ids)) if agree[i]]
    spatial_ok=(int(agree.sum())>=min_sectors and consensus>=min_sector_consensus)
    if spatial_ok and len(accepted_ids)==2:
        opposed=min((accepted_ids[0]-accepted_ids[1])%4,
                    (accepted_ids[1]-accepted_ids[0])%4)>=2
        spatial_ok=opposed
    temporal_rescue=False
    if spatial_ok:
        accepted=np.concatenate([sector_values[i] for i in range(len(sector_values)) if agree[i]])
    else:
        # During the final fade, one coherent contour arc may be all that
        # remains. Accept it only when it agrees with recent measured angular
        # velocity and is strongly tangential; do not let background sectors
        # dilute or replace the local edge evidence.
        temporal_candidates=[]
        if expected_delta_deg is not None:
            min_sector_points=max(12,min_samples//2)
            for i,(med_s,mad_s,count_s) in enumerate(zip(sector_medians,sector_mads,sector_counts)):
                delta_s=float(np.degrees(med_s))
                sector_radial=radial[keep][sector==sector_ids[i]]
                radial_median=float(np.median(sector_radial)) if len(sector_radial) else 1.0
                radial_p75=float(np.percentile(sector_radial,75)) if len(sector_radial) else 1.0
                bg_delta=bg_sector_medians.get(sector_ids[i])
                background_separated=(not require_background_separation or bg_delta is None or
                    abs(med_s-bg_delta)>=max(.008,.25*abs(med_s)))
                if (count_s>=min_sector_points and mad_s<=max_mad and
                    radial_median<=.30 and radial_p75<=.38 and background_separated and
                    abs(delta_s-float(expected_delta_deg))<=temporal_tolerance_deg and
                    np.sign(delta_s)==np.sign(float(expected_delta_deg))):
                    temporal_candidates.append(i)
        if temporal_candidates:
            best=min(temporal_candidates,key=lambda i:(
                abs(np.degrees(sector_medians[i])-float(expected_delta_deg)),-sector_counts[i]))
            accepted=sector_values[best]; temporal_rescue=True
        else:
            global_med=float(np.median(values))
            global_mad=float(np.median(np.abs(values-global_med)))
            global_delta=float(np.degrees(global_med))
            same_sign=np.flatnonzero(np.sign(sector_medians)==np.sign(global_med))
            temporal_ok=(expected_delta_deg is not None and
                         abs(global_delta-float(expected_delta_deg))<=temporal_tolerance_deg)
            if not (temporal_ok and len(same_sign)>=2 and global_mad<=max_mad):
                return bad("angular_sector_consensus_low",int(keep.sum()),
                           sectors=len(sector_medians),consensus=consensus)
            accepted=values; temporal_rescue=True
    selected_sector_ids=(
        [sector_ids[i] for i in range(len(sector_ids)) if agree[i]]
        if spatial_ok else ([sector_ids[temporal_candidates[0]]] if temporal_rescue and 'temporal_candidates' in locals() and temporal_candidates else [])
    )
    if require_background_separation and len(bg_sector_medians)>=2 and selected_sector_ids:
        comparable=[bg_sector_medians[sid] for sid in selected_sector_ids if sid in bg_sector_medians]
        if comparable:
            bg_med=float(np.median(comparable))
            accepted_med=float(np.median(accepted))
            if abs(accepted_med-bg_med)<max(.008,.25*abs(accepted_med)):
                return bad("target_flow_not_separated_from_background",int(keep.sum()),
                           sectors=len(sector_medians),consensus=consensus)
    med=float(np.median(accepted)); mad=float(np.median(np.abs(accepted-med)))
    cutoff=max(.025,3*1.4826*mad)
    inliers=np.abs(accepted-med)<=cutoff
    if int(inliers.sum())<min_samples:
        return bad("incoherent_angular_flow",int(inliers.sum()),mad,len(sector_medians),consensus)
    med=float(np.median(accepted[inliers])); mad=float(np.median(np.abs(accepted[inliers]-med)))
    if mad>max_mad:
        return bad("angular_flow_mad_too_high",int(inliers.sum()),mad,len(sector_medians),consensus)
    delta=float(np.degrees(med))
    if abs(delta)<.05:
        return bad("angular_flow_below_noise",int(inliers.sum()),mad,len(sector_medians),consensus)
    return AngularFlow(True,delta,n,int(inliers.sum()),mad,"residual_edge_rotation",
                       len(sector_medians),consensus,temporal_rescue)


@dataclass
class FadePose:
    valid: bool
    angle_deg: float
    centroid: Optional[tuple[float,float]]
    delta_deg: Optional[float]
    confidence: float
    status: str
    sector_count: int = 0
    sector_consensus_ratio: float = 0.0
    temporal_rescue: bool = False
    background_model: str = "unknown"


class FadeRotationTracker:
    def __init__(self,pivot,template_mask,seed_centroid,angle_deg,
                 direction=None,min_confidence=.35,max_failures=4,
                 max_fade_gap_s=.75,max_angular_deviation_deg=3.0):
        self.pivot=tuple(map(float,pivot)); self.template=template_mask.copy()
        self.seed=tuple(map(float,seed_centroid)); self.angle=float(angle_deg)
        self.direction=int(np.sign(direction)) if direction else None
        self.min_conf=float(min_confidence); self.max_fail=max(1,int(max_failures))
        self.max_fade_gap_s=float(max_fade_gap_s)
        self.max_angular_deviation_deg=float(max_angular_deviation_deg)
        self.last_visible_time=None
        self.delta_history=deque(maxlen=5)
        self.failures=0; self.lost=False
        ys,xs=np.where(self.template>0)
        if len(xs)<100: raise ValueError("seed mask too small")
        h,w=self.template.shape; pad=38
        x0=max(0,int(xs.min())-pad); y0=max(0,int(ys.min())-pad)
        x1=min(w,int(xs.max())+pad+1); y1=min(h,int(ys.max())+pad+1)
        self.bbox=(x0,y0,x1-x0,y1-y0)
        self.motion=DualMotionEstimator(margin=40,sample_step=3,min_samples=18,
                                        min_inlier_ratio=.18,min_rotation_deg=.8)
        self.last_background_model="unknown"
    def set_visible_pose(self,angle,direction=None,time_s=None):
        self.angle=float(angle)
        if direction: self.direction=int(np.sign(direction))
        if time_s is not None: self.last_visible_time=float(time_s)
        self.failures=0; self.lost=False
    def note_visible_evidence(self,time_s):
        # A detected remnant of the white target is weaker than a successful
        # silhouette-angle match, but it confirms that the fade is still in
        # progress and extends the short residual-flow observation window.
        if not self.lost:
            self.last_visible_time=float(time_s)
    def _fail(self,status):
        self.failures+=1
        if self.failures>=self.max_fail: self.lost=True
        return FadePose(False,self.angle,None,None,0,
                        "fade_tracker_lost:"+status if self.lost else status)
    def update(self,previous,current,time_s=None):
        if self.lost: return FadePose(False,self.angle,None,None,0,"fade_tracker_latched_lost")
        if (time_s is not None and self.last_visible_time is not None
                and float(time_s)-self.last_visible_time > self.max_fade_gap_s):
            self.lost=True
            return FadePose(False,self.angle,None,None,0,"fade_tracker_lost:visible_evidence_timeout")
        dm=self.motion.update(previous,current,self.bbox)
        self.last_background_model=dm.background_model
        # `dm.valid` describes whether a single global similarity transform
        # explains most residual pixels. During fade, the background may still
        # dominate that global fit even while the target's local contour carries
        # coherent tangential motion. Keep the background-compensated dense flow
        # and let the contour-local angular estimator make its own decision.
        if dm.status == "background_fit_failed" or dm.residual_flow is None:
            return self._fail(dm.status)
        if dm.residual_mask is None or not np.any(dm.residual_mask):
            return self._fail(dm.status if dm.status else "no_residual_mask")
        expected_delta=(float(np.median(np.asarray(self.delta_history,dtype=np.float64)))
                        if len(self.delta_history)>=2 else None)
        af=estimate_residual_edge_rotation(dm.residual_flow,dm.residual_mask,
                                           self.pivot,self.template,self.angle,
                                           max_mad=.12,
                                           expected_delta_deg=expected_delta,
                                           temporal_tolerance_deg=self.max_angular_deviation_deg,
                                           require_background_separation=True)
        if not af.valid: return self._fail(af.status)
        delta=float(af.delta_angle_deg)
        if abs(delta)>12: return self._fail("angular_step_too_large")
        if self.direction and int(np.sign(delta))!=self.direction: return self._fail("rotation_direction_flip")
        if len(self.delta_history) >= 2:
            expected=float(np.median(np.asarray(self.delta_history,dtype=np.float64)))
            if abs(delta-expected)>self.max_angular_deviation_deg:
                return self._fail("angular_temporal_inconsistency")
        new_angle=self.angle+delta
        M=cv2.getRotationMatrix2D(self.pivot,new_angle,1)
        p=M@np.array([self.seed[0],self.seed[1],1.])
        # During blending the local flow becomes noisier. Use a soft robust
        # coherence score, while the temporal median and direction gates reject
        # isolated background-flow excursions.
        coherence=float(np.exp(-af.mad_rad/.08))
        support=min(1,af.inlier_count/120)
        conf=float(coherence*support)
        if conf<self.min_conf: return self._fail("fade_rotation_confidence_low")
        self.angle=new_angle; self.failures=0
        if time_s is not None:
            self.last_visible_time=float(time_s)
        self.delta_history.append(delta)
        return FadePose(True,self.angle,(float(p[0]),float(p[1])),delta,conf,
                        "fade_rotation_flow_tracked",af.sector_count,
                        af.sector_consensus_ratio,af.temporal_rescue,
                        dm.background_model)

@dataclass
class RotationGeometryResult:
    valid: bool
    point: Optional[tuple[float,float]]
    angle_deg: Optional[float]
    confidence: float
    source: str
    status: str
    pivot: Optional[tuple[float,float]] = None
    orbit_samples: int = 0


class RotationGeometryTracker:
    """Runtime adapter: calibrate a fixed pivot while visible, then track fade."""
    def __init__(self,pivot_gate_px=18.0,max_observations=80,
                 max_fade_gap_s=.75,min_fade_confidence=.55):
        self.pivot_gate_px=float(pivot_gate_px)
        self.max_observations=max(12,int(max_observations))
        self.max_fade_gap_s=float(max_fade_gap_s)
        self.min_fade_confidence=float(min_fade_confidence)
        self.reset()

    def reset(self):
        self.observations=deque(maxlen=self.max_observations)
        self.seed_mask=None; self.seed_center=None; self.seed_point=None
        self.seed_bbox_area=None; self.last_mask=None
        self.anchor=None; self.pivot=None; self.last_angle=None
        self.direction=None; self.fade=None; self.previous_frame=None
        self.last_status="search"; self.rejected_off_orbit=0; self.orbit=None
        self.observation_count=0; self.last_fit_observation_count=0
        self._last_valid_result=None

    @staticmethod
    def _centroid(detection):
        mask=(detection.mask>0).astype(np.uint8)
        moments=cv2.moments(mask,binaryImage=True)
        if moments["m00"]<=0: return None
        return (float(moments["m10"]/moments["m00"]),
                float(moments["m01"]/moments["m00"]))

    @staticmethod
    def _angle_delta(a,b):
        return (float(a)-float(b)+180.0)%360.0-180.0

    def _append_visible(self,detection,time_s):
        center=self._centroid(detection)
        if center is None: return False
        h,w=detection.mask.shape[:2]
        bbox_area=float(detection.bbox[2]*detection.bbox[3])
        if bbox_area/max(1.0,float(h*w)) < .006:
            self.rejected_off_orbit+=1; self.last_status="component_too_small_for_rotation_seed"
            return False
        if self.seed_bbox_area is not None:
            ratio=bbox_area/max(1.0,self.seed_bbox_area)
            if not .42<=ratio<=2.25:
                self.rejected_off_orbit+=1; self.last_status="component_scale_mismatch"
                return False
        if self.anchor is not None and self.pivot is None:
            distance=float(np.hypot(center[0]-self.anchor[0],center[1]-self.anchor[1]))
            anchor_gate=max(34.0,.55*np.hypot(detection.bbox[2],detection.bbox[3]))
            if distance>anchor_gate:
                self.rejected_off_orbit+=1; self.last_status="off_anchor_component_rejected"
                return False
        if self.pivot is not None:
            distance=float(np.hypot(center[0]-self.pivot[0],center[1]-self.pivot[1]))
            if distance>max(self.pivot_gate_px,28.0):
                self.rejected_off_orbit+=1; self.last_status="off_orbit_component_rejected"
                return False
        if self.seed_mask is None:
            self.seed_mask=detection.mask.copy(); self.seed_center=center
            self.seed_point=tuple(map(float,detection.safe_point)); self.seed_bbox_area=bbox_area
            self.anchor=center
        self.observations.append(CentroidObservation(
            float(time_s),center,float(detection.area),tuple(map(int,detection.bbox)),
            float(np.clip(detection.score/2.5,.2,1.0)),detection.mask.copy()))
        self.observation_count+=1
        should_fit=(self.orbit is None or not self.orbit.valid or
                    self.observation_count-self.last_fit_observation_count>=8)
        if should_fit:
            self.orbit=fit_centroid_orbit(list(self.observations),max_hypotheses=180)
            self.last_fit_observation_count=self.observation_count
            if self.orbit.valid:
                self.pivot=self.orbit.center
            else:
                pts=np.asarray([o.center for o in self.observations],np.float64)
                med=np.median(pts,axis=0); residual=np.linalg.norm(pts-med,axis=1)
                mad=float(np.median(np.abs(residual-np.median(residual))))
                gate=max(7.0,3.5*1.4826*mad); inside=residual<=gate
                if int(inside.sum())>=8 and float(inside.mean())>=.75:
                    spread=float(np.median(residual[inside]))
                    if spread<=12.0:
                        self.pivot=(float(np.median(pts[inside,0])),float(np.median(pts[inside,1])))
                        self.orbit=OrbitEstimate(True,self.pivot,0.0,len(pts),int(inside.sum()),
                            float(inside.mean()),0.0,spread,"stable_centroid_pivot")
        return True

    def _ensure_fade(self,angle,time_s):
        if self.fade is None and self.direction:
            self.fade=FadeRotationTracker(self.pivot,self.seed_mask,self.seed_point,
                float(angle),direction=self.direction,max_fade_gap_s=self.max_fade_gap_s,
                max_failures=10)
        if self.fade is not None:
            self.fade.pivot=tuple(map(float,self.pivot))
            self.fade.set_visible_pose(float(angle),self.direction,time_s)

    def recent_result(self,time_s,max_age_s=.10):
        if self._last_valid_result is None: return None
        result,stamp=self._last_valid_result
        age=float(time_s)-stamp
        if age<0.0 or age>float(max_age_s): return None
        return result

    def advance_frame(self,frame):
        self.previous_frame=frame.copy()

    def sync_from_point(self,point,time_s=None):
        """Use a healthy appearance track point as a low-cost angular observation."""
        if self.pivot is None or self.seed_point is None or point is None: return False
        r0=np.asarray(self.seed_point,dtype=np.float64)-np.asarray(self.pivot)
        r1=np.asarray(point,dtype=np.float64)-np.asarray(self.pivot)
        n0=float(np.linalg.norm(r0)); n1=float(np.linalg.norm(r1))
        if n0<8.0 or not .70*n0<=n1<=1.30*n0: return False
        delta=float(np.degrees(np.arctan2(
            r0[0]*r1[1]-r0[1]*r1[0],float(np.dot(r0,r1)))))
        if self.last_angle is not None:
            expected=self._angle_delta(delta,self.last_angle)
            if abs(expected)>15.0: return False
            if self.direction and abs(expected)>.2 and int(np.sign(expected))!=self.direction:
                return False
            self.last_angle=delta
            if self.fade is not None:
                self.fade.angle=delta
                if abs(expected)>.08: self.fade.delta_history.append(expected)
        else:
            self.last_angle=delta
        if self.fade is not None and time_s is not None:
            self.fade.last_visible_time=float(time_s)
        return True

    def update(self,frame,detection=None,time_s=None):
        if time_s is None:
            import time as _time
            time_s=_time.monotonic()
        time_s=float(time_s)
        result=RotationGeometryResult(False,None,self.last_angle,0.0,"none",
            "pivot_not_ready",self.pivot,len(self.observations))
        accepted=detection is not None
        if accepted: accepted=self._append_visible(detection,time_s)
        if self.pivot is not None and self.seed_mask is not None and accepted:
            first_match=self.last_mask is None
            template=self.last_mask if self.last_mask is not None else self.seed_mask
            match=match_rotated_silhouette(
                template,detection.mask,self.pivot,roi_radius=88,
                coarse_step=2.0 if first_match else 1.5,refine_step=.25,
                angle_hint_deg=None if first_match else 0.0,
                search_radius_deg=180.0 if first_match else 14.0,
                min_score=.16,min_margin=.0025)
            if match.valid:
                delta=float(match.angle_deg)
                if abs(delta)>.15: self.direction=int(np.sign(delta))
                self.last_angle=delta if self.last_angle is None else self.last_angle+delta
                self.last_mask=detection.mask.copy()
                self._ensure_fade(self.last_angle,time_s)
                if self.fade is not None and not first_match and abs(delta)>.15:
                    self.fade.delta_history.append(delta)
                result=RotationGeometryResult(False,None,self.last_angle,match.score,
                    "silhouette","visible_incremental_silhouette_matched",self.pivot,len(self.observations))
            elif self.fade is not None and self.previous_frame is not None:
                self.fade.note_visible_evidence(time_s)
                pose=self.fade.update(self.previous_frame,frame,time_s)
                result=self._fade_result(pose)
            else:
                result=RotationGeometryResult(False,None,self.last_angle,match.score,
                    "none",match.status,self.pivot,len(self.observations))
        elif self.fade is not None and self.previous_frame is not None:
            pose=self.fade.update(self.previous_frame,frame,time_s)
            result=self._fade_result(pose)
        elif self.pivot is None:
            result=RotationGeometryResult(False,None,self.last_angle,0.0,"none",
                self.orbit.status if self.orbit is not None else "pivot_not_ready",
                None,len(self.observations))
        if self.pivot is not None and self.seed_mask is not None and not accepted and self.fade is None:
            result=RotationGeometryResult(False,None,self.last_angle,0.0,"none",
                "rotation_direction_not_known",self.pivot,len(self.observations))
        self.previous_frame=frame.copy()
        self.last_status=result.status
        if result.valid: self._last_valid_result=(result,time_s)
        return result

    def _fade_result(self,pose):
        if pose.valid and pose.centroid is not None and pose.confidence>=self.min_fade_confidence:
            return RotationGeometryResult(True,pose.centroid,pose.angle_deg,pose.confidence,
                "rotation_geometry","fade_rotation_flow_tracked",self.pivot,len(self.observations))
        return RotationGeometryResult(False,None,pose.angle_deg,pose.confidence,"none",
            pose.status,self.pivot,len(self.observations))
