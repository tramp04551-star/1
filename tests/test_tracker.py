import numpy as np
from gpu_backend import WhiteMaskProcessor
from tracker import WhiteTargetTracker

def test_cpu_mask_detects_white_and_rejects_color():
    frame=np.zeros((80,100,3),np.uint8); frame[10:30,10:30]=(255,255,255); frame[40:60,10:30]=(255,255,0)
    mask=WhiteMaskProcessor(prefer_gpu=False).white_mask(frame,215)
    assert np.all(mask[10:30,10:30]==255) and np.all(mask[40:60,10:30]==0)

def test_cpu_fallback_runs_without_gpu():
    p=WhiteMaskProcessor(prefer_gpu=False); frame=np.full((20,20,3),255,np.uint8)
    assert np.all(p.white_mask(frame,215)==255) and not p.gpu_active

def test_tracker_acquires_and_predicts_occlusion():
    f=np.zeros((240,320,3),np.uint8); f[80:160,110:210]=255
    t=WhiteTargetTracker(max_missing=5,use_gpu=False); point=t.update(f,210,500)
    assert point is not None and 145<=point[0]<=175 and 105<=point[1]<=135
    assert t.update(np.zeros_like(f),210,500) is not None

def test_reset_preserves_processor():
    t=WhiteTargetTracker(use_gpu=False); backend=t.mask_processor; t.reset()
    assert t.mask_processor is backend and not t.initialized
