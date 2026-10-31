import numpy as np
from tracker import WhiteTargetTracker

def test_white_target_acquisition():
    f=np.zeros((240,320,3),np.uint8); f[80:160,110:210]=255
    t=WhiteTargetTracker(); p=t.update(f,210,500)
    assert p is not None and 145<=p[0]<=175 and 105<=p[1]<=135
    assert t.last_detected

def test_short_occlusion_prediction():
    t=WhiteTargetTracker(max_missing=5)
    f=np.zeros((240,320,3),np.uint8); f[80:160,110:210]=255
    assert t.update(f,210,500) is not None
    assert t.update(np.zeros_like(f),210,500) is not None
    assert not t.last_detected

def test_reset_clears_tracking():
    t=WhiteTargetTracker(); f=np.zeros((240,320,3),np.uint8); f[80:160,110:210]=255
    t.update(f,210,500); t.reset()
    assert not t.initialized and t.last_box is None
