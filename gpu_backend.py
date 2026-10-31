"""Cross-vendor OpenCL mask processing with a safe OpenCV CPU fallback."""
import cv2
import numpy as np

_KERNEL = r"""
__kernel void white_mask(__global const uchar *rgb, __global uchar *mask,
                         const int count, const int threshold) {
    int i = get_global_id(0);
    if (i >= count) return;
    int r=(int)rgb[3*i], g=(int)rgb[3*i+1], b=(int)rgb[3*i+2];
    int hi=max(r,max(g,b)), lo=min(r,min(g,b));
    int lum=(299*r+587*g+114*b+500)/1000;
    int low_sat=(hi==0)||((hi-lo)*510<=hi*191);
    mask[i]=(lum>=threshold && low_sat)?(uchar)255:(uchar)0;
}
"""

class WhiteMaskProcessor:
    def __init__(self, prefer_gpu=True):
        self.cl=None; self.context=None; self.queue=None; self.program=None
        self.device_name=None; self.gpu_active=False; self.failure_reason=None
        if not prefer_gpu: return
        try:
            import pyopencl as cl
            devices=[]
            for platform in cl.get_platforms():
                try: devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
                except Exception: pass
            if not devices: raise RuntimeError("未找到支持 OpenCL 的 GPU")
            devices.sort(key=lambda d: ("intel" in d.name.lower(), d.name.lower()))
            device=devices[0]
            self.context=cl.Context([device]); self.queue=cl.CommandQueue(self.context,device)
            self.program=cl.Program(self.context,_KERNEL).build()
            self.cl=cl; self.device_name=device.name.strip(); self.gpu_active=True
        except Exception as exc:
            self.failure_reason=f"{type(exc).__name__}: {exc}"

    @property
    def backend_name(self):
        if self.gpu_active: return f"OpenCL GPU · {self.device_name}"
        if self.failure_reason: return f"CPU 回退 · {self.failure_reason}"
        return "OpenCV CPU"

    def white_mask(self,frame,threshold=215):
        if self.gpu_active:
            try: return self._opencl_mask(frame,threshold)
            except Exception as exc:
                self.failure_reason=f"OpenCL 运行失败: {type(exc).__name__}: {exc}"
                self.gpu_active=False
        return self._cpu_mask(frame,threshold)

    def _opencl_mask(self,frame,threshold):
        cl=self.cl; rgb=np.ascontiguousarray(frame,dtype=np.uint8); h,w=rgb.shape[:2]
        source=cl.Buffer(self.context,cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=rgb)
        output=cl.Buffer(self.context,cl.mem_flags.WRITE_ONLY,size=h*w)
        self.program.white_mask(self.queue,(h*w,),None,source,output,np.int32(h*w),np.int32(threshold))
        result=np.empty((h,w),dtype=np.uint8)
        cl.enqueue_copy(self.queue,result,output).wait()
        return result

    @staticmethod
    def _cpu_mask(frame,threshold):
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY); hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(gray,int(threshold),255)
        mask &= cv2.inRange(hsv[:,:,1],0,95)
        return mask
