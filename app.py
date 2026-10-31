"""MapleStory mini-game helper: track the fading shape, detect success, confirm, and wait again."""
import threading,time,tkinter as tk
from tkinter import ttk
import cv2,mss,numpy as np,pyautogui
from tracker import WhiteTargetTracker
pyautogui.FAILSAFE=True
pyautogui.PAUSE=0

def success_banner(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([15,120,232]),np.array([42,255,255]))
    h,w=mask.shape; roi=mask[int(h*.22):int(h*.82),int(w*.18):int(w*.82)]
    roi=cv2.morphologyEx(roi,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    # Gold success lettering is bright and concentrated near the center;
    # the rock-texture background is darker and is excluded by V >= 232.
    return cv2.countNonZero(roi)>650

def confirm_button(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([4,110,120]),np.array([28,255,255]))
    h,w=mask.shape; ox,oy=int(w*.45),int(h*.55); roi=mask[oy:,ox:]
    roi=cv2.morphologyEx(roi,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    cs,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    for c in cs:
        x,y,cw,ch=cv2.boundingRect(c); area=cv2.contourArea(c)
        if 25<=cw<=220 and 12<=ch<=90 and area>=120:
            found.append((area,ox+x+cw//2,oy+y+ch//2))
    if not found:return None
    _,x,y=max(found); return int(x),int(y)

class App:
    def __init__(self,root):
        self.root=root; root.title("冒险岛小游戏助手"); root.geometry("430x470"); root.resizable(False,False)
        self.tracker=WhiteTargetTracker(); self.stop_event=threading.Event(); self.running=False
        self.threshold=tk.IntVar(value=215); self.area=tk.IntVar(value=700); self.smooth=tk.DoubleVar(value=.55)
        self.follow=tk.BooleanVar(value=True); self.auto=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="状态：已停止")
        self.backend_status=tk.StringVar(value=f"图像掩膜：{self.tracker.backend_name}")
        self.state="monitor"; self.success_hits=0; self.dialog_hits=0
        self.build(); root.protocol("WM_DELETE_WINDOW",self.close)

    def build(self):
        ttk.Label(self.root,text="冒险岛小游戏助手",font=("Microsoft YaHei UI",17,"bold")).pack(pady=14)
        ttk.Label(self.root,text="白色目标锁定 · 透明期预测 · 完成确认 · 下一轮监控",foreground="#666").pack()
        box=ttk.LabelFrame(self.root,text="目标识别"); box.pack(fill="x",padx=16,pady=8)
        self.slider(box,"白色阈值",self.threshold,150,255); self.slider(box,"目标面积",self.area,100,5000)
        box=ttk.LabelFrame(self.root,text="鼠标控制"); box.pack(fill="x",padx=16,pady=8)
        ttk.Checkbutton(box,text="鼠标持续跟随目标",variable=self.follow).pack(anchor="w",padx=10)
        self.slider(box,"平滑度",self.smooth,.1,1,.05)
        box=ttk.LabelFrame(self.root,text="自动流程"); box.pack(fill="x",padx=16,pady=8)
        ttk.Checkbutton(box,text="检测成功后自动点击确认并等待下一轮",variable=self.auto).pack(anchor="w",padx=10,pady=8)
        row=ttk.Frame(self.root); row.pack(fill="x",padx=16,pady=12)
        self.start_btn=ttk.Button(row,text="开始监控",command=self.start); self.start_btn.pack(side="left",expand=True,fill="x",padx=4)
        self.stop_btn=ttk.Button(row,text="停止",command=self.stop,state="disabled"); self.stop_btn.pack(side="left",expand=True,fill="x",padx=4)
        ttk.Label(self.root,textvariable=self.status,font=("Microsoft YaHei UI",10,"bold")).pack(pady=8)
        ttk.Label(self.root,textvariable=self.backend_status,foreground="#245").pack()
        ttk.Label(self.root,text="OpenCL 自动调用 Intel / AMD / NVIDIA GPU；不支持时自动使用 CPU。",foreground="#666").pack(pady=3)
        ttk.Label(self.root,text="紧急停止：鼠标移到屏幕左上角或点击停止。",foreground="#955").pack(pady=10)

    def slider(self,p,label,var,lo,hi,step=1):
        row=ttk.Frame(p); row.pack(fill="x",padx=8,pady=3); ttk.Label(row,text=label,width=13).pack(side="left")
        tk.Scale(row,variable=var,from_=lo,to=hi,resolution=step,orient="horizontal",length=250,highlightthickness=0).pack(side="left",fill="x",expand=True)

    def start(self):
        if self.running:return
        self.running=True; self.stop_event.clear(); self.tracker.reset(); self.state="monitor"; self.success_hits=self.dialog_hits=0
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.status.set("状态：监控中，等待目标出现"); threading.Thread(target=self.loop,daemon=True).start()

    def stop(self):
        self.running=False; self.stop_event.set(); self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.status.set("状态：已停止")

    def say(self,s): self.root.after(0,lambda:self.status.set(s))

    def loop(self):
        try:
            with mss.mss() as cap:
                screen=cap.monitors[1]
                last_backend = None
                while not self.stop_event.is_set():
                    frame=np.asarray(cap.grab(screen))[:,:,:3][:,:,::-1].copy()
                    if self.state=="monitor":
                        point=self.tracker.update(frame,self.threshold.get(),self.area.get())
                        current_backend = self.tracker.backend_name
                        if current_backend != last_backend:
                            self.root.after(0,lambda value=current_backend:self.backend_status.set(f"图像掩膜：{value}"))
                            last_backend = current_backend
                        if point is not None:
                            x,y=point
                            if self.follow.get():
                                px,py=pyautogui.position(); a=float(self.smooth.get())
                                pyautogui.moveTo(int(px+(x-px)*a),int(py+(y-py)*a),_pause=False)
                            self.say(f"状态：{'锁定' if self.tracker.last_detected else '透明期预测'} {x},{y}")
                        else:self.say("状态：等待目标 / 下一轮")
                        if self.auto.get() and success_banner(frame): self.success_hits+=1
                        else:self.success_hits=max(0,self.success_hits-1)
                        if self.auto.get() and self.success_hits>=2:
                            self.state="dialog"; self.dialog_hits=0; self.say("状态：检测成功，等待完成确认框")
                    elif self.state=="dialog":
                        button=confirm_button(frame)
                        self.dialog_hits=self.dialog_hits+1 if button else 0
                        if self.dialog_hits>=3:
                            pyautogui.click(button[0],button[1]); self.state="after_confirm"
                            self.say("状态：已点击确认，等待下一轮")
                    elif self.state=="after_confirm":
                        if confirm_button(frame) is None:
                            self.tracker.reset(); self.state="monitor"; self.success_hits=self.dialog_hits=0
                            self.say("状态：已恢复监控，等待下一轮小游戏")
                    time.sleep(.025)
        except pyautogui.FailSafeException:self.root.after(0,self.stop)
        except Exception as e:self.say(f"运行错误：{e}"); self.root.after(0,self.stop)

    def close(self):
        self.stop(); self.root.after(100,self.root.destroy)

if __name__=="__main__":
    root=tk.Tk(); App(root); root.mainloop()
