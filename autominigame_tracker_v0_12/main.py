import sys


def _smoke_test():
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from tracking.csrt_tracker import create_csrt
    from vision.round_events import RoundEventDetector
    from ui.main_window import MainWindow

    detector = RoundEventDetector()
    if not detector.available:
        raise RuntimeError("round-result templates were not packaged")

    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 50), (110, 100), (255, 255, 255), -1)
    tracker = create_csrt()
    tracker.init(frame, (60, 50, 51, 51))
    ok, bbox = tracker.update(frame.copy())
    if not ok or bbox is None:
        raise RuntimeError("OpenCV CSRT smoke test failed")

    app = QApplication([])
    window = MainWindow()
    window.close()
    app.processEvents()
    app.quit()


from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
        raise SystemExit(0)
    app = QApplication(sys.argv)
    app.setApplicationName("自动化小游戏程序")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
