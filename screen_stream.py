"""Android scrcpy window capture for Flask Live Detection."""

import ctypes
import threading
import time
from ctypes import wintypes

import cv2
import numpy as np
import pygetwindow as gw


user32 = ctypes.windll.user32


class ScreenStream:

    def __init__(self):

        self.window = None
        self.hwnd = None

        self.lock = threading.Lock()

        self.last_search = 0

        self.last_error = "Waiting for scrcpy"


    def find_window(self, force=False):

        if (
            not force
            and time.time() - self.last_search < 1
        ):
            return self.hwnd is not None


        self.last_search = time.time()

        self.window = None
        self.hwnd = None


        windows = gw.getAllWindows()


        for w in windows:

            try:

                title = w.title.strip().lower()


                # scrcpy 4.x uses phone model name
                if title == PHONE_WINDOW_TITLE:

                    if (
                        w.width > 100
                        and w.height > 100
                    ):

                        self.window = w
                        self.hwnd = w._hWnd


                        print(
                            "PHONE WINDOW FOUND:",
                            w.title
                        )

                        print(
                            "POSITION:",
                            w.left,
                            w.top,
                            w.width,
                            w.height
                        )


                        return True


            except Exception:
                pass



        self.last_error = "Phone mirror window not found"

        return False



    def capture_window(self):

        hwnd = self.hwnd


        rect = wintypes.RECT()


        user32.GetClientRect(
            hwnd,
            ctypes.byref(rect)
        )


        width = rect.right - rect.left
        height = rect.bottom - rect.top


        if width <= 0 or height <= 0:
            return None



        hwnd_dc = user32.GetWindowDC(hwnd)


        srcdc = ctypes.windll.gdi32.CreateCompatibleDC(
            hwnd_dc
        )


        bmp = ctypes.windll.gdi32.CreateCompatibleBitmap(
            hwnd_dc,
            width,
            height
        )


        ctypes.windll.gdi32.SelectObject(
            srcdc,
            bmp
        )


        ctypes.windll.user32.PrintWindow(
            hwnd,
            srcdc,
            2
        )


        buffer = ctypes.create_string_buffer(
            width * height * 4
        )


        ctypes.windll.gdi32.GetBitmapBits(
            bmp,
            len(buffer),
            buffer
        )


        img = np.frombuffer(
            buffer,
            dtype=np.uint8
        )


        img = img.reshape(
            height,
            width,
            4
        )


        ctypes.windll.gdi32.DeleteObject(
            bmp
        )


        ctypes.windll.gdi32.DeleteDC(
            srcdc
        )


        user32.ReleaseDC(
            hwnd,
            hwnd_dc
        )


        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2BGR
        )


        return img



    def remove_title_bar(self, frame):

        if frame is None:
            return None


        h, w = frame.shape[:2]


        # remove scrcpy top bar only
        if h > 100 and w > 100:

            frame = frame[
                35:h,
                0:w
            ]


        return frame



    def read(self):

        if self.hwnd is None:

            if not self.find_window(True):

                return False, None



        try:

            with self.lock:

                frame = self.capture_window()



            if frame is None:

                return False, None



            frame = self.remove_title_bar(
                frame
            )



            return True, frame



        except Exception as e:


            print(
                "CAPTURE ERROR:",
                e
            )


            self.hwnd = None
            self.window = None


            return False, None




stream = ScreenStream()