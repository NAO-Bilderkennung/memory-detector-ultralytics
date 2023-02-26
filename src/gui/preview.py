import wx
import cv2 as cv


class ImagePreview(wx.Panel):
    def __init__(self, parent, label, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.image = None

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Bind(wx.EVT_SIZE,  self.on_resize)

        self.update_timer = wx.Timer(self)
        self.update_timer.Start(100)

        self.label = label
        self.should_label = True

    def update_image(self, image, clear_background=False):
        self.image = image.copy()
        self.Refresh(clear_background)

    def on_paint(self, _event):
        dc = wx.PaintDC(self)
        self._draw_image(dc)

    def on_timer(self, _event):
        self.Refresh(False)

    def on_resize(self, _event):
        self.Refresh()

    def _get_best_image_size_and_offset(self, dc):
        image_size = self.image.shape[:2]
        dc_size = dc.GetSize()

        if dc_size[0] > dc_size[1]:
            size = (int(image_size[1] / image_size[0] * dc_size[1]), dc_size[1])
            offset = (int((dc_size[0] - size[0]) / 2), 0)

            return size, offset

        size = (dc_size[0], int(image_size[0] / image_size[1] * dc_size[0]))
        offset = (0, int((dc_size[1] - size[1]) / 2))

        return size, offset

    def _draw_image(self, dc):
        if self.image is None:
            return

        best_size, offset = self._get_best_image_size_and_offset(dc)

        resized_image = cv.resize(self.image, best_size, interpolation=cv.INTER_CUBIC)
        resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

        if self.should_label:
            (text_width, text_height), baseline = cv.getTextSize(self.label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(resized_image, (0, 0), (text_width + 10, text_height + baseline + 10), (255, 255, 255), -1)
            cv.putText(resized_image, self.label, (5, text_height + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        bitmap = wx.Bitmap.FromBuffer(best_size[0], best_size[1], resized_image)

        dc.DrawBitmap(bitmap, offset[0], offset[1])
