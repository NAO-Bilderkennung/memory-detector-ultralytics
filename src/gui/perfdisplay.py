import wx


class PerfDisplay(wx.StaticText):
    def __init__(self, label, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = label
        self.time = None

        self._times = []
        self._max_count = 100

        self._update_label_text("Not yet measured")

    def _update_label_text(self, time):
        self.SetLabelText(f"{self.label} time: {time}")

    def update_time(self, time, unit="ms"):
        self.time = time

        count = len(self._times)

        if count + 1 == self._max_count:
            self._times = self._times[1:]

        self._times.append(time)
        count += 1

        average = sum(self._times) / count
        self._update_label_text(f"{round(self.time)}{unit} ({count} avg.: {round(average, 1)}{unit})")

    def reset(self):
        self.time = None
        self._times = []
        self._update_label_text("Not yet measured")
