import os.path
import sys
from time import perf_counter

import wx
import cv2 as cv

from src.detection import ObjectDetector
from src.gui.perfdisplay import PerfDisplay
from src.gui.preview import ImagePreview


class MDFrame(wx.Frame):
    def __init__(self, model_types, default_model, cuda_devices, cuda_device_names, default_cuda_device):
        super().__init__(parent=None, title="Memory Detector")

        self.model_types = model_types
        self.current_model = model_types[default_model]
        self.cuda_devices = cuda_devices
        self.cuda_device = cuda_devices[default_cuda_device]
        self.detector = ObjectDetector(model_type=self.current_model, cuda_device=self.cuda_device)

        panel = wx.Panel(self)

        self.input_panel = ImagePreview(panel, size=(640, 480), label="Input")
        self.detection_panel = ImagePreview(panel, size=(640, 480), label="Detected objects")
        config_panel = wx.Panel(panel)

        model_panel = wx.Panel(config_panel)

        model_label = wx.StaticText(model_panel, label="YOLO model")

        self.model_picker = wx.Choice(model_panel, choices=model_types)
        self.model_picker.Select(default_model)
        self.model_picker.Bind(wx.EVT_CHOICE, self._on_model_changed)

        device_label = wx.StaticText(model_panel, label="Device")

        self.device_picker = wx.Choice(model_panel, choices=cuda_device_names)
        self.device_picker.Select(default_cuda_device)
        self.device_picker.Bind(wx.EVT_CHOICE, self._on_device_changed)

        model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_sizer.Add(model_label, flag=wx.ALIGN_CENTER_VERTICAL)
        model_sizer.AddSpacer(5)
        model_sizer.Add(self.model_picker)
        model_sizer.AddSpacer(10)
        model_sizer.Add(device_label, flag=wx.ALIGN_CENTER_VERTICAL)
        model_sizer.AddSpacer(5)
        model_sizer.Add(self.device_picker)
        model_panel.SetSizer(model_sizer)

        self.detection_time = PerfDisplay("Detection", config_panel)
        self.box_draw_time = PerfDisplay("Box drawing", config_panel)
        self.paint_time = PerfDisplay("Paint", config_panel)

        self.filter_conf_checkbox = wx.CheckBox(config_panel, wx.ID_ANY, "Suppress confidences under 50%")
        self.filter_conf_checkbox.SetValue(True)  # Filters by default
        self.filter_conf_checkbox.Bind(wx.EVT_CHECKBOX, self._on_filter_conf_changed)

        self.label_images = wx.CheckBox(config_panel, wx.ID_ANY, "Label preview images")
        self.label_images.SetValue(True)  # Label by default
        self.label_images.Bind(wx.EVT_CHECKBOX, self._on_label_images_changed)

        self.prev_next_panel = wx.Panel(config_panel)

        self.prev_button = wx.Button(self.prev_next_panel, wx.ID_ANY, "Previous")
        self.prev_button.Disable()
        self.next_button = wx.Button(self.prev_next_panel, wx.ID_ANY, "Next")
        self.next_button.Disable()

        self.current_image_label = wx.StaticText(self.prev_next_panel)

        prev_next_sizer = wx.BoxSizer(wx.HORIZONTAL)
        prev_next_sizer.Add(self.prev_button)
        prev_next_sizer.Add(self.next_button)
        prev_next_sizer.AddSpacer(5)
        prev_next_sizer.Add(self.current_image_label, flag=wx.ALIGN_CENTER_VERTICAL)
        self.prev_next_panel.SetSizer(prev_next_sizer)

        self.detection_list = wx.ListCtrl(config_panel, size=(-1, 450), style=wx.LC_REPORT)
        self.detection_list.AppendColumn("Class")
        self.detection_list.AppendColumn("Confidence")

        config_sizer = wx.BoxSizer(wx.VERTICAL)
        config_sizer.Add(model_panel)
        config_sizer.AddSpacer(5)
        config_sizer.Add(self.detection_time)
        config_sizer.Add(self.box_draw_time)
        config_sizer.Add(self.paint_time)
        config_sizer.AddSpacer(10)
        config_sizer.Add(self.filter_conf_checkbox)
        config_sizer.Add(self.label_images)
        config_sizer.AddSpacer(10)
        config_sizer.Add(self.prev_next_panel, flag=wx.EXPAND)
        config_sizer.Add(self.detection_list, flag=wx.EXPAND)
        config_panel.SetSizer(config_sizer)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.input_panel, 2, wx.EXPAND)
        sizer.AddSpacer(10)
        sizer.Add(self.detection_panel, 2, wx.EXPAND)
        sizer.Add(config_panel, 1, wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(sizer)

        self._create_menu()

        self.image = cv.imread("test.png")
        self.input_panel.update_image(self.image)

        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_BUTTON, self._on_prev_clicked, source=self.prev_button)
        self.Bind(wx.EVT_BUTTON, self._on_next_clicked, source=self.next_button)

        self.timer = wx.Timer(self)
        self.timer.Start(50, wx.TIMER_ONE_SHOT)

        self.folder = None
        self.image_index = -1

        self.SetClientSize(panel.GetBestSize())
        self.Show()

    def _create_menu(self):
        menu_bar = wx.MenuBar()

        file_menu = wx.Menu()

        file_open_image = file_menu.Append(wx.ID_ANY, "&Open image", "Open a local image")
        file_open_folder = file_menu.Append(wx.ID_ANY, "Open &folder", "Open a folder with indexed images")
        file_menu.AppendSeparator()
        file_exit = file_menu.Append(wx.ID_EXIT, "E&xit", "Close the application")

        self.Bind(wx.EVT_MENU, self._on_open_image, source=file_open_image)
        self.Bind(wx.EVT_MENU, self._on_open_folder, source=file_open_folder)
        self.Bind(wx.EVT_MENU, self.on_close, source=file_exit)

        menu_bar.Append(file_menu, "&File")

        self.SetMenuBar(menu_bar)

    def _on_model_changed(self, _event):
        self.current_model = self.model_types[self.model_picker.GetCurrentSelection()]
        self._reload_object_detector()

    def _on_device_changed(self, _event):
        self.cuda_device = self.cuda_devices[self.device_picker.GetCurrentSelection()]
        self._reload_object_detector()

    def _reload_object_detector(self):
        self.timer.Stop()

        self.detector = ObjectDetector(model_type=self.current_model, cuda_device=self.cuda_device)

        self._reset_timers()
        self.timer.Start()

    def _on_filter_conf_changed(self, _event):
        self.detector.should_filter = self.filter_conf_checkbox.IsChecked()

    def _on_label_images_changed(self, _event):
        self.input_panel.should_label = self.label_images.IsChecked()
        self.detection_panel.should_label = self.label_images.IsChecked()

    def _on_open_image(self, _event):
        self.timer.Stop()

        dialog = wx.FileDialog(self, "Open an image", wildcard="Image files (*.jpg;*.png)|*.jpg;*.png",
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            self.timer.Start()
            return

        try:
            self.folder = None
            self.index = -1
            self.current_image_label.SetLabel("")

            self.image = cv.imread(dialog.GetPath())
            self.input_panel.update_image(self.image, clear_background=True)
            self.detection_panel.update_image(self.image, clear_background=True)

            self.prev_button.Disable()
            self.next_button.Disable()
            self._reset_timers()
        except Exception as e:
            print(e, file=sys.stderr)
            wx.MessageBox(f"Couldn't load image: {e}", "Error", style=wx.ICON_ERROR)
        finally:
            self.timer.Start()

    def _on_open_folder(self, _event):
        self.timer.Stop()

        dialog = wx.DirDialog(self, "Open a folder", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            self.timer.Start()
            return

        try:
            self.folder = dialog.GetPath()
            self.image = cv.imread(os.path.join(self.folder, "0.png"))
            self.input_panel.update_image(self.image, clear_background=True)
            self.detection_panel.update_image(self.image, clear_background=True)

            self.index = 0
            self.current_image_label.SetLabel(f"Current image: {self.index}")
            self.prev_button.Enable()
            self.next_button.Enable()
            self._reset_timers()
        except Exception as e:
            print(e, file=sys.stderr)
            wx.MessageBox(f"Couldn't load image: {e}", "Error", style=wx.ICON_ERROR)
        finally:
            self.timer.Start()

    def _on_prev_clicked(self, _event):
        if self.index == 0:
            wx.Bell()
            return

        self._update_image_index(self.index - 1)

    def _on_next_clicked(self, _event):
        next_index = self.index + 1

        if not os.path.exists(os.path.join(self.folder, f"{next_index}.png")):
            wx.Bell()
            return

        self._update_image_index(next_index)

    def _update_image_index(self, new_index):
        self.timer.Stop()

        try:
            self.image = cv.imread(os.path.join(self.folder, f"{new_index}.png"))
            self.input_panel.update_image(self.image)

            self.index = new_index
            self.current_image_label.SetLabel(f"Current image: {self.index}")
        except Exception as e:
            print(e, file=sys.stderr)
            wx.MessageBox(f"Couldn't load image: {e}", "Error", style=wx.ICON_ERROR)
        finally:
            self.timer.Start()

    def _reset_timers(self):
        self.detection_time.reset()
        self.box_draw_time.reset()
        self.paint_time.reset()

    def on_timer(self, _event):
        self.detector.detect(self.image)

        starting_time = perf_counter()
        self.detection_panel.update_image(self.detector.result_image)
        paint_time = perf_counter() - starting_time

        self.detection_time.update_time(self.detector.prediction_time * 1000, "ms")
        self.box_draw_time.update_time(self.detector.box_draw_time * 1000, "ms")
        self.paint_time.update_time(paint_time * 1000, "ms")

        self.detection_list.DeleteAllItems()

        for result in self.detector.results:
            for box in result.boxes:
                class_name = self.detector.get_class_name_of_box(box)
                confidence_percent = f"{round(box.conf.item() * 100, 2)}%"

                self.detection_list.Append([class_name, confidence_percent])

        self.timer.Start(50, wx.TIMER_ONE_SHOT)

    def on_close(self, _event):
        self.timer.Stop()  # We need to stop the timer or our application won't quit
        wx.CallAfter(self.Destroy)
