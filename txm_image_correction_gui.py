import sys, os, yaml, time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QProgressBar,
    QFileDialog, QLineEdit, QMessageBox, QHBoxLayout, QSlider, QScrollBar
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDIM_Sampler
from data_io import read_txm_raw
from utils import min_max_percentile, save_mosaic


class DiffusionGUI(QWidget):
    def __init__(self):
        super().__init__()
        font = QFont("Times New Roman", 14)  # 調整整體字體大小
        self.setFont(font)
        self.setWindowTitle("TXM Background Correction on DDPM")
        self.setGeometry(200, 200, 800, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # GPU 狀態顯示
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("GPU 狀態:")
        self.gpu_status_label = QLabel("檢測中…")
        self.gpu_status_label.setFixedHeight(20)
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_status_label)
        gpu_layout.addStretch(1)
        self.layout.addLayout(gpu_layout)
        self.check_cuda_availability("cuda:0")

        # Path inputs
        self.config_input = self.create_path_input("模型設定檔:", "configs/BGC_v1_inference.yml", filter="*.yml")
        self.ckpt_input = self.create_path_input("模型權重:", "checkpoints/ddpm_pair_ft_10K.pt", filter="*.pt")
        self.input_input = self.create_path_input("TXM原始檔:", filter="*.xrm *.txrm")

        # Run button
        self.run_button = QPushButton("Run Inference")
        self.run_button.clicked.connect(self.run_inference)
        self.layout.addWidget(self.run_button)

        # Progress bar + ETA
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.eta_label = QLabel("剩餘時間: -")
        self.eta_label.setAlignment(Qt.AlignCenter)
        self.eta_label.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.eta_label)
        self.layout.addLayout(progress_layout)

        # Image preview
        self.image_label = QLabel("Preview.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Slider for browsing stack
        scroll_layout = QVBoxLayout()
        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setFixedHeight(20)
        scroll_layout.addWidget(self.index_label)

        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.scrollbar.setEnabled(False)
        self.scrollbar.valueChanged.connect(self.scrollbar_changed)
        scroll_layout.addWidget(self.scrollbar)
        self.layout.addLayout(scroll_layout)
        
        # Variables
        self.image_path = ""
        self.images = None
        self.metadata = None
        self.current_index = 0
        self.viewer = None
        self.thread = None

    def create_path_input(self, label_text, default="", filter=None, is_dir=False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit(default)
        browse_btn = QPushButton("Browse")
        if is_dir:
            browse_btn.clicked.connect(lambda: self.browse_folder(line_edit, filter))
        else:
            browse_btn.clicked.connect(lambda: self.browse_file(line_edit, filter))
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        self.layout.addLayout(layout)
        return line_edit

    def browse_file(self, line_edit, filter):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter)
        if path:
            self.image_path = path
            line_edit.setText(path)
            if line_edit is self.input_input:
                self.load_raw_file(path)

    def load_raw_file(self, filepath):
        mode = 'tomo' if filepath.endswith('.txrm') else 'mosaic'
        images, metadata, _ = read_txm_raw(filepath, mode=mode)
        self.images = images
        self.metadata = metadata
        self.scrollbar.setEnabled(True)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(len(images) - 1)
        self.scrollbar.setValue(0)
        self.show_image(0)

    def show_image(self, index):
        if self.images is None:
            return
        self.current_index = index
        img = self.images[index]
        img = ((img / img.max()) * 255).astype(np.uint8)

        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(512, 512, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.index_label.setText(f"{index+1} / {len(self.images)}")

    def scrollbar_changed(self, value):
        self.show_image(value)

    def check_cuda_availability(self, device_str):
        available = torch.cuda.is_available()
        if available:
            self.gpu_status_label.setText(f"{device_str} ✅ 可用")
            self.gpu_status_label.setStyleSheet("color: green;")
        else:
            self.gpu_status_label.setText(f"{device_str} ❌ 無法使用")
            self.gpu_status_label.setStyleSheet("color: red;")

    def run_inference(self):
        if self.images is not None:
            configs = self.config_input.text()
            model_ckpt = self.ckpt_input.text()
            raw_txm_file = self.input_input.text()
            sample_name = os.path.splitext(os.path.basename(raw_txm_file))[0]
            parent_dir = os.path.dirname(raw_txm_file)
            save_dir = f"{parent_dir}/{sample_name}(AI Dref)"
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

            self.progress_bar.setValue(0)
            self.eta_label.setText("剩餘時間: -")

            self.viewer = InferenceViewer()
            self.viewer.show()
            self.thread = InferenceThread(configs, model_ckpt, self.images, save_dir)
            self.thread.progress_updated.connect(self.update_progress)
            self.thread.image_ready.connect(self.viewer.add_image)
            self.thread.finished.connect(self.inference_done)
            self.thread.failed.connect(self.inference_failed)
            self.thread.start()

    def update_progress(self, current, total, elapsed):
        self.progress_bar.setFormat("處理第 %v 張 / 共 %m 張")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current > 0 and elapsed:
            avg_time = elapsed / current
            eta = avg_time * (total - current)
            self.eta_label.setText(f"剩餘時間: {self.format_eta(eta)}")

    def inference_done(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setFormat("✅ 推論完成")
        self.eta_label.setText("完成 ✅")

        if self.metadata['mosaic_row'] > 1 and self.metadata['mosaic_column'] > 1:
            save_mosaic(self.save_dir, n_cols=self.metadata['mosaic_column'], auto_contrast=True)

    def inference_failed(self, error_msg):
        QMessageBox.critical(self, "Error", f"Inference failed: {error_msg}")

    def format_eta(self, seconds):
        if seconds < 60:
            return f"{int(seconds)} 秒"
        elif seconds < 3600:
            return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
        else:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hrs} 小時 {mins} 分"


class InferenceViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inference Preview")
        self.setGeometry(300, 300, 600, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel("等待結果中...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.slider = QScrollBar(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_changed)
        self.layout.addWidget(self.slider)

        self.images = []
        self.current_index = 0

    def add_image(self, image):
        self.images.append(image)
        self.slider.setEnabled(True)
        self.slider.setMaximum(len(self.images) - 1)
        self.slider.setValue(len(self.images) - 1)  # 自動跳到最新一張
        self.show_image(len(self.images) - 1)

    def show_image(self, index):
        if not self.images:
            return
        self.current_index = index
        img = self.images[index]
        img = min_max_percentile(img)
        qimg = QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(512, 512, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.setWindowTitle(f"Inference Preview {index+1}/{len(self.images)}")

    def slider_changed(self, value):
        self.show_image(value)


class InferenceThread(QThread):
    progress_updated = pyqtSignal(int, int, float)
    image_ready = pyqtSignal(object) 
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, configs, model_ckpt, raw_images, img_save_dir, device="cuda:0", seed=0):
        super().__init__()
        self.configs = configs
        self.model_ckpt = model_ckpt
        self.raw_images = raw_images
        self.img_save_dir = img_save_dir
        self.device = device
        self.seed = seed

    def run(self):
        try:
            with open(self.configs, 'r') as f:
                configs = yaml.safe_load(f)

            model_configs = configs['model_settings']
            model = Diffusion_UNet(model_configs).to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(self.model_ckpt, map_location=self.device), strict=True)
            model.eval()
            sampler = DDIM_Sampler(model, configs['ddpm_settings'], ddim_sampling_steps=50).to(self.device)

            glob_max = np.max(self.raw_images)

            start_time = time.time()
            with torch.no_grad():
                i = 0
                n_steps = len(self.raw_images) - 1
                self.progress_updated.emit(0, n_steps, 0.0)

                for i in range(n_steps):
                    max_g = np.max(self.raw_images[i:i + 2])
                    input_1 = torch.tensor(self.raw_images[i] / max_g).unsqueeze(0).float().to(self.device)
                    input_2 = torch.tensor(self.raw_images[i + 1] / max_g).unsqueeze(0).float().to(self.device)

                    if input_1.shape[1] != 256:
                        input_1 = F.interpolate(input_1.unsqueeze(0), size=(256, 256), mode='bicubic').squeeze(0)
                        input_2 = F.interpolate(input_2.unsqueeze(0), size=(256, 256), mode='bicubic').squeeze(0)

                    input_imgs = torch.cat([input_1, input_2], dim=0)

                    torch.manual_seed(self.seed)
                    noise = torch.randn(size=[1, 1, 256, 256], device=self.device)
                    pred = sampler(input_imgs.unsqueeze(0), noise).squeeze().cpu().numpy()
                    pred = pred / pred.max()

                    obj_pred_1 = input_1.squeeze().cpu().numpy() / pred * max_g
                    obj_pred_2 = input_2.squeeze().cpu().numpy() / pred * max_g

                    im = Image.fromarray(np.clip(obj_pred_1 / glob_max * 255, 0, 255).astype(np.uint8))
                    im_path = os.path.join(self.img_save_dir, f'{i+1:03d}.tif')
                    im.save(im_path)
                    self.image_ready.emit(np.array(im))

                    if i == (len(self.raw_images) - 2):
                        im = Image.fromarray(np.clip(obj_pred_2 / glob_max * 255, 0, 255).astype(np.uint8))
                        im.save(os.path.join(self.img_save_dir, f'{i+2:03d}.tif'))
                        self.image_ready.emit(np.array(im))

                    elapsed = time.time() - start_time
                    self.progress_updated.emit(i+1, n_steps, elapsed)

            self.finished.emit()

        except Exception as e:
            self.failed.emit(str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = DiffusionGUI()
    gui.show()
    sys.exit(app.exec_())