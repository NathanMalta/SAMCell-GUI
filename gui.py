import sys
import os
import time
from enum import Enum

from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QMainWindow, QListWidget, QListWidgetItem
from PyQt6.QtWidgets import QSplitter, QFrame, QGridLayout, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QTableWidget
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QByteArray, QBuffer

import cv2
from model import FinetunedSAM
from pipeline import SlidingWindowPipeline
import torch
import numpy as np
import time

import torch
torch.set_num_threads(os.cpu_count())

class ImageProcessingState(Enum):
    '''Enum to track image processing state
    '''
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETE = 2

class CustomListItem(QListWidgetItem):
    '''Custom list item to store image, text, etc
    '''
    def __init__(self, file_path):
        #get name from filepath
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        super().__init__(self.name)
        #load as grayscale
        self.image = QtGui.QImage(file_path).convertToFormat(QtGui.QImage.Format.Format_Grayscale8)

        self.output_gray = None
        self.dist_map = None
        self.overlay = None

        self.processing_state = ImageProcessingState.NOT_STARTED

    def updateState(self, state):
        self.processing_state = state
        if self.processing_state == ImageProcessingState.NOT_STARTED:
            self.setBackground(QtGui.QColor(255, 255, 255))
        elif self.processing_state == ImageProcessingState.IN_PROGRESS:
            self.setBackground(QtGui.QColor(255, 255, 0))
        elif self.processing_state == ImageProcessingState.COMPLETE:
            self.setBackground(QtGui.QColor(0, 255, 0))

    def setCompleteInfo(self, output_gray, dist_map, overlay):
        self.output_gray = output_gray
        self.dist_map = dist_map
        self.overlay = overlay

    def computeMetrics(self):
        if self.output_gray is None or self.dist_map is None:
            return
        
        #compute cell count
        cell_count = len(np.unique(self.output_gray)) - 1
        
        #compute cell area
        total_cell_area = np.sum(self.output_gray != 0)
        cell_area = total_cell_area / cell_count

        #compute confluency
        confluency = total_cell_area / (self.dist_map.shape[0] * self.dist_map.shape[1])

        #compute number of neighbors per cell
        neighbors = []
        for cell in np.unique(self.output_gray):
            if cell == 0:
                continue
            cell_coords = self.output_gray == cell
            #add 5 pixel buffer around cell
            cell_coords = np.where(cell_coords)
            cell_coords = (np.clip(cell_coords[0] - 5, 0, self.output_gray.shape[0] - 1), np.clip(cell_coords[1] - 5, 0, self.output_gray.shape[1] - 1))

            #get all cells within 10 pixels
            neighbor_cells = np.unique(self.output_gray[cell_coords[0], cell_coords[1]])
            neighbors.append(len(neighbor_cells) - 1)

        #compute average number of neighbors
        avg_neighbors = np.mean(neighbors)

        #round metrics to 2 decimal places
        cell_count = round(cell_count, 2)
        cell_area = round(cell_area, 2)
        confluency = round(confluency, 2)
        avg_neighbors = round(avg_neighbors, 2)

        #convert confluency to percentage
        confluency *= 100
        confluency = f'{int(confluency)}%'

        return cell_count, cell_area, confluency, avg_neighbors


    def getNumpyImage(self):
        '''  Converts a QImage into an opencv MAT format  '''

        # Create a NumPy array view of the raw data
        img = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        return img

    def text(self):
        return self.name

    def getPixelMap(self):
        return QtGui.QPixmap.fromImage(self.image)
    
    def getOutputPixelMap(self):
        if self.overlay is None:
            return None
        else:
            return QtGui.QPixmap.fromImage(QtGui.QImage(self.overlay.data, self.overlay.shape[1], self.overlay.shape[0], QtGui.QImage.Format.Format_RGB888))

    def __str__(self):
        return self.text()
    
class ImageLoader(QThread):
    '''Class to handle async image loading
    '''
    update_progress = pyqtSignal(int)
    widgets_complete = pyqtSignal(list)
    valid_img_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tif', 'tiff']

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        return_widgets = []
        for i, file_path in enumerate(self.image_paths):
            #check if file is an image
            if file_path.split('.')[-1].lower() not in self.valid_img_formats:
                continue
            return_widgets.append(CustomListItem(file_path))
            self.update_progress.emit((i + 1) * 100 // len(self.image_paths))
        self.widgets_complete.emit(return_widgets)

class ImageProcessor(QThread):
    '''Class to handle async image processing
    '''

    def __init__(self, widgetItems):
        super().__init__()
        self.widgetItems = widgetItems
        self.modelHelper = ModelHelper()
    
    def run(self):
        for item in self.widgetItems:
            item.updateState(ImageProcessingState.IN_PROGRESS)
            input_img = item.getNumpyImage()
            
            #rescale to 1000 (biggest dimension)
            long_side = 1000
            if input_img.shape[0] > input_img.shape[1]:
                new_size = (int(input_img.shape[1] * (long_side / input_img.shape[0])), long_side)
            else:
                new_size = (long_side, int(input_img.shape[0] * (long_side / input_img.shape[1])))
            resized = cv2.resize(input_img, new_size, interpolation=cv2.INTER_CUBIC)

            output_gray, dist_map, output_rgb = self.modelHelper.run(resized)

            overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            cell_coords = np.where(dist_map != 0)
            overlay[cell_coords[0], cell_coords[1], :] = output_rgb[cell_coords[0], cell_coords[1], :] * 0.5 + overlay[cell_coords[0], cell_coords[1], :] * 0.5
            
            item.setCompleteInfo(output_gray, dist_map, overlay)
            item.updateState(ImageProcessingState.COMPLETE)
            item.setSelected(False)

class ModelHelper():
    def __init__(self):
        #check if cuda is available
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        model = FinetunedSAM('facebook/sam-vit-base')
        trained_samcell_path = 'samcell-cyto/pytorch_model.bin'
        model.load_weights(trained_samcell_path, map_location=device)

        self.pipeline = SlidingWindowPipeline(model, device, crop_size=256)

    def convert_label_to_rainbow(self, label):
        label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for cell in np.unique(label):
            if cell == 0:
                continue #background
            label_rainbow[label == cell] = np.random.rand(3) * 255
        return label_rainbow

    def run(self, image):
        output, dist_map = self.pipeline.run(image, return_dist_map=True)
        output_rgb = self.convert_label_to_rainbow(output)

        return output, dist_map, output_rgb


class Menu(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        #initial window stuff
        self.setWindowTitle("SAMCell GUI")
        self.resize(600, 400)
        self.setMinimumSize(600, 400)
        self.setAcceptDrops(True)

        # Create a horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Add widgets to the splitter
        self.left_widget = QWidget()
        self.list_widget = QListWidget()
        self.list_widget.sortItems(Qt.SortOrder.AscendingOrder)
        self.right_widget = QWidget()
        self.image_label = QLabel('Drag and drop images or a folder here')
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setFrameShadow(QFrame.Shadow.Plain)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #make image label expand to fill right widget
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.overlay_label = QLabel('Overlay')
        self.overlay_label.setFrameShape(QFrame.Shape.Box)
        self.overlay_label.setFrameShadow(QFrame.Shadow.Plain)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #make image label expand to fill right widget
        self.overlay_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.overlay_label.setStyleSheet("border: 1px solid gray;")

        self.metrics_display = QTableWidget()
        self.metrics_display.setRowCount(4)
        self.metrics_display.setColumnCount(2)
        self.metrics_display.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        #make image label expand to fill right widget
        self.metrics_display.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.metrics_display.setStyleSheet("border: 1px solid gray;")
        self.metrics_display.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics_display.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.right_widget)

        #only resize right widget when splitter is resized
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        #setup left widget
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        left_header = QLabel('Images To Process')
        left_header.setFixedHeight(25)
        layout.addWidget(left_header)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.left_widget.setLayout(layout)
        self.loading_label = QLabel('Loading images...')
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.loading_label)
        self.loading_label.hide()

        #setup right widget
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        right_header = QWidget()
        right_header.setFixedHeight(left_header.height())
        right_header.setLayout(QHBoxLayout())
        right_header_label = QLabel('Image')
        right_header.layout().addWidget(right_header_label)
        right_header.layout().setContentsMargins(0, 0, 0, 0)
        right_header.layout().setSpacing(0)

        process_button = QtWidgets.QPushButton('Process This Image')
        process_button.clicked.connect(self.proccess_image)

        spacer = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        process_all_button = QtWidgets.QPushButton('Process All Images')
        process_all_button.clicked.connect(self.process_all_images)

        right_header.layout().addItem(spacer)
        right_header.layout().addWidget(process_button)
        right_header.layout().addItem(spacer)
        right_header.layout().addWidget(process_all_button)

        layout.addWidget(right_header)

        #add tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.South)

        self.tab_widget.addTab(self.image_label, 'Image')
        self.tab_widget.addTab(self.overlay_label, 'Segmentation Result')
        self.tab_widget.addTab(self.metrics_display, 'Metrics')

        #gray out tabs 2 and 3
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)

        layout.addWidget(self.tab_widget)
        self.right_widget.setLayout(layout)

        #add progress bar under right widget
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        #hide progress bar
        self.progress_bar.hide()

        #if list changes selection, update the right widget
        self.list_widget.itemSelectionChanged.connect(self.update_right_widget)

        #setup splitter parameters
        self.left_widget.setMinimumWidth(150)
        self.right_widget.setMinimumWidth(300)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        #add splitter to the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        layout.addWidget(splitter)

        self.show()

    def _update_progress(self, value):
        if value < 100:
            self.progress_bar.show()
            self.loading_label.show()
        else:
            self.progress_bar.hide()
            self.loading_label.hide()
        self.progress_bar.setValue(value)

    def _add_widgets(self, widgets):
        for widget in widgets:
            self.list_widget.addItem(widget)

        self.progress_bar.hide()
        self.loading_label.hide()

    def update_right_widget(self):
        if len(self.list_widget.selectedItems()) == 0:
            #set right widget to default text
            self.image_label.setText('Drag and drop images or a folder here')
            return
        
        #get the selected item's text
        selected_item = self.list_widget.selectedItems()[0]
        self.image_label.setPixmap(selected_item.getPixelMap().scaledToWidth(self.right_widget.width()))

        if selected_item.processing_state == ImageProcessingState.COMPLETE:
            self.overlay_label.setPixmap(selected_item.getOutputPixelMap().scaledToWidth(self.right_widget.width()))
            
            cell_count, cell_area, confluency, avg_neighbors = selected_item.computeMetrics()

            self.metrics_display.setItem(0, 0, QtWidgets.QTableWidgetItem('Cell Count'))
            self.metrics_display.setItem(0, 1, QtWidgets.QTableWidgetItem(str(cell_count)))

            self.metrics_display.setItem(1, 0, QtWidgets.QTableWidgetItem('Avg Cell Area (px)'))
            self.metrics_display.setItem(1, 1, QtWidgets.QTableWidgetItem(str(cell_area)))

            self.metrics_display.setItem(2, 0, QtWidgets.QTableWidgetItem('Confluency'))
            self.metrics_display.setItem(2, 1, QtWidgets.QTableWidgetItem(str(confluency)))

            self.metrics_display.setItem(3, 0, QtWidgets.QTableWidgetItem('Avg Neighbors'))
            self.metrics_display.setItem(3, 1, QtWidgets.QTableWidgetItem(str(avg_neighbors)))

            self.tab_widget.setTabEnabled(1, True)
            self.tab_widget.setTabEnabled(2, True)
        elif selected_item.processing_state == ImageProcessingState.IN_PROGRESS:
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setTabEnabled(2, False)
        else:
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setTabEnabled(2, False)

    def keyPressEvent(self, event):
        #delete selected items from list when delete or backspace key is pressed
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            for item in self.list_widget.selectedItems():
                self.list_widget.takeItem(self.list_widget.row(item))

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.DropAction.CopyAction)
            path = event.mimeData().urls()[0].toLocalFile()
            files = []
            #check if path is a directory
            if not os.path.isfile(path):
                #get all files in director y
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        files.append(file_path)
            else:
                #check if file is an image
                files.append(path)
                if len(event.mimeData().urls()) > 1:
                    for url in event.mimeData().urls()[1:]:
                        path = url.toLocalFile()
                        if os.path.isfile(path):
                            files.append(path)

            self.image_loader = ImageLoader(files)
            self._update_progress(0)
            self.image_loader.update_progress.connect(self._update_progress)
            self.image_loader.start()
            self.image_loader.widgets_complete.connect(self._add_widgets)

            event.accept()
        else:
            event.ignore()

    def proccess_image(self):
        if len(self.list_widget.selectedItems()) < 1:
            return
        selected_item = self.list_widget.selectedItems()[0]
        if selected_item.processing_state == ImageProcessingState.NOT_STARTED:
            self.image_processor = ImageProcessor([selected_item])
            self.image_processor.start()

    def process_all_images(self):
        items = self.list_widget.findItems("", Qt.MatchFlag.MatchContains)
        unprocessed = [item for item in items if item.processing_state == ImageProcessingState.NOT_STARTED]
        self.image_processor = ImageProcessor(unprocessed)
        self.image_processor.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = Menu()
    sys.exit(app.exec())
