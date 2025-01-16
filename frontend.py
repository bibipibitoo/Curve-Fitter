import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from backend import Backend  # Import the backend


class SolverThread(QThread):
    """Thread to solve the Shockley equation in the background."""
    finished = pyqtSignal(np.ndarray, np.ndarray, float)  # Signal to emit results

    def __init__(self, V_values, I_sc, I_o, n, T, R_s, R_sh, backend):
        super().__init__()
        self.V_values = V_values
        self.I_sc = I_sc
        self.I_o = I_o
        self.n = n
        self.T = T
        self.R_s = R_s
        self.R_sh = R_sh
        self.backend = backend

    def run(self):
        """Run the solver in a separate thread."""
        try:
            # Solve the Shockley equation
            I_values = self.backend.solve_shockley(self.V_values, self.I_sc, self.I_o, self.n, self.T, self.R_s, self.R_sh)

            # Interpolate theoretical I_values to match the uploaded voltage points
            theoretical_currents = np.interp(self.backend.uploaded_data[:, 0], self.V_values, I_values)

            # Calculate R² value
            r_squared = self.backend.calculate_r_squared(self.backend.uploaded_data[:, 1], theoretical_currents)
            r_squared = max(0, r_squared)  # Clip negative R² to zero

            # Emit results
            self.finished.emit(self.V_values, I_values, r_squared)
        except Exception as e:
            print(f"Error in solver thread: {e}")
            # Emit empty results if an error occurs
            self.finished.emit(np.array([]), np.array([]), 0.0)


class SliderWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize backend
        self.backend = Backend()

        # Initialize variables
        self.sliders = {}
        self.labels = {}
        self.input_boxes = {}
        self.parameters = {
            'I_sc': (0, 2000, 1000, 1000),  # (min, max, default, scaling factor)
            'I_o': (0, 100, 10, 1000),      # (min, max, default, scaling factor)
            'n': (1000, 5000, 3000, 1000),  # (min, max, default, scaling factor)
            'T': (200, 400, 300, 1),        # (min, max, default, scaling factor)
            'R_s': (0, 10000, 10, 1000),    # (min, max, default, scaling factor)
            'R_sh': (0, 10000, 100, 1000)    # (min, max, default, scaling factor)
        }
        self.debounce_timer = QTimer()  # For debouncing parameter updates
        self.solver_thread = None  # For background computation
        self.theoretical_I_values = None  # Store theoretical current values
        self.plot_mode = "normal"  # Track the current plot mode ("normal" or "log")

        # Initialize UI
        self.initUI()

    def initUI(self):
        # Main layout: horizontal split
        main_layout = QHBoxLayout()

        # Left side: parameters
        left_layout = QVBoxLayout()

        # Create parameter controls
        for param, (min_val, max_val, default_val, scaling) in self.parameters.items():
            left_layout.addLayout(self.create_parameter_controls(param, min_val, max_val, default_val, scaling))

        # Add upload button
        self.upload_button = QPushButton('Upload Data File')
        self.upload_button.clicked.connect(self.uploadFile)
        left_layout.addWidget(self.upload_button)

        # Add plot button
        self.plot_button = QPushButton('Plot Theoretical Curve')
        self.plot_button.clicked.connect(lambda: self.plotData(log_mode=False))
        left_layout.addWidget(self.plot_button)

        # Add log transformation button
        self.log_button = QPushButton('Plot Log(Current)')
        self.log_button.clicked.connect(lambda: self.plotData(log_mode=True))
        left_layout.addWidget(self.log_button)

        # Add R^2 label
        self.r_squared_label = QLabel('R²: N/A')
        left_layout.addWidget(self.r_squared_label)

        # Add stretch to push everything to the top
        left_layout.addStretch()

        # Right side: plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)

        # Add left and right layouts to the main layout
        main_layout.addLayout(left_layout, 1)  # Left side takes 1 part of the space
        main_layout.addLayout(right_layout, 2)  # Right side (plot) takes 2 parts of the space

        self.setLayout(main_layout)
        self.setWindowTitle('Parameter Adjuster')
        self.setGeometry(100, 100, 1200, 600)  # Set initial window size

        # Configure debounce timer
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.handlePlotUpdate)

        self.show()

    def create_parameter_controls(self, param, min_val, max_val, default_val, scaling):
        """Create sliders, labels, and input boxes for a parameter."""
        param_layout = QHBoxLayout()

        # Add label
        label = QLabel(f'{param}:')
        param_layout.addWidget(label)
        self.labels[param] = label

        # Add slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(lambda value, p=param: self.updateInputBox(p, value))
        slider.valueChanged.connect(self.startDebounceTimer)
        param_layout.addWidget(slider)
        self.sliders[param] = slider

        # Add input box
        input_box = QLineEdit()
        input_box.setText(str(default_val / scaling))
        input_box.setFixedWidth(100)
        input_box.returnPressed.connect(lambda p=param: self.updateSliderFromInput(p))
        input_box.returnPressed.connect(self.startDebounceTimer)
        param_layout.addWidget(input_box)
        self.input_boxes[param] = input_box

        return param_layout

    def updateInputBox(self, param, value):
        """Update the input box when the slider changes."""
        scaling = self.parameters[param][3]
        self.input_boxes[param].setText(f"{value / scaling:.4f}")

    def updateSliderFromInput(self, param):
        """Update the slider when the input box changes."""
        try:
            value = float(self.input_boxes[param].text())
            if value < 0:
                raise ValueError("Value must be non-negative.")
            scaling = self.parameters[param][3]
            self.sliders[param].setValue(int(value * scaling))
        except ValueError as e:
            print(f"Invalid input: {e}")
            # Show error message to the user
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setWindowTitle("Invalid Input")
            error_dialog.setText(f"Invalid value for {param}: {e}")
            error_dialog.exec_()

    def startDebounceTimer(self):
        """Start the debounce timer to delay plot updates."""
        self.debounce_timer.start(500)  # Wait 500ms before updating the plot

    def handlePlotUpdate(self):
        """Handle plot updates based on the current plot mode."""
        self.plotData(log_mode=(self.plot_mode == "log"))

    def uploadFile(self):
        """Allow the user to upload a file and plot the data."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if file_name:
            if self.backend.load_data(file_name):
                self.plotUploadedData()

    def plotUploadedData(self):
        """Plot the uploaded data."""
        if self.backend.uploaded_data is not None:
            # Clear the previous plot
            self.figure.clear()

            # Create a new plot
            ax = self.figure.add_subplot(111)
            ax.plot(self.backend.uploaded_data[:, 0], self.backend.uploaded_data[:, 1], 'bo', label='Uploaded Data')  # Current in mA
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel('Current (mA)')
            ax.set_title(f'I-V Characteristics: {self.backend.uploaded_file_name}')
            ax.legend()
            ax.grid(True)

            # Set plot limits to center on the uploaded data
            voltage_range = max(self.backend.uploaded_data[:, 0]) - min(self.backend.uploaded_data[:, 0])
            current_range = max(self.backend.uploaded_data[:, 1]) - min(self.backend.uploaded_data[:, 1])
            ax.set_xlim(min(self.backend.uploaded_data[:, 0]) - 0.1 * voltage_range, max(self.backend.uploaded_data[:, 0]) + 0.1 * voltage_range)
            ax.set_ylim(min(self.backend.uploaded_data[:, 1]) - 0.1 * current_range, max(self.backend.uploaded_data[:, 1]) + 0.1 * current_range)

            # Refresh the canvas
            self.canvas.draw()

    def plotData(self, log_mode=False):
        """Plot the theoretical I-V curve or its log transformation."""
        if self.backend.uploaded_data is None:
            return  # Do nothing if no data is uploaded

        # Set plot mode
        self.plot_mode = "log" if log_mode else "normal"

        # Get parameter values
        params = self.get_parameter_values()
        I_sc, I_o, n, T, R_s, R_sh = params['I_sc'], params['I_o'], params['n'], params['T'], params['R_s'], params['R_sh']

        # Define V_values based on the uploaded data range
        V_values = np.linspace(min(self.backend.uploaded_data[:, 0]), max(self.backend.uploaded_data[:, 0]), 100)

        # Create and start the solver thread
        self.solver_thread = SolverThread(V_values, I_sc, I_o, n, T, R_s, R_sh, self.backend)
        self.solver_thread.finished.connect(self.updatePlot)  # Connect to the finished signal
        self.solver_thread.start()

    def get_parameter_values(self):
        """Get and scale all parameter values."""
        return {
            'I_sc': self.sliders['I_sc'].value() / self.parameters['I_sc'][3],
            'I_o': self.sliders['I_o'].value() / self.parameters['I_o'][3],
            'n': self.sliders['n'].value() / self.parameters['n'][3],
            'T': self.sliders['T'].value() / self.parameters['T'][3],
            'R_s': self.sliders['R_s'].value() / self.parameters['R_s'][3],
            'R_sh': self.sliders['R_sh'].value() / self.parameters['R_sh'][3]
        }

    def updatePlot(self, V_values, I_values, r_squared):
        """Update the plot with the results from the solver thread."""
        if V_values.size == 0 or I_values.size == 0:
            # Handle solver failure
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("Failed to solve the Shockley equation.")
            error_dialog.exec_()
            return

        # Store theoretical current values
        self.theoretical_I_values = I_values

        # Clear the previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.plot_mode == "normal":
            # Plot normal data
            ax.plot(self.backend.uploaded_data[:, 0], self.backend.uploaded_data[:, 1], 'bo', ms=1, label='Uploaded Data')
            ax.plot(V_values, I_values, 'r-', label='Theoretical Curve')
            ax.set_ylabel('Current (mA)')
        else:
            # Plot log data
            log_data = self.backend.compute_log_data()
            log_theoretical = self.backend.compute_log_theoretical(V_values, I_values)
            ax.plot(log_data[:, 0], log_data[:, 1], 'bo', label='Log(Current)')
            ax.plot(log_theoretical[:, 0], log_theoretical[:, 1], 'r-', label='Log(Theoretical Current)')
            ax.set_ylabel('Log(Current)')

        ax.set_xlabel('Voltage (V)')
        ax.set_title(f'{self.plot_mode.capitalize()} I-V Characteristics: {self.backend.uploaded_file_name}')
        ax.legend()
        ax.grid(True)

        # Update R² label
        self.r_squared_label.setText(f'R²: {r_squared:.4f}')

        # Refresh the canvas
        self.canvas.draw()