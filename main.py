import sys
from frontend import SliderWindow  # Import the frontend
from PyQt5.QtWidgets import QApplication

def main():
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    window = SliderWindow()
    window.show()

    # Run the application event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()