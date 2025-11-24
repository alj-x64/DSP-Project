import sys
from PyQt5.QtWidgets import QApplication
from ui import ImageUI

def main():
    app = QApplication(sys.argv)
    ui = ImageUI()
    ui.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()