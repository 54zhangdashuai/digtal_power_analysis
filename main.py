# main.py
import sys
from PyQt5.QtWidgets import QApplication
from main_window import DataAnalysisApp

def main():
    """主函数，应用程序的入口点"""
    app = QApplication(sys.argv)
    win = DataAnalysisApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()