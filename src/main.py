from PySide6.QtWidgets import QApplication

from GUI import NNBuilderApp

def main():
    app = QApplication([])
    window = NNBuilderApp()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()