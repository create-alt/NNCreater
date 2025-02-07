from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QDialog, QComboBox, QLineEdit, QMessageBox, QLabel, QHBoxLayout
import torch.nn as nn

class NNBuilderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Builder")
        self.resize(800, 600)  # 画面サイズをさらに拡大
        
        main_layout = QVBoxLayout()
        
        # モデル名入力欄
        self.model_name_entry = QLineEdit()
        self.model_name_entry.setPlaceholderText("Enter model name")
        main_layout.addWidget(QLabel("Model Name:"))
        main_layout.addWidget(self.model_name_entry)
        
        # レイヤーリスト
        self.layer_list = QListWidget()
        main_layout.addWidget(QLabel("Model Architecture:"))
        main_layout.addWidget(self.layer_list)
        
        # ボタンレイアウト
        button_layout = QHBoxLayout()
        
        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        button_layout.addWidget(self.add_layer_button)
        
        self.add_activation_button = QPushButton("Add Activation")
        self.add_activation_button.clicked.connect(self.add_activation)
        button_layout.addWidget(self.add_activation_button)
        
        self.generate_button = QPushButton("Generate Code")
        self.generate_button.clicked.connect(self.generate_code)
        button_layout.addWidget(self.generate_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        self.layers = []
        self.layer_options = ["Linear", "BatchNorm1d", "Dropout"]
        self.activation_options = ["ReLU", "Sigmoid", "Tanh"]
    
    def add_layer(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Layer")
        layout = QVBoxLayout()
        
        layer_type_dropdown = QComboBox()
        layer_type_dropdown.addItems(self.layer_options)
        layout.addWidget(layer_type_dropdown)
        
        param_entry = QLineEdit()
        param_entry.setPlaceholderText("Enter parameters")
        layout.addWidget(param_entry)
        
        def update_placeholder():
            if layer_type_dropdown.currentText() == "Linear":
                param_entry.setPlaceholderText("input_size, output_size")
            else:
                param_entry.setPlaceholderText("Enter parameters")
        
        layer_type_dropdown.currentIndexChanged.connect(update_placeholder)
        update_placeholder()
        
        def confirm():
            layer_type = layer_type_dropdown.currentText()
            params = param_entry.text()
            self.layers.append((layer_type, params))
            self.layer_list.addItem(f"{layer_type}({params})")
            dialog.accept()
        
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(confirm)
        layout.addWidget(confirm_button)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def add_activation(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Activation Function")
        layout = QVBoxLayout()
        
        activation_dropdown = QComboBox()
        activation_dropdown.addItems(self.activation_options)
        layout.addWidget(activation_dropdown)
        
        def confirm():
            activation = activation_dropdown.currentText()
            self.layers.append((activation, ""))
            self.layer_list.addItem(f"{activation}()")
            dialog.accept()
        
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(confirm)
        layout.addWidget(confirm_button)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def generate_code(self):
        model_name = self.model_name_entry.text().strip()
        if not model_name:
            model_name = "CustomNN"
        
        code = f"import torch.nn as nn\n\nclass {model_name}(nn.Module):\n    def __init__(self):\n        super({model_name}, self).__init__()\n        self.model = nn.Sequential(\n"
        
        for layer_type, params in self.layers:
            if params:
                code += f"            nn.{layer_type}({params}),\n"
            else:
                code += f"            nn.{layer_type}(),\n"
        
        code += "        )\n\n    def forward(self, x):\n        return self.model(x)\n"
        
        filename = f"{model_name.lower()}.py"
        with open(filename, "w") as f:
            f.write(code)
        
        QMessageBox.information(self, "Success", f"Model code saved as {filename}!")

if __name__ == "__main__":
    app = QApplication([])
    window = NNBuilderApp()
    window.show()
    app.exec()
