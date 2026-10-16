import PySimpleGUI as sg
import os
import subprocess

MODEL_FILTERS = [
    ("Model Files (.keras, .h5)", "*.keras;*.h5"),
    ("Model Files (.keras)", "*.keras"),
    ("Model Files (.h5)", "*.h5"),
    ("All Files", "*.*"),
]

layout = [
    [sg.Text("Select data CSV"), sg.InputText(key='data'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
    [sg.Text("Select model file (.keras/.h5)"),
     sg.InputText(key='model'),
     sg.FileBrowse(file_types=MODEL_FILTERS, tooltip="If the file is not listed, choose 'All Files' and select it manually.")],
    [sg.Text("Select scaler (.pkl)"), sg.InputText(key='scaler'), sg.FileBrowse(file_types=(("Pickle Files", "*.pkl"),))],
    [sg.Text("Window Size:"), sg.InputText("60", key='window', size=(10,1))],
    [sg.Text("Steps Ahead:"), sg.InputText("1", key='steps', size=(10,1))],
    [sg.Text("Output file:"), sg.InputText("logs/backtest_LTCUSDT.png", key='output'), sg.FileSaveAs(file_types=(("PNG Files", "*.png"),))],
    [sg.Text("Start balance:"), sg.InputText("1000", key='start_balance', size=(10,1))],
    [sg.Button("Run Backtest"), sg.Button("Exit")],
    [sg.Output(size=(90, 15))]
]

window = sg.Window("Backtest Launcher", layout, finalize=True)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    if event == "Run Backtest":
        # Validate fields
        if not all([values['data'], values['model'], values['scaler'], values['output']]):
            print("❗️ Not all fields are filled in!")
            continue
        command = [
            "python3",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest.py"),
            "--data", values['data'],
            "--model", values['model'],
            "--scaler", values['scaler'],
            "--window", values['window'],
            "--steps", values['steps'],
            "--output", values['output'],
            "--start-balance", values['start_balance']
        ]
        print("== Running command:")
        print(" ".join(command))
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("stderr:", result.stderr)
        except Exception as e:
            print("Launch error:", e)

window.close()
