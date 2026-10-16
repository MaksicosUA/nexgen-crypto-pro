import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import scrolledtext, ttk

from predict import DEFAULT_MODELS_ROOT, discover_model_bundles, run_prediction, write_json


DEFAULT_OUTPUT = "/tmp/nexgen_prediction_gui.json"


def bundle_label(bundle):
    return f"{bundle.id}  |  {bundle.model_type}  |  {bundle.quality_bucket}  |  {bundle.pair}"


class PredictionGui:
    def __init__(self, root):
        self.root = root
        self.root.title("NEXGEN Inference Launcher")
        self.root.geometry("980x720")
        self.root.minsize(880, 620)

        self.result_queue = queue.Queue()
        self.bundles = []
        self.bundle_by_label = {}

        self.model_var = StringVar()
        self.data_var = StringVar()
        self.output_var = StringVar(value=DEFAULT_OUTPUT)
        self.status_var = StringVar(value="Ready")
        self.all_sequences_var = BooleanVar(value=False)
        self.limit_var = StringVar(value="10")

        self._build_layout()
        self.refresh_models()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(8, weight=1)

        ttk.Label(main, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model_combo = ttk.Combobox(main, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Refresh list", command=self.refresh_models).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(main, text="CSV data").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(main, textvariable=self.data_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Choose CSV", command=self.browse_data).grid(row=1, column=2, padx=(8, 0), pady=4)

        ttk.Label(main, text="JSON output").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(main, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(main, text="Save as", command=self.browse_output).grid(row=2, column=2, padx=(8, 0), pady=4)

        options = ttk.Frame(main)
        options.grid(row=3, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            options,
            text="Forecast for all sequences",
            variable=self.all_sequences_var,
        ).pack(side="left")
        ttk.Label(options, text="  limit:").pack(side="left")
        ttk.Entry(options, textvariable=self.limit_var, width=8).pack(side="left", padx=(4, 0))

        actions = ttk.Frame(main)
        actions.grid(row=4, column=1, sticky="w", pady=(8, 4))
        self.run_button = ttk.Button(actions, text="Run prediction", command=self.run_prediction_clicked)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Clear output", command=self.clear_output).pack(side="left", padx=(8, 0))

        ttk.Label(main, textvariable=self.status_var).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Separator(main).grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(main, text="Result").grid(row=7, column=0, sticky="nw", padx=(0, 8), pady=4)
        self.output_text = scrolledtext.ScrolledText(main, wrap="word", height=24)
        self.output_text.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=4)

    def refresh_models(self):
        try:
            self.bundles = discover_model_bundles(DEFAULT_MODELS_ROOT)
            labels = [bundle_label(bundle) for bundle in self.bundles]
            self.bundle_by_label = dict(zip(labels, self.bundles))
            self.model_combo["values"] = labels
            if labels:
                self.model_var.set(labels[0])
                self.status_var.set(f"Models found: {len(labels)}")
            else:
                self.model_var.set("")
                self.status_var.set("No models with a paired scaler found")
        except Exception as exc:
            messagebox.showerror("Model list error", str(exc))
            self.status_var.set("Failed to refresh the model list")

    def browse_data(self):
        path = filedialog.askopenfilename(
            title="Choose a CSV data file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.data_var.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Where to save the prediction JSON",
            defaultextension=".json",
            initialfile=Path(self.output_var.get() or DEFAULT_OUTPUT).name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def clear_output(self):
        self.output_text.delete("1.0", "end")

    def append_output(self, text):
        self.output_text.insert("end", text)
        self.output_text.see("end")

    def validate_form(self):
        selected_label = self.model_var.get()
        bundle = self.bundle_by_label.get(selected_label)
        if bundle is None:
            raise ValueError("Choose a model from the drop-down list")

        data_path = self.data_var.get().strip()
        if not data_path:
            raise ValueError("Choose a CSV data file")
        if not Path(data_path).exists():
            raise ValueError(f"CSV file not found: {data_path}")

        output_path = self.output_var.get().strip() or DEFAULT_OUTPUT
        limit_raw = self.limit_var.get().strip() or "10"
        try:
            limit = int(limit_raw)
        except ValueError as exc:
            raise ValueError(f"limit must be an integer, got: {limit_raw!r}") from exc
        if limit < 0:
            raise ValueError("limit cannot be negative")

        return bundle, data_path, output_path, limit

    def run_prediction_clicked(self):
        try:
            bundle, data_path, output_path, limit = self.validate_form()
        except Exception as exc:
            messagebox.showwarning("Check the parameters", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Prediction in progress...")
        self.clear_output()
        self.append_output(
            "Starting inference-only prediction\n"
            f"Model: {bundle.id}\n"
            f"Data: {data_path}\n"
            f"Output: {output_path}\n\n"
        )

        worker = threading.Thread(
            target=self._prediction_worker,
            args=(bundle, data_path, output_path, limit, self.all_sequences_var.get()),
            daemon=True,
        )
        worker.start()
        self.root.after(250, self._poll_result_queue)

    def _prediction_worker(self, bundle, data_path, output_path, limit, all_sequences):
        try:
            result = run_prediction(
                data_path=data_path,
                model_path=bundle.model_path,
                scaler_path=bundle.scaler_path,
                all_sequences=all_sequences,
                limit=limit,
            )
            result["selected_model"] = asdict(bundle)
            write_json(output_path, result)
            self.result_queue.put(("success", output_path, result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc), None))

    def _poll_result_queue(self):
        try:
            status, payload, result = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(250, self._poll_result_queue)
            return

        self.run_button.configure(state="normal")
        if status == "error":
            self.status_var.set("Prediction failed")
            self.append_output(f"Error: {payload}\n")
            messagebox.showerror("Prediction error", payload)
            return

        self.status_var.set(f"Prediction ready: {payload}")
        self.append_output(json.dumps(result, ensure_ascii=False, indent=2))
        self.append_output(f"\n\nPrediction JSON saved to {payload}\n")


def main():
    root = Tk()
    app = PredictionGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
