"""
NBA MVP Predictor - Native Desktop GUI (Tkinter)
Provides an interactive desktop interface with manual feature entry,
explicit input limits, historical averages, examples, and MVP classification output.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from model_service import MVPPredictionEngine, FEATURE_METADATA, FEATURES, PRESETS


class NBAMVPDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA MVP Predictor - Machine Learning GUI")
        self.root.geometry("980x760")
        self.root.minsize(880, 680)

        # Style & Colors
        self.bg_color = "#0b0f19"
        self.card_color = "#131a29"
        self.accent_gold = "#f59e0b"
        self.text_color = "#f8fafc"
        self.text_muted = "#94a3b8"

        self.root.configure(bg=self.bg_color)

        # Initialize ML engine
        self.engine = MVPPredictionEngine()

        self.input_vars = {}
        self.create_widgets()

        # Load first preset by default
        self.load_preset("jokic_2024")

    def create_widgets(self):
        # Header Frame
        header = tk.Frame(self.root, bg=self.card_color, pady=12, padx=16)
        header.pack(fill="x", padx=12, pady=(12, 6))

        title_lbl = tk.Label(
            header,
            text="🏆 NBA MVP MACHINE LEARNING PREDICTOR",
            font=("Helvetica", 16, "bold"),
            fg=self.accent_gold,
            bg=self.card_color
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header,
            text="Enter player stats manually below. Every feature lists its allowed limits, NBA averages, and MVP benchmarks.",
            font=("Helvetica", 9),
            fg=self.text_muted,
            bg=self.card_color
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # Preset Selector Bar
        preset_bar = tk.Frame(self.root, bg=self.bg_color)
        preset_bar.pack(fill="x", padx=12, pady=6)

        tk.Label(
            preset_bar,
            text="Historical Presets:",
            font=("Helvetica", 10, "bold"),
            fg=self.text_color,
            bg=self.bg_color
        ).pack(side="left", padx=(0, 8))

        for p_key, p_val in [
            ("jokic_2024", "Jokić '24 (MVP)"),
            ("sga_2025", "SGA '25 (MVP)"),
            ("curry_2016", "Curry '16 (MVP)"),
            ("jordan_1996", "Jordan '96 (MVP)"),
            ("all_star_non_mvp", "All-Star (Non-MVP)"),
            ("role_player", "Role Player (Non-MVP)")
        ]:
            btn = tk.Button(
                preset_bar,
                text=p_val,
                font=("Helvetica", 8, "bold"),
                bg="#1e293b",
                fg="#f1f5f9",
                activebackground=self.accent_gold,
                activeforeground="#000000",
                relief="flat",
                padx=8,
                pady=3,
                command=lambda k=p_key: self.load_preset(k)
            )
            btn.pack(side="left", padx=3)

        # Main Split Content: Left = Inputs, Right = Results
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=12, pady=6)

        # Left: Inputs (Scrollable or Clean Grid)
        left_frame = tk.LabelFrame(
            content_frame,
            text=" Manual Feature Inputs & Statistical Limits ",
            font=("Helvetica", 10, "bold"),
            fg=self.accent_gold,
            bg=self.card_color,
            padx=12,
            pady=10
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Model Option Controls inside Left Frame
        controls_row = tk.Frame(left_frame, bg=self.card_color)
        controls_row.pack(fill="x", pady=(0, 8))

        tk.Label(controls_row, text="Algorithm:", font=("Helvetica", 9), fg=self.text_muted, bg=self.card_color).pack(side="left")
        self.model_choice = tk.StringVar(value="rf")
        model_dropdown = ttk.Combobox(
            controls_row,
            textvariable=self.model_choice,
            values=["rf", "logreg", "nb", "ensemble"],
            state="readonly",
            width=10
        )
        model_dropdown.pack(side="left", padx=(4, 16))

        tk.Label(controls_row, text="Mode:", font=("Helvetica", 9), fg=self.text_muted, bg=self.card_color).pack(side="left")
        self.mode_choice = tk.StringVar(value="full")
        mode_dropdown = ttk.Combobox(
            controls_row,
            textvariable=self.mode_choice,
            values=["full", "split"],
            state="readonly",
            width=8
        )
        mode_dropdown.pack(side="left", padx=(4, 8))

        # Feature Inputs Grid Frame
        grid_frame = tk.Frame(left_frame, bg=self.card_color)
        grid_frame.pack(fill="both", expand=True)

        for idx, feat in enumerate(FEATURES):
            meta = FEATURE_METADATA[feat]
            var = tk.StringVar(value=str(meta["example"]))
            self.input_vars[feat] = var

            row_frame = tk.Frame(grid_frame, bg=self.card_color, pady=3)
            row_frame.pack(fill="x")

            # Col 1: Name and Abbreviation
            lbl_name = tk.Label(
                row_frame,
                text=f"{meta['name']} ({meta['short']}):",
                font=("Helvetica", 9, "bold"),
                fg=self.text_color,
                bg=self.card_color,
                width=24,
                anchor="w"
            )
            lbl_name.pack(side="left")

            # Col 2: Text Input Field
            entry = tk.Entry(
                row_frame,
                textvariable=var,
                font=("Consolas", 10),
                width=8,
                bg="#090d16",
                fg="#f8fafc",
                insertbackground="#f8fafc",
                relief="solid",
                bd=1
            )
            entry.pack(side="left", padx=(0, 8))

            # Col 3: Limits & Examples Guidance
            limit_text = f"Limits: [{meta['min']} - {meta['max']}] | NBA Avg: {meta['league_avg']} | MVP Avg: {meta['mvp_avg']} (e.g. {meta['example']})"
            lbl_meta = tk.Label(
                row_frame,
                text=limit_text,
                font=("Helvetica", 8),
                fg=self.text_muted,
                bg=self.card_color,
                anchor="w"
            )
            lbl_meta.pack(side="left", fill="x", expand=True)

        # Predict Button
        btn_predict = tk.Button(
            left_frame,
            text="⚡ PREDICT MVP STATUS",
            font=("Helvetica", 11, "bold"),
            bg="#f59e0b",
            fg="#0b0f19",
            activebackground="#fbbf24",
            relief="flat",
            pady=6,
            command=self.execute_prediction
        )
        btn_predict.pack(fill="x", pady=(12, 0))

        # Right: Results Panel
        right_frame = tk.LabelFrame(
            content_frame,
            text=" Model Prediction Output & Consensus ",
            font=("Helvetica", 10, "bold"),
            fg=self.accent_gold,
            bg=self.card_color,
            padx=12,
            pady=10
        )
        right_frame.pack(side="right", fill="both", expand=True, padx=(6, 0))

        # Main Verdict Box
        self.verdict_box = tk.Frame(right_frame, bg="#1e293b", padx=16, pady=16, bd=2, relief="solid")
        self.verdict_box.pack(fill="x", pady=(0, 10))

        self.verdict_lbl = tk.Label(
            self.verdict_box,
            text="PREDICTED MVP: --",
            font=("Helvetica", 16, "bold"),
            fg="#ffffff",
            bg="#1e293b"
        )
        self.verdict_lbl.pack()

        self.prob_lbl = tk.Label(
            self.verdict_box,
            text="MVP Probability: --%",
            font=("Helvetica", 11),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.prob_lbl.pack(pady=(4, 0))

        # Consensus Breakdown Label
        tk.Label(
            right_frame,
            text="Algorithms Consensus Breakdown:",
            font=("Helvetica", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color
        ).pack(anchor="w", pady=(6, 2))

        self.consensus_text = tk.Text(
            right_frame,
            height=7,
            bg="#090d16",
            fg="#e2e8f0",
            font=("Consolas", 9),
            bd=1,
            relief="solid",
            padx=8,
            pady=6
        )
        self.consensus_text.pack(fill="x", pady=(0, 10))

        # Scouting Notes Label
        tk.Label(
            right_frame,
            text="Scouting & Comparative Notes:",
            font=("Helvetica", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color
        ).pack(anchor="w", pady=(0, 2))

        self.notes_text = tk.Text(
            right_frame,
            height=8,
            bg="#090d16",
            fg="#e2e8f0",
            font=("Consolas", 9),
            bd=1,
            relief="solid",
            padx=8,
            pady=6
        )
        self.notes_text.pack(fill="both", expand=True)

    def load_preset(self, preset_key):
        if preset_key in PRESETS:
            preset = PRESETS[preset_key]
            for feat, val in preset["stats"].items():
                if feat in self.input_vars:
                    self.input_vars[feat].set(str(val))
            self.execute_prediction()

    def execute_prediction(self):
        # Validate inputs
        features_dict = {}
        for feat in FEATURES:
            try:
                val = float(self.input_vars[feat].get())
                meta = FEATURE_METADATA[feat]
                # soft check warning if out of absolute limits
                if val < meta["min"] or val > meta["max"]:
                    messagebox.showwarning(
                        "Out of Bounds Warning",
                        f"{meta['name']} value ({val}) is outside standard NBA limits [{meta['min']} - {meta['max']}]."
                    )
                features_dict[feat] = val
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid numeric value for {FEATURE_METADATA[feat]['name']}.")
                return

        model_name = self.model_choice.get()
        mode = self.mode_choice.get()

        res = self.engine.predict(features_dict, model_name=model_name, mode=mode)

        # Update Verdict Display
        if res["is_mvp"]:
            self.verdict_box.config(bg="#064e3b")
            self.verdict_lbl.config(text="🏆 PREDICTED MVP: YES", fg="#6ee7b7", bg="#064e3b")
            self.prob_lbl.config(
                text=f"Likelihood: {res['prob_percentage']}% (Score: {res['calibrated_score']}/100)",
                fg="#a7f3d0",
                bg="#064e3b"
            )
        else:
            self.verdict_box.config(bg="#7f1d1d")
            self.verdict_lbl.config(text="❌ PREDICTED MVP: NO", fg="#fca5a5", bg="#7f1d1d")
            self.prob_lbl.config(
                text=f"Likelihood: {res['prob_percentage']}% (Score: {res['calibrated_score']}/100)",
                fg="#fecaca",
                bg="#7f1d1d"
            )

        # Update Consensus Text
        self.consensus_text.delete("1.0", tk.END)
        c = res["models_consensus"]
        lines = [
            f"• Random Forest (#1):       {'MVP: YES' if c['rf']['is_mvp'] else 'MVP: NO '} ({c['rf']['prob_pct']}%) [{c['rf']['rank_grade']}]",
            f"• Logistic Regression:      {'MVP: YES' if c['logreg']['is_mvp'] else 'MVP: NO '} ({c['logreg']['prob_pct']}%)",
            f"• Gaussian Naive Bayes:     {'MVP: YES' if c['nb']['is_mvp'] else 'MVP: NO '} ({c['nb']['prob_pct']}%)",
        ]
        if res.get("eligibility_warning"):
            lines.append(f"\n⚠️  {res['eligibility_warning']}")
        self.consensus_text.insert(tk.END, "\n".join(lines))

        # Update Notes Text
        self.notes_text.delete("1.0", tk.END)
        notes = []
        for insight in res.get("insights", []):
            notes.append(f"• {insight}")
        notes.append("\nTop RF Feature Importances:")
        for imp in res.get("feature_importance", [])[:4]:
            notes.append(f"  - {imp['name']} ({imp['short']}): {imp['importance']}% importance (Your val: {imp['value']})")
        self.notes_text.insert(tk.END, "\n".join(notes))


if __name__ == "__main__":
    root = tk.Tk()
    app = NBAMVPDesktopApp(root)
    root.mainloop()
