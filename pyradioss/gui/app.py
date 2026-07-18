"""
The Tkinter application window for the pyradioss GUI.

No Fortran counterpart (see the package docstring; ``OpenRadioss/
openradioss_gui`` — Altair's job launcher — is the inspiration). This module
wires the pure logic in :mod:`pyradioss.gui.runner` and the plotting in
:mod:`pyradioss.gui.plots` into a live window:

* **Job panel** — pick a ``*_0000.rad`` starter deck (remembering the last
  directory in the JSON config), auto-derive the ``*_0001.rad`` engine deck,
  choose the compute backend (auto/numpy/numba, ``auto`` preselected to match
  the M40 default), Run (STARTER then ENGINE) and Stop.
* **Log pane** — the children's stdout, streamed line by line.
* **Progress** — the parsed cycle/time/dt/error into a status bar and a
  progress bar against the ``/RUN`` end time; the TERMINATION banner shown
  prominently (green NORMAL / red ERROR).
* **Results tab** — load and plot the T01 channels (embedded matplotlib, or a
  textual table when matplotlib is absent).
* **Deck info tab** — the port's read-only deck reader's model summary.

All subprocess I/O is drained from the Tk event loop via ``after()`` so the
UI never blocks; closing the window terminates any live child.
"""

from __future__ import annotations

import os
import queue
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from . import postproc
from .plots import DEFAULT_CHANNELS, ResultsView, available_channels
from .runner import (GuiConfig, JobRunner, build_deck_summary,
                     derive_engine_deck, load_t01)

_POLL_MS = 100          # queue-drain cadence
_BACKENDS = ("auto", "numpy", "numba")


class PyradiossGUI:
    """The application: builds the widgets on a (caller-supplied) Tk root."""

    def __init__(self, root: tk.Tk, deck: Optional[str] = None):
        self.root = root
        self.root.title("pyradioss — run & monitor")
        self.config = GuiConfig().load()
        self.runner: Optional[JobRunner] = None
        self.t01 = None

        self.deck_var = tk.StringVar(value=deck or "")
        self.engine_var = tk.StringVar(value="")
        self.backend_var = tk.StringVar(
            value=str(self.config.get("backend", "auto")))
        self.nthread_var = tk.IntVar(
            value=int(self.config.get("nthread", 0) or 0))
        self.status_var = tk.StringVar(value="Ready.")
        self.term_var = tk.StringVar(value="")
        self.channel_vars: dict = {}

        # -- post-processing state --------------------------------------
        self.post_runner = None
        self.exec_dir_var = tk.StringVar(
            value=str(self.config.get("exec_dir", "")
                      or postproc.DEFAULT_EXEC_DIR))
        self.auto_d3plot_var = tk.BooleanVar(
            value=bool(self.config.get("auto_d3plot", False)))
        self.auto_vtk_var = tk.BooleanVar(
            value=bool(self.config.get("auto_vtk", False)))
        self.auto_th_csv_var = tk.BooleanVar(
            value=bool(self.config.get("auto_th_csv", False)))
        self.post_dir_var = tk.StringVar(value="")
        self.artifacts_var = tk.StringVar(value="")

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if deck:
            self._on_deck_chosen(deck)
        self.root.after(_POLL_MS, self._drain_queue)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        job = ttk.LabelFrame(self.root, text="Job")
        job.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(job, text="Starter deck (*_0000.rad):").grid(
            row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(job, textvariable=self.deck_var, width=60).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=4, pady=4)
        ttk.Button(job, text="Browse…", command=self.browse_deck).grid(
            row=0, column=4, padx=4, pady=4)

        ttk.Label(job, text="Engine deck:").grid(
            row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(job, textvariable=self.engine_var,
                  foreground="#555").grid(
            row=1, column=1, columnspan=3, sticky="w", padx=4, pady=4)

        ttk.Label(job, text="Backend:").grid(
            row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(job, textvariable=self.backend_var, values=_BACKENDS,
                     state="readonly", width=10).grid(
            row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(job, text="Threads (0=auto):").grid(
            row=2, column=2, sticky="e", padx=4, pady=4)
        ttk.Spinbox(job, from_=0, to=64, textvariable=self.nthread_var,
                    width=6).grid(row=2, column=3, sticky="w", padx=4, pady=4)

        self.run_btn = ttk.Button(job, text="Run", command=self.run_job)
        self.run_btn.grid(row=2, column=4, padx=4, pady=4, sticky="we")
        self.stop_btn = ttk.Button(job, text="Stop", command=self.stop_job,
                                   state=tk.DISABLED)
        self.stop_btn.grid(row=3, column=4, padx=4, pady=(0, 4), sticky="we")
        job.columnconfigure(1, weight=1)

        # auto-convert-after-run toggles (persisted in the config)
        auto = ttk.Frame(job)
        auto.grid(row=4, column=0, columnspan=4, sticky="w", padx=4,
                  pady=(2, 4))
        ttk.Label(auto, text="After run:").pack(side=tk.LEFT)
        ttk.Checkbutton(auto, text="d3plot", variable=self.auto_d3plot_var,
                        command=self._persist_post_opts).pack(side=tk.LEFT,
                                                              padx=(4, 0))
        ttk.Checkbutton(auto, text="VTK", variable=self.auto_vtk_var,
                        command=self._persist_post_opts).pack(side=tk.LEFT,
                                                              padx=(4, 0))
        ttk.Checkbutton(auto, text="TH->CSV", variable=self.auto_th_csv_var,
                        command=self._persist_post_opts).pack(side=tk.LEFT,
                                                              padx=(4, 0))

        # -- notebook: Log / Results / Deck info ------------------------
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Log tab
        log_tab = ttk.Frame(self.nb)
        self.nb.add(log_tab, text="Log")
        self.log = scrolledtext.ScrolledText(
            log_tab, wrap=tk.NONE, height=20, font=("Courier New", 9))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_config("normal", foreground="#0a0")
        self.log.tag_config("error", foreground="#c00")

        # Results tab
        res_tab = ttk.Frame(self.nb)
        self.nb.add(res_tab, text="Results")
        self._build_results_tab(res_tab)

        # Deck info tab
        info_tab = ttk.Frame(self.nb)
        self.nb.add(info_tab, text="Deck info")
        self._build_info_tab(info_tab)

        # Post-processing tab
        post_tab = ttk.Frame(self.nb)
        self.nb.add(post_tab, text="Post-processing")
        self._build_postproc_tab(post_tab)

        # -- status bar + progress --------------------------------------
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.term_label = tk.Label(bar, textvariable=self.term_var,
                                   font=("Segoe UI", 9, "bold"), width=22,
                                   anchor="w")
        self.term_label.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(bar, length=180, mode="determinate",
                                        maximum=100.0)
        self.progress.pack(side=tk.LEFT, padx=8)
        ttk.Label(bar, textvariable=self.status_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=4)
        ttk.Button(top, text="Load / reload T01",
                   command=self.load_results).pack(side=tk.LEFT, padx=4)
        self.channels_frame = ttk.LabelFrame(top, text="Channels")
        self.channels_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        # default channel toggles (rebuilt when a T01 is loaded)
        selected = set(self.config.get("channels", DEFAULT_CHANNELS) or [])
        for chan in DEFAULT_CHANNELS:
            self._add_channel_toggle(chan, chan in selected or not selected)

        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.results_view = ResultsView(plot_frame)

    def _add_channel_toggle(self, chan: str, on: bool) -> None:
        if chan in self.channel_vars:
            return
        var = tk.BooleanVar(value=on)
        self.channel_vars[chan] = var
        ttk.Checkbutton(self.channels_frame, text=chan, variable=var,
                        command=self._refresh_plot).pack(side=tk.LEFT)

    def _build_info_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=4)
        ttk.Button(top, text="Inspect deck",
                   command=self.inspect_deck).pack(side=tk.LEFT, padx=4)
        self.info_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Courier New", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Post-processing tab
    # ------------------------------------------------------------------
    def _build_postproc_tab(self, parent: ttk.Frame) -> None:
        # run-directory picker
        dirf = ttk.LabelFrame(parent, text="Run directory")
        dirf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Entry(dirf, textvariable=self.post_dir_var, width=54).grid(
            row=0, column=0, columnspan=2, sticky="we", padx=4, pady=4)
        ttk.Button(dirf, text="Browse…",
                   command=self.browse_post_dir).grid(
            row=0, column=2, padx=4, pady=4)
        ttk.Button(dirf, text="Detect artifacts",
                   command=self.refresh_artifacts).grid(
            row=0, column=3, padx=4, pady=4)
        dirf.columnconfigure(0, weight=1)

        # converter exe directory (config)
        ttk.Label(dirf, text="OpenRadioss exec dir:").grid(
            row=1, column=0, sticky="w", padx=4, pady=(0, 4))
        ttk.Entry(dirf, textvariable=self.exec_dir_var, width=44).grid(
            row=1, column=1, sticky="we", padx=4, pady=(0, 4))
        ttk.Button(dirf, text="Set", command=self._persist_post_opts).grid(
            row=1, column=2, padx=4, pady=(0, 4))

        # artifact summary + converter buttons
        act = ttk.LabelFrame(parent, text="Converters")
        act.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(act, textvariable=self.artifacts_var, foreground="#555",
                  justify=tk.LEFT).grid(row=0, column=0, columnspan=3,
                                        sticky="w", padx=4, pady=4)
        self.d3_btn = ttk.Button(act, text="Convert to d3plot",
                                 command=lambda: self.run_conversion("d3plot"))
        self.d3_btn.grid(row=1, column=0, padx=4, pady=4, sticky="we")
        self.vtk_btn = ttk.Button(act, text="ANIM -> VTK",
                                  command=lambda: self.run_conversion("vtk"))
        self.vtk_btn.grid(row=1, column=1, padx=4, pady=4, sticky="we")
        self.thcsv_btn = ttk.Button(
            act, text="TH -> CSV",
            command=lambda: self.run_conversion("th_csv"))
        self.thcsv_btn.grid(row=1, column=2, padx=4, pady=4, sticky="we")
        for c in range(3):
            act.columnconfigure(c, weight=1)

        # output list
        outf = ttk.LabelFrame(parent, text="Outputs (see the Log tab for "
                                           "streamed progress)")
        outf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.post_out = scrolledtext.ScrolledText(
            outf, wrap=tk.NONE, height=8, font=("Courier New", 9))
        self.post_out.pack(fill=tk.BOTH, expand=True)
        self._render_artifacts(None)

    def browse_post_dir(self) -> None:
        init = self.post_dir_var.get() or self.config.get("last_dir", "") \
            or os.getcwd()
        path = filedialog.askdirectory(title="Select run directory",
                                       initialdir=init)
        if path:
            self.post_dir_var.set(path)
            self.refresh_artifacts()

    def _persist_post_opts(self) -> None:
        self.config.set("exec_dir", self.exec_dir_var.get().strip())
        self.config.set("auto_d3plot", bool(self.auto_d3plot_var.get()))
        self.config.set("auto_vtk", bool(self.auto_vtk_var.get()))
        self.config.set("auto_th_csv", bool(self.auto_th_csv_var.get()))
        self.config.save()

    def refresh_artifacts(self) -> None:
        run_dir = self.post_dir_var.get().strip()
        arts = postproc.detect_artifacts(run_dir) if run_dir else None
        self._render_artifacts(arts)

    def _render_artifacts(self, arts) -> None:
        if not arts or not arts.get("run_dir"):
            self.artifacts_var.set("Pick a run directory, then Detect "
                                   "artifacts.")
            for btn in (self.d3_btn, self.vtk_btn, self.thcsv_btn):
                btn.configure(state=tk.DISABLED)
            return
        n_anim = len(arts["anim_files"])
        lines = [
            f"A-files (anim): {n_anim}"
            + (f"  stem={os.path.basename(arts['anim_stem'])}"
               if arts["anim_stem"] else ""),
            f"binary TH (Txx): {'yes' if arts['th_binary'] else 'no'}"
            + ("  (pyradioss emits T01 as CSV natively; TH->CSV is for a "
               "Fortran binary)" if not arts["th_binary"] else ""),
            f"port-native CSV: {len(arts['port_csv'])}   "
            f"port-native VTK: {len(arts['port_vtk'])}   "
            f"existing d3plot: {'yes' if arts['d3plot'] else 'no'}",
        ]
        if not postproc.vortex_available():
            lines.append("Vortex-Radioss not installed -> d3plot disabled. "
                         "Install: " + postproc.VORTEX_INSTALL_HINT)
        self.artifacts_var.set("\n".join(lines))
        busy = self.post_runner is not None and self.post_runner.is_running()
        self.d3_btn.configure(
            state=(tk.NORMAL if arts["can_d3plot"]
                   and postproc.vortex_available() and not busy
                   else tk.DISABLED))
        self.vtk_btn.configure(
            state=(tk.NORMAL if arts["can_vtk"] and not busy
                   else tk.DISABLED))
        self.thcsv_btn.configure(
            state=(tk.NORMAL if arts["can_th_csv"] and not busy
                   else tk.DISABLED))

    def run_conversion(self, action: str) -> None:
        run_dir = self.post_dir_var.get().strip()
        if not run_dir or not os.path.isdir(run_dir):
            messagebox.showerror("pyradioss", "Select an existing run "
                                             "directory first.")
            return
        if self.post_runner is not None and self.post_runner.is_running():
            messagebox.showinfo("pyradioss", "A conversion is already "
                                            "running.")
            return
        self._persist_post_opts()
        self.nb.select(0)                       # show the Log tab
        self.status_var.set(f"Converting ({action})…")
        self.post_out.delete("1.0", tk.END)
        self.post_runner = postproc.PostProcRunner(
            run_dir, [action], exec_dir=self.exec_dir_var.get().strip()
            or None, python_exe=None)
        self.refresh_artifacts()                # greys the buttons (busy)
        self.post_runner.start()

    # ------------------------------------------------------------------
    # Job panel actions
    # ------------------------------------------------------------------
    def browse_deck(self) -> None:
        init_dir = self.config.get("last_dir", "") or os.getcwd()
        path = filedialog.askopenfilename(
            title="Select starter deck",
            initialdir=init_dir,
            filetypes=[("Radioss starter deck", "*_0000.rad"),
                       ("Radioss deck", "*.rad"), ("All files", "*.*")])
        if path:
            self._on_deck_chosen(path)

    def _on_deck_chosen(self, path: str) -> None:
        self.deck_var.set(path)
        self.engine_var.set(os.path.basename(derive_engine_deck(path)))
        self.config.set("last_dir", os.path.dirname(os.path.abspath(path)))
        self.config.save()

    def run_job(self) -> None:
        deck = self.deck_var.get().strip()
        if not deck or not os.path.exists(deck):
            messagebox.showerror("pyradioss", "Select an existing starter "
                                             "deck (*_0000.rad).")
            return
        if self.runner is not None and self.runner.is_running():
            messagebox.showinfo("pyradioss", "A job is already running.")
            return
        # persist current selections
        self.config.set("backend", self.backend_var.get())
        self.config.set("nthread", int(self.nthread_var.get()))
        self._persist_post_opts()

        post_actions = []
        if self.auto_d3plot_var.get():
            post_actions.append("d3plot")
        if self.auto_vtk_var.get():
            post_actions.append("vtk")
        if self.auto_th_csv_var.get():
            post_actions.append("th_csv")

        self.log.delete("1.0", tk.END)
        self.term_var.set("")
        self.term_label.configure(foreground="black")
        self.progress.configure(value=0.0)
        self.status_var.set("Starting…")
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

        self.runner = JobRunner(
            deck, backend=self.backend_var.get(),
            nthread=int(self.nthread_var.get()),
            post_actions=post_actions,
            exec_dir=self.exec_dir_var.get().strip() or None)
        self.runner.start()

    def stop_job(self) -> None:
        if self.runner is not None and self.runner.is_running():
            self.status_var.set("Stopping…")
            self.runner.stop()

    # ------------------------------------------------------------------
    # Queue draining (runs in the Tk event loop)
    # ------------------------------------------------------------------
    def _drain_queue(self) -> None:
        for src in (self.runner, self.post_runner):
            if src is not None:
                try:
                    while True:
                        self._handle_event(src.queue.get_nowait())
                except queue.Empty:
                    pass
        self.root.after(_POLL_MS, self._drain_queue)

    def _handle_event(self, event: tuple) -> None:
        tag = event[0]
        if tag == "line":
            _, phase, text = event
            self._append_log(text)
        elif tag == "phase":
            self.status_var.set(f"Running {event[1]}…")
        elif tag == "t_end":
            self._t_end = event[1]
        elif tag == "status":
            self._update_status(event[2])
        elif tag == "term":
            self._show_termination(event[1], event[2], event[3])
        elif tag == "done":
            self._job_done(event[1])
        elif tag == "post_done":
            self._post_done(event[1], event[2], event[3], event[4])
        elif tag == "post_all_done":
            self._post_all_done(event[1])

    def _append_log(self, text: str) -> None:
        tag = ""
        if "TERMINATION : NORMAL" in text:
            tag = "normal"
        elif "TERMINATION : ERROR" in text or "** ERROR" in text:
            tag = "error"
        self.log.insert(tk.END, text + "\n", tag)
        self.log.see(tk.END)

    def _update_status(self, status: dict) -> None:
        cycle = status.get("cycle", 0)
        t = status.get("time", 0.0)
        dt = status.get("dt", 0.0)
        err = status.get("err", 0.0)
        self.status_var.set(
            f"cycle {cycle}   t={t:.5E}   dt={dt:.3E}   error={err:.2f}%")
        t_end = getattr(self, "_t_end", 0.0)
        if t_end and t_end > 0:
            self.progress.configure(value=max(0.0, min(100.0,
                                                       100.0 * t / t_end)))

    def _show_termination(self, kind: str, status: str, reason: str) -> None:
        if status == "NORMAL":
            self.term_var.set(f"{kind}: NORMAL")
            self.term_label.configure(foreground="#0a0")
            if kind == "ENGINE":
                self.progress.configure(value=100.0)
        else:
            msg = f"{kind}: ERROR"
            if reason:
                msg += f" — {reason}"
            self.term_var.set(msg)
            self.term_label.configure(foreground="#c00")

    def _job_done(self, returncode: int) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        # seed the Post-processing tab with this run's directory + artifacts
        if self.runner is not None and self.runner.work_dir:
            self.post_dir_var.set(self.runner.work_dir)
            self.refresh_artifacts()
        if returncode == 0:
            self.status_var.set("Finished.")
            # auto-load results at the end of a clean run
            self.load_results(quiet=True)
        else:
            self.status_var.set(f"Finished (exit {returncode}).")

    # ------------------------------------------------------------------
    # Post-processing results
    # ------------------------------------------------------------------
    def _post_done(self, kind: str, ok: bool, outputs: list,
                   message: str) -> None:
        mark = "OK" if ok else "FAILED"
        self.post_out.insert(tk.END, f"[{kind}] {mark}: {message}\n")
        for path in outputs:
            self.post_out.insert(tk.END, f"    {path}\n")
        self.post_out.see(tk.END)

    def _post_all_done(self, results: dict) -> None:
        self.status_var.set("Post-processing finished.")
        self.refresh_artifacts()

    # ------------------------------------------------------------------
    # Results tab
    # ------------------------------------------------------------------
    def load_results(self, quiet: bool = False) -> None:
        path = self.runner.t01_path if self.runner is not None else None
        if not path:
            deck = self.deck_var.get().strip()
            if deck:
                from .runner import t01_path_for
                path = t01_path_for(derive_engine_deck(deck))
        if not path or not os.path.exists(path):
            if not quiet:
                messagebox.showinfo("pyradioss", "No T01 file found yet.")
            return
        try:
            self.t01 = load_t01(path)
        except OSError as exc:
            if not quiet:
                messagebox.showerror("pyradioss", f"Could not read T01:\n{exc}")
            return
        # extend the channel toggles with any extra channels the T01 has
        for chan in available_channels(self.t01):
            if chan not in self.channel_vars:
                self._add_channel_toggle(chan, False)
        self._refresh_plot()

    def _selected_channels(self) -> list:
        return [c for c, v in self.channel_vars.items() if v.get()]

    def _refresh_plot(self) -> None:
        if self.t01 is None:
            return
        channels = self._selected_channels()
        self.config.set("channels", channels)
        self.config.save()
        self.results_view.update_plot(
            self.t01, channels, getattr(self, "_t_end", 0.0))

    # ------------------------------------------------------------------
    # Deck info tab
    # ------------------------------------------------------------------
    def inspect_deck(self) -> None:
        deck = self.deck_var.get().strip()
        if not deck or not os.path.exists(deck):
            messagebox.showerror("pyradioss", "Select an existing deck first.")
            return
        self.status_var.set("Inspecting deck…")
        self.root.update_idletasks()
        summary = build_deck_summary(deck)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, self._format_summary(summary))
        self.status_var.set("Ready.")

    @staticmethod
    def _format_summary(summary: dict) -> str:
        lines = []
        lines.append(f"Deck: {summary['path']}")
        lines.append(f"Title: {summary['title']}")
        if summary["error"]:
            lines.append("")
            lines.append(f"!! FAILED TO READ DECK: {summary['error']}")
        lines.append("")
        lines.append(f"Nodes: {summary['nodes']}")
        lines.append("Elements by type:")
        elems = summary["elements"]
        if elems:
            for etype, count in sorted(elems.items()):
                lines.append(f"    {etype:<10s} {count}")
        else:
            lines.append("    (none)")

        lines.append("")
        lines.append(f"Materials ({len(summary['materials'])}):")
        for mat in summary["materials"]:
            lines.append(f"    MAT {mat['id']:<6d} LAW{mat['law']:<3d} "
                         f"{mat['law_name']:<12s} {mat['title']}")
        lines.append("")
        lines.append(f"Properties ({len(summary['properties'])}):")
        for prop in summary["properties"]:
            lines.append(f"    PROP {prop['id']:<6d} TYPE{prop['type']:<3d} "
                         f"{prop['title']}")
        lines.append("")
        lines.append(f"Parts ({len(summary['parts'])}):")
        for part in summary["parts"]:
            lines.append(f"    PART {part['id']:<6d} "
                         f"PROP={part['prop_id']:<6d} "
                         f"MAT={part['mat_id']:<6d} {part['title']}")
        lines.append("")
        lines.append(f"Contacts ({len(summary['contacts'])}):")
        for itf in summary["contacts"]:
            lines.append(f"    INTER {itf['id']:<6d} TYPE{itf['type']}")
        lines.append("")
        lines.append("Keywords:")
        for kw, count in summary["keywords"]:
            lines.append(f"    /{kw:<24s} x{count}")
        lines.append("")
        lines.append(f"{summary['errors']} error(s), "
                     f"{summary['warnings']} warning(s) while reading.")
        if summary["listing"].strip():
            lines.append("")
            lines.append("---- reader listing ----")
            lines.append(summary["listing"].rstrip())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def on_close(self) -> None:
        if self.runner is not None and self.runner.is_running():
            if not messagebox.askokcancel(
                    "pyradioss", "A job is running. Stop it and close?"):
                return
            self.runner.stop()
            self.runner.join(timeout=5)
        if self.post_runner is not None and self.post_runner.is_running():
            self.post_runner.join(timeout=5)
        self.root.destroy()


def launch(deck: Optional[str] = None) -> None:
    """Create the Tk root and run the application main loop."""
    root = tk.Tk()
    PyradiossGUI(root, deck=deck)
    root.mainloop()
