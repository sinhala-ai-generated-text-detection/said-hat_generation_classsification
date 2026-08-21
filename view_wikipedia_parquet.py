"""GUI viewer for wikipedia.parquet.

Opens a window listing every row (id, title, url) with a search box,
and a text pane that shows the full 'text' / 'raw_mediawiki' content
of whichever row you select. Avoids the terminal and avoids Excel's
per-cell character truncation on long article bodies.

Usage:
    python view_wikipedia_parquet.py [path/to/file.parquet]
"""

import sys
import tkinter as tk
from tkinter import ttk

import pandas as pd

DEFAULT_PATH = "source_data/wikipedia-monthly/wikipedia.parquet"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


class ParquetViewer(tk.Tk):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.filtered_index = list(df.index)

        self.title(f"Parquet Viewer - {len(df)} rows")
        self.geometry("1200x700")

        self._build_widgets()
        self._populate_rows(self.filtered_index)

    def _build_widgets(self):
        # Top: search bar
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(top, text="Search title/url:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(top, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        search_entry.bind("<Return>", lambda e: self._apply_filter())
        ttk.Button(top, text="Search", command=self._apply_filter).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear", command=self._clear_filter).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        # Main split: row list (left) | detail pane (right)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Left: Treeview of rows
        left = ttk.Frame(paned)
        columns = ("id", "title", "url")
        self.tree = ttk.Treeview(
            left, columns=columns, show="headings", selectmode="browse"
        )
        for col, width in zip(columns, (100, 300, 300)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=tk.W)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(left, weight=1)

        # Right: detail text pane with tabs for text / raw_mediawiki
        right = ttk.Frame(paned)
        self.detail_header = ttk.Label(
            right,
            text="",
            wraplength=600,
            justify=tk.LEFT,
            font=("Segoe UI", 10, "bold"),
        )
        self.detail_header.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(notebook)
        self.text_widget = tk.Text(text_frame, wrap=tk.WORD)
        text_vsb = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.text_widget.yview
        )
        self.text_widget.configure(yscrollcommand=text_vsb.set)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_vsb.pack(side=tk.LEFT, fill=tk.Y)
        notebook.add(text_frame, text="text")

        raw_frame = ttk.Frame(notebook)
        self.raw_widget = tk.Text(raw_frame, wrap=tk.WORD)
        raw_vsb = ttk.Scrollbar(
            raw_frame, orient=tk.VERTICAL, command=self.raw_widget.yview
        )
        self.raw_widget.configure(yscrollcommand=raw_vsb.set)
        self.raw_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        raw_vsb.pack(side=tk.LEFT, fill=tk.Y)
        notebook.add(raw_frame, text="raw_mediawiki")

        paned.add(right, weight=2)

    def _populate_rows(self, indices):
        self.tree.delete(*self.tree.get_children())
        for idx in indices:
            row = self.df.loc[idx]
            self.tree.insert(
                "", tk.END, iid=str(idx), values=(row["id"], row["title"], row["url"])
            )

    def _apply_filter(self):
        query = self.search_var.get().strip().lower()
        if not query:
            self._clear_filter()
            return
        mask = self.df["title"].astype(str).str.lower().str.contains(
            query, na=False
        ) | self.df["url"].astype(str).str.lower().str.contains(query, na=False)
        self.filtered_index = list(self.df[mask].index)
        self.title(f"Parquet Viewer - {len(self.filtered_index)}/{len(self.df)} rows")
        self._populate_rows(self.filtered_index)

    def _clear_filter(self):
        self.search_var.set("")
        self.filtered_index = list(self.df.index)
        self.title(f"Parquet Viewer - {len(self.df)} rows")
        self._populate_rows(self.filtered_index)

    def _on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        row = self.df.loc[idx]

        self.detail_header.config(
            text=f"id: {row['id']}    title: {row['title']}\nurl: {row['url']}"
        )

        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, str(row["text"]))

        self.raw_widget.delete("1.0", tk.END)
        self.raw_widget.insert(tk.END, str(row["raw_mediawiki"]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    df = load_data(path)
    app = ParquetViewer(df)
    app.mainloop()


if __name__ == "__main__":
    main()
