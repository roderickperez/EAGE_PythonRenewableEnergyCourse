"""Serve the local book with notebooks labelled JSON for download regression checks."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOOK_HTML = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse" / "_build" / "html"


class JsonNotebookHandler(SimpleHTTPRequestHandler):
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map, **{ ".ipynb": "application/json" })


if __name__ == "__main__":
    if not BOOK_HTML.is_dir():
        raise SystemExit("Build the standalone book before running this server")
    server = ThreadingHTTPServer(("127.0.0.1", 18344), partial(JsonNotebookHandler, directory=str(BOOK_HTML)))
    try:
        server.serve_forever()
    finally:
        server.server_close()
