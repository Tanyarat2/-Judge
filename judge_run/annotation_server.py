#!/usr/bin/env python3
"""Serve a local run picker and the latest annotation view for each judge run.

No database and no API key are used by this server. It reads only input.json and
out.*.json under the selected runs folder. Judge API credentials are only needed
by the separate CLI judge commands.
"""

import argparse
import html
import pathlib
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_annotation as annotation  # noqa: E402

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
JUDGES = ("mtrag",)


def available_runs(runs_dir):
    if not runs_dir.is_dir():
        return []
    rows = []
    for folder in sorted(runs_dir.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir() or not NAME_PATTERN.fullmatch(folder.name):
            continue
        input_exists = (folder / "input.json").is_file()
        found = [name for name in JUDGES if (folder / ("out.%s.json" % name)).is_file()]
        if input_exists or found:
            rows.append({"name": folder.name, "has_input": input_exists, "judges": found})
    return rows


def picker_page(runs_dir, port):
    rows = available_runs(runs_dir)
    if rows:
        content = []
        for row in rows:
            name = html.escape(row["name"])
            statuses = " ".join("<span class='tag'>%s</span>" % html.escape(n) for n in row["judges"])
            if not statuses:
                statuses = "<span class='muted'>no judge output yet</span>"
            elif len(row["judges"]) < len(JUDGES):
                missing = [n for n in JUDGES if n not in row["judges"]]
                statuses += " " + " ".join("<span class='missing'>%s: not run</span>" % html.escape(n)
                                               for n in missing)
            content.append("<a class='run' href='/run/%s'><strong>%s</strong>"
                           "<span>%s</span></a>" % (name, name, statuses))
        listing = "".join(content)
    else:
        listing = "<p class='muted'>No runs found yet. Create one with <code>python run.py prepare --name pilot7_clapnq</code>.</p>"
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'><title>RAG judge runs</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#1b1f23;font:15px system-ui,sans-serif}
main{max-width:900px;margin:36px auto;padding:0 18px}h1{font-size:25px;margin:0 0 8px}
p{line-height:1.55}.muted{color:#68717d}.runs{display:grid;gap:9px;margin-top:24px}
a.run{display:grid;gap:8px;padding:14px;background:white;border:1px solid #d5d8de;border-radius:6px;color:inherit;text-decoration:none}
a.run:hover{border-color:#27827c;box-shadow:0 3px 12px #1b1f2314}.run strong{font-size:16px}
.tag,.missing{display:inline-block;margin:0 5px 0 0;padding:3px 7px;border-radius:4px;background:#d9eee3;color:#205b38;font:12px ui-monospace,monospace}
.missing{background:#eceef1;color:#59616b}code{background:#e9ebee;padding:2px 4px}
.actions{display:flex;gap:12px;align-items:center;margin-top:20px}button{padding:8px 12px;border:1px solid #b9c0c8;border-radius:4px;background:white;cursor:pointer}
</style></head><body><main><h1>RAG judge runs</h1>
<p class='muted'>Local run folders; no database or browser API key. The list refreshes automatically while this page is open. Select a run to annotate the judge outputs currently available.</p>
<div class='actions'><button onclick='location.reload()'>Refresh list</button><span class='muted'>This page checks for new runs every 10 seconds.</span></div>
<section class='runs'>%s</section>
</main><script>setTimeout(()=>location.reload(),10000)</script></body></html>""" % listing


def make_handler(runs_dir):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/" or path == "/index.html":
                self.respond(200, picker_page(runs_dir, self.server.server_port))
                return
            match = re.fullmatch(r"/run/([^/]+)", path)
            if not match or not NAME_PATTERN.fullmatch(match.group(1)):
                self.respond(404, "Not found")
                return
            work_dir = runs_dir / match.group(1)
            if not work_dir.is_dir():
                self.respond(404, "Run not found")
                return
            try:
                page = annotation.render_page(annotation.build_data(work_dir))
            except SystemExit as exc:
                message = html.escape(str(exc))
                page = ("<!doctype html><meta charset='utf-8'><title>Run unavailable</title>"
                        "<p>%s</p><p><a href='/'>Back to runs</a></p>" % message)
            else:
                nav = ("<nav style='position:sticky;top:0;z-index:9;padding:8px 16px;"
                       "background:#edf1f3;border-bottom:1px solid #ccd'>"
                       "<a href='/'>All runs</a> &nbsp; | &nbsp; "
                       "<a href='/run/%s'>Refresh latest judge outputs</a></nav>" %
                       html.escape(match.group(1), quote=True))
                page = page.replace("<body>", "<body>" + nav, 1)
            self.respond(200, page)

        def respond(self, status, body):
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=pathlib.Path, default=HERE / "runs")
    parser.add_argument("--host", default="127.0.0.1", help="local-only by default")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.runs_dir.resolve()))
    url = "http://%s:%d/" % (args.host, server.server_port)
    print("RAG judge annotation: %s" % url, flush=True)
    print("Press Ctrl+C to stop. Judge runs can continue in another PowerShell window.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
