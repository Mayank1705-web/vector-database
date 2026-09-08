from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes


# Force the correct MIME type for CSS.
mimetypes.add_type("text/css", ".css", strict=True)


class FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    frontend_dir = project_root / "frontend"

    if not frontend_dir.is_dir():
        raise RuntimeError(f"Frontend directory not found: {frontend_dir}")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 5500),
        lambda *args, **kwargs: FrontendHandler(
            *args,
            directory=str(frontend_dir),
            **kwargs,
        ),
    )

    print(f"Serving frontend from: {frontend_dir}")
    print("Frontend available at: http://127.0.0.1:5500")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()