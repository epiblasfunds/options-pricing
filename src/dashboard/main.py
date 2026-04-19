"""CLI entry point for the explainability dashboard."""

import os

from src.dashboard.app import create_app


def main() -> None:
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)

app = create_app()
server = app.server

if __name__ == "__main__":
    main()    main()