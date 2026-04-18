"""CLI entry point for the explainability dashboard."""

from src.dashboard.app import create_app


def main() -> None:
    """Launch the dashboard."""

    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
