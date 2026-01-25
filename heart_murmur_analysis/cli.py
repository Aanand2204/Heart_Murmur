import sys
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    # Find the app.py file relative to this cli.py file
    app_path = Path(__file__).parent / "app.py"
    
    # Construct the arguments for streamlit run
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
    ] + sys.argv[1:]
    
    # Run the streamlit CLI
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
