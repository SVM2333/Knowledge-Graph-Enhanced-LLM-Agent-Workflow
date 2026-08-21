"""主程序入口"""
import sys
import webbrowser
from pathlib import Path
from threading import Timer

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def open_browser():
    """延迟打开浏览器"""
    webbrowser.open('http://localhost:8503')

if __name__ == "__main__":
    import streamlit.web.cli as stcli

    app_path = project_root / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.port=8503"]

    # 延迟3秒后自动打开浏览器
    Timer(3, open_browser).start()

    sys.exit(stcli.main())
