import sys
from pathlib import Path

# Tambahkan root directory ke sys.path secara eksplisit
root_dir = Path(__file__).parent.resolve()
sys.path.append(str(root_dir))

if __name__ == "__main__":
    from src.main import main
    main()
