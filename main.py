from tkinter import Tk

from src.ui import PPOUI


def main():
    root = Tk()
    app = PPOUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
