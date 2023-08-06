from typing import Any
from tkinter import Tk, Canvas

root = Tk()

canvas = Canvas(root, width=400, height=400)

grid = (8, 8)

matrix = [[0 for _ in range(grid[0])] for _ in range(grid[1])]

for i in range(grid[0]):
    for j in range(grid[1]):
        x = i * 50
        y = j * 50
        canvas.create_rectangle(x, y, x + 50, y + 50, fill="white", width=0)

def on_click(event: Any) -> None:
    x = event.x
    y = event.y

    x = (x // 50) * 50
    y = (y // 50) * 50

    matrix[x // 50][y // 50] += 1

    match matrix[x // 50][y // 50]:
        case 0:
            color = "#fff"
        case 1:
            color = "#eee"
        case 2:
            color = "#ddd"
        case 3:
            color = "#ccc"
        case 4:
            color = "#bbb"
        case 5:
            color = "#aaa"
        case 6:
            color = "#999"
        case 7:
            color = "#888"
        case 8:
            color = "#777"
        case 9:
            color = "#666"
        case 10:
            color = "#555"
        case 11:
            color = "#444"
        case 12:
            color = "#333"
        case 13:
            color = "#222"
        case 14:
            color = "#111"
        case 15:
            color = "#000"
            
    if matrix[x // 50][y // 50] > 15:
        return

    canvas.create_rectangle(x, y, x + 50, y + 50, fill=color, width=0)

canvas.bind("<B1-Motion>", on_click)
canvas.pack(expand=True)

root.mainloop()