from typing import Any, Callable

from tkinter import Canvas, Misc, Event

class PaintCanvas(Canvas):
    ''' A canvas that can be drawn on '''
    
    def __init__(
        self,
        master: Misc, 
        rows: int = 8, 
        cols: int = 8,
        width: int = 400,
        height: int = 400,
        max_value: int = 16,
        on_update: Callable[[list[int]], None] = lambda _: None,
        **kwargs: Any
    ):
        super().__init__(master, width=width, height=height, **kwargs)
        
        self.width = width
        self.height = height
        
        self.max_value = max_value
        
        self.rows = rows
        self.cols = cols
    
        self.on_update = on_update
    
        self.data: list[list[int]] = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.pixels: list[list[int]] = self.create_pixels()
    
        self.bind('<B1-Motion>', self.draw)
        self.bind('<ButtonRelease-1>', self.clear)
        
        self.focus_set()
    
    def create_pixels(self) -> list[list[int]]:
        ''' Create the pixels '''
        
        pixel_width = self.width / self.cols
        pixel_height = self.height / self.rows
        
        pixels: list[list[int]] = []
        
        for i in range(self.rows):
            row: list[int] = []
            
            for j in range(self.cols):
                
                x0 = j * pixel_width
                y0 = i * pixel_height
                
                x1 = x0 + pixel_width
                y1 = y0 + pixel_height
                                
                row.append(self.create_rectangle(x0, y0, x1, y1, fill='#fff', width=0))

            pixels.append(row)
        
        return pixels
    
    def update(self) -> None:
        ''' Update the canvas '''
        
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] < 0 or self.data[i][j] > self.max_value:
                    continue
                
                gray = int((self.data[i][j] / self.max_value) * 255)
                
                if gray < 0 or gray > 255:
                    continue
                
                gray = 255 - gray
                
                color = '#' + f'{gray:02x}' * 3
                
                self.itemconfig(self.pixels[i][j], fill=color)        
    
        self.on_update([item for row in self.data for item in row])
    
    def draw(self, event: 'Event[Canvas]') -> None:
        ''' Draw on the canvas '''
    
        pixel_width = self.width // self.cols
        pixel_height = self.height // self.rows
        
        x = event.x // pixel_width
        y = event.y // pixel_height
        
        if x >= self.cols or y >= self.rows:
            return
    
        if self.data[y][x] >= self.max_value:
            return
    
        self.data[y][x] += 1
        
        self.update()
    
    def clear(self, event: 'Event[Canvas]') -> None:
        ''' Reset the canvas '''
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = 0
        
        self.update()