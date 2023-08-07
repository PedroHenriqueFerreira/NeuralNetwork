from tkinter import Tk, Label

from neural_network import NeuralNetwork

from database import Database

from database.scalers import StandardScaler
from database.encoders import OneHotEncoder

from paint_canvas import PaintCanvas

scaler = StandardScaler.from_json('digits/X_scaler.json')
encoder = OneHotEncoder.from_json('digits/y_encoder.json')

neural_network = NeuralNetwork.from_json('digits/neural_network.json')

root = Tk()
root.config(background='#fff')
root.option_add('*background', '#fff')

label = Label(root, text='...', font=('Minecraft', 20))
label.pack(padx=10, pady=10)

def predict(data: list[int]) -> None:
    if all(pixel == 0 for pixel in data):
        label.config(text='...')
        return
    
    db = Database(values=[data])

    db = scaler.transform(db)
    
    predictions = neural_network.predict(db.values)
    
    output = encoder.inverse_transform(Database(values=predictions)).values[0][0]
    
    label.config(text=output)

canvas = PaintCanvas(root, on_update=predict)
canvas.pack()

root.mainloop()
