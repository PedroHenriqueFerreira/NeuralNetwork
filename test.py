from tkinter import Tk, Label
from paint_canvas import PaintCanvas

from neural_network import NeuralNetwork

from database import Database

from database.scalers import StandardScaler
from database.encoders import OneHotEncoder

scaler = StandardScaler.from_json('digits/X_scaler.json')
encoder = OneHotEncoder.from_json('digits/y_encoder.json')

neural_network = NeuralNetwork.from_json('digits/neural_network.json')

root = Tk()
root.config(bg='white')

label = Label(root, text='...', font=('Minecraft', 20), bg='white')
label.pack(padx=10, pady=10)

def on_update(data: list[int]) -> None:
    if all(pixel == 0 for pixel in data):
        label.config(text='...')
        
        return
    
    db_data = Database(values=[data])

    scaled_data = scaler.transform(db_data)
    predictions = neural_network.predict(scaled_data)
    decoded_data = encoder.inverse_transform(predictions)

    output = decoded_data.values[0][0]
    
    label.config(text=output)

canvas = PaintCanvas(root, on_update=on_update)
canvas.pack()

root.mainloop()
