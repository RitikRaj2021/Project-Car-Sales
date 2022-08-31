#Used for reading the files
import pandas as pd
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras

from tkinter import Tk
import PySimpleGUI as sg

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

from ann_visualizer.visualize import ann_viz;
import os
import graphviz

#File path to the graphviz instaliation folder
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'



#Import data
data = pd.read_csv('car_sales_dataset.txt   ', encoding='ISO-8859-1')
print(data)


#Plot data
sns.pairplot(data)
plt.show(block=True)


### -------------------------- SR04 -------------------------- ###

#Create input dataset from data
inputs = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Purchase_Amount'], axis = 1)
#Show Input Data
print(inputs)
#Show Input Shape
print("Input data Shape=",inputs.shape)


#Create output dataset from data
output = data['Purchase_Amount']
#Show Output Data
print(output)
#Transform Output
output = output.values.reshape(-1,1)
#Show Output Transformed Shape
print("Output Data Shape=",output.shape)


### -------------------------- SR05 -------------------------- ###

#Scale input
scaler_in = MinMaxScaler()
input_scaled = scaler_in.fit_transform(inputs)
print(input_scaled)



#Scale output
scaler_out = MinMaxScaler()
output_scaled = scaler_out.fit_transform(output)
print(output_scaled)



#Create model
def Model():
    model = Sequential()
    model.add(Dense(25, input_dim=5, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return (model)
    
    
print(Model().summary())



def Train():
#Train model
    epochs_hist = Model().fit(input_scaled, output_scaled, epochs=20, batch_size=10, verbose=1, validation_split=0.2)
    print(epochs_hist.history.keys()) #print dictionary keys


    #Plot the training graph to see how quickly the model learnss
    plt.plot(epochs_hist.history['loss'])
    plt.plot(epochs_hist.history['val_loss'])
    plt.title('Model Loss Progression During Training/Validation')
    plt.ylabel('Training and Validation Losses')
    plt.xlabel('Epoch Number')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show(block=True)

print(Train())



### -------------------------- SR07 -------------------------- ###

# Define the window's contents
layout = [  [sg.Text("Customer Name")],     # Part 2 - The Layout
            [sg.Input()],
            [sg.Text("Customer Email")],
            [sg.Input()],
            [sg.Text("Country")],
            [sg.Input()],
            [sg.Text("Purschase Amount")],
            [sg.Input()],
            [sg.Button('Ok')] ]

# Create the window
window = sg.Window('Custom Values', layout)      # Part 3 - Window Defintion

# Display and interact with the Window
event, values = window.read()                   # Part 4 - Event loop or Window.read call

# Do something with the information gathered
#print('Hello', values[0], "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()                                  # Part 5 - Close the Window





# #GUI Window popup
# from tkinter import *
# from tkinter.ttk import Combobox
# window=Tk()

# btn=Button(window, text="This is Button widget", fg='blue')
# btn.place(x=80, y=100)
# lbl=Label(window, text="This is Label widget", fg='red', font=("Helvetica", 16))
# lbl.place(x=40, y=30)
# txtfld=Entry(window, text="This is Entry Widget", bd=5)
# txtfld.place(x=80, y=150)
# lbl=Label(window, text="This is Label widget", fg='red', font=("Helvetica", 16))
# lbl.place(x=40, y=53)
# txtfld=Entry(window, text="This is Entry Widget", bd=5)
# txtfld.place(x=80, y=150)

# # var = StringVar()
# # var.set("one")
# # data=("one", "two", "three", "four")
# # cb=Combobox(window, values=data)
# # cb.place(x=60, y=150)

# # lb=Listbox(window, height=5, selectmode='multiple')
# # for num in data:
# #     lb.insert(END,num)
# # lb.place(x=250, y=150)

# # v0=IntVar()
# # v0.set(1)
# # r1=Radiobutton(window, text="male", variable=v0,value=1)
# # r2=Radiobutton(window, text="female", variable=v0,value=2)
# # r1.place(x=100,y=50)
# # r2.place(x=180, y=50)
                
# # v1 = IntVar()
# # v2 = IntVar()
# # C1 = Checkbutton(window, text = "Cricket", variable = v1)
# # C2 = Checkbutton(window, text = "Tennis", variable = v2)
# # C1.place(x=100, y=100)
# # C2.place(x=180, y=100)

# window.title('Hello Python')
# window.geometry("600x400+10+10")
# window.mainloop()





### -------------------------- SR06 -------------------------- ###
# Evaluate model
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 
# ***(Note that input data must be normalized)***

input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
#input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])

#Scale input test sample data
#input_test_sample_scaled = scaler_in.transform(customer_Name, customer_Email, country, purschase_Amount)
input_test_sample_scaled = scaler_in.transform(input_test_sample)
#input_test_sample_scaled = scaler_in.transform(inputs)   

#Predict output
output_predict_sample_scaled = Model().predict(input_test_sample_scaled)

#Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled)

#Unscale output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)


ann_viz(Model(), title="Project car Sales Predictor")