import PySimpleGUI as sg
import numpy as np
import script


sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text("Please enter the credits to predict car sales price")],
        [sg.Text('Gender: 0 is female, 1 is male')],
        [sg.Input("")],
        [sg.Text('Age')],
        [sg.Input(" ")],
        [sg.Text('Annual Salary')],
        [sg.Input("")],
        [sg.Text('Credit card debt')],
        [sg.Input("")],
        [sg.Text('Net worth')],
        [sg.Input("")],

        [sg.Button('Predict'), sg.Button('Cancel')],
        [sg.Output(size=(10, 10), key='-OUTPUT-')] ]

# Create the Window
window = sg.Window('Car Sales Prediction', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    #script.InputLisI=values
    print('Predict sales is ', script.output_predict_sample)

window.close()











#def show_gui():
#    sg.theme('DarkAmber')   # Add a touch of color
#    # Gender, Age, Annual Salary, Credit Card Debt, Net Worth
#    layout = [
 #       [sg.Text("Please enter the credits to predict car sales price")],
 #       [sg.Text('Gender: 0 is female, 1 is male')],
 #       [sg.Input("")],
  #      [sg.Text('Age')],
#  #      [sg.Input(" ")],
 #       [sg.Text('Annual Salary')],
 #       [sg.Input("")],
 ##       [sg.Input("")],
 #       [sg.Text('Net worth')],
 #       [sg.Input("")],
#
  #      [sg.Button('Predict'), sg.Button('Cancel')],
  #      [sg.Output(size=(10, 10), key='-OUTPUT-')]
  #  ]
#
 #   # Create the Window
  #  window = sg.Window('Car Sales Prediction', layout)
  #  # Event Loop to process "events" and get the "values" of the inputs
 #   while True:
  #      event, values = window.read()
  #     if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
  #          break
  #      if event == 'Predict':
  #          return np.array([[
  #              int(values[0].strip()),
   #             int(values[1].strip()),
   #             int(values[2].strip()),
   #             int(values[3].strip()),
   #             int(values[4].strip()),
   #         ]])


