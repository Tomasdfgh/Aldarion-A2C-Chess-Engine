import MTCS as mt
import board_display as bd
import board_reader as br
import model as md
import training_script as ts
import torch
import torch.optim as optim
import openpyxl


if __name__ == "__main__":
    #Model Setup
    model = md.ChessNet().double()
    model.load_state_dict(torch.load('Aldarion_Alpha_Zero.pth'))
    optimizer = optim.AdamW(model.parameters(), lr = 0.001)

    #Load up the workbook
    workbook = openpyxl.load_workbook('DataSet.xlsx')
    sheet = workbook.active

    #Setting up Model's Hyperparameter
    training_split = 80
    validation_split = 10
    barch_size = 5
    epoch = 3

    #Training Begins
    ts.training(model, training_split ,validation_split, barch_size, sheet, optimizer, epoch)


    temperature = 0.8
    num_sim = 300
    path_taken = mt.run_game(model, temperature, num_sim)

    for i in path_taken:
        sheet.append(i)
    workbook.save('DataSet.xlsx')