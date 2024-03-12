import MTCS as mt
import board_display as bd
import board_reader as br
import model as md
import training_script as ts
import torch
import torch.optim as optim
import openpyxl
import matplotlib.pyplot as plt


if __name__ == "__main__":

    #Model Setup
    model = md.ChessNet().double()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)

    for i in range(10000):

        #---------Setting up Workbooks and Model---------#

        #Load up the dataset workbook
        workbook = openpyxl.load_workbook('DataSet.xlsx')
        sheet = workbook.active

        #load up the loss workbook
        workbook_training = openpyxl.load_workbook('Testing_Result.xlsx')
        sheet_training = workbook_training.active

        #Load the model again
        model.load_state_dict(torch.load('Aldarion_Alpha_Zero.pth'))

        #---------Generating Data Through Self-Play---------#

        #Setting up for 2 games before training can begin again
        temperature = 0.8
        num_sim = 300

        for _ in range(2):

            fig, ax = bd.get_fig_ax()
            path_taken = mt.run_game(model, temperature, num_sim, fig, ax)

            for z in path_taken:
                sheet.append(z)
            workbook.save('DataSet.xlsx')

            plt.close(fig)

        #---------Begin Training---------#

        #Setting up Model's Hyperparameter
        training_split = 80
        validation_split = 10
        batch_size = 20
        epoch = 20

        #Training Begins
        loss_ = ts.training(model, training_split ,validation_split, batch_size, sheet, optimizer, epoch)

        #Save the training result. Order of saving: Training Loss, Validation Loss, Testing Loss
        for l in loss_:
            sheet_training.append(l)
        workbook_training.save('Testing_Result.xlsx')