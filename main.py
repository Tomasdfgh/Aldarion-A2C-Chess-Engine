import MTCS as mt
import board_display as bd
import board_reader as br
import model as md
import training_script as ts
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import openpyxl


if __name__ == "__main__":
    #Model Setup
    model = md.ChessNet().double()
    model.load_state_dict(torch.load('Aldarion_Alpha_Zero.pth'))
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)

    for i in range(10000):

        #Load up the dataset workbook
        workbook = openpyxl.load_workbook('DataSet.xlsx')
        sheet = workbook.active

        #load up the loss workbook
        workbook_training = openpyxl.load_workbook('Testing_Result.xlsx')
        sheet_training = workbook_training.active

        #Setting up Model's Hyperparameter
        training_split = 80
        validation_split = 10
        batch_size = 20
        epoch = 3

        #Training Begins
        # loss_ = ts.training(model, training_split ,validation_split, batch_size, sheet, optimizer, epoch)

        # #Save the training result
        # for l in loss_:
        #     sheet_training.append(l)
        # workbook_training.save('Testing_Result.xlsx')

        #Load the model up again after training
        model.load_state_dict(torch.load('Aldarion_Alpha_Zero.pth'))

        #Setting up for 2 games before training can begin again
        temperature = 0.8
        num_sim = 5
        for _ in range(2):
            fig, ax = bd.get_fig_ax()
            path_taken = mt.run_game(model, temperature, num_sim, fig, ax)

            for z in path_taken:
                sheet.append(z)
            workbook.save('DataSet.xlsx')

            plt.close(fig)