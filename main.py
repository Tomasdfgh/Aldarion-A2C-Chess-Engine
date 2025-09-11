import MTCS as mt
import board_reader as br
import model as md
import training_script as ts
import torch
import torch.optim as optim
import openpyxl
import matplotlib.pyplot as plt


'''
Activate environment:
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate aldarion
'''


if __name__ == "__main__":
    
    model = md.ChessNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    mt.test_board(model, device)


    main_loop = False
    if main_loop:

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        #Model Setup
        model = md.ChessNet().double().to(device)
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
            model.load_state_dict(torch.load('Aldarion_Alpha_Zero.pth', weights_only=True))

            #---------Generating Data Through Self-Play---------#

            #Setting up for 2 games before training can begin again
            temperature = 0.8
            num_sim = 300

            for _ in range(2):
                path_taken = mt.run_game(model, temperature, num_sim, device)

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
            loss_ = ts.training(model, training_split ,validation_split, batch_size, sheet, optimizer, epoch, device)

            #Save the training result. Order of saving: Training Loss, Validation Loss, Testing Loss
            for l in loss_:
                sheet_training.append(l)
            workbook_training.save('Testing_Result.xlsx')