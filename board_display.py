import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
from io import BytesIO
import torch

def get_fig_ax():
	fig, ax = plt.subplots()
	ax.axis('off')
	return fig, ax

def render_board(board, fig, ax, move=None,):
    if move is not None:
        # Apply the move to the board
        board.push_uci(move)

    # Clear the existing figure
    ax.cla()

    # Create SVG representation of the chessboard
    svg_data = chess.svg.board(board=board)

    # Convert SVG to PNG using CairoSVG
    png_data = BytesIO()
    cairosvg.svg2png(file_obj=BytesIO(svg_data.encode("UTF-8")), write_to=png_data)

    # Load the PNG image using Matplotlib
    img = plt.imread(BytesIO(png_data.getvalue()))

    # Display the chessboard
    ax.imshow(img)
    plt.draw()
    plt.pause(0.1)  # Add a small delay to allow for rendering


def plot_loss(fig, ax, x, y1, y2, y3, y4):

    # Clear the existing figure
    ax.cla()
    plt.title("Losses for each Game")
    plt.plot(x,y1, label = "White Actor Loss")
    plt.plot(x,y2, label = "White Critic Loss")
    plt.plot(x,y3, label = "Black Actor Loss")
    plt.plot(x,y4, label = "Black Critic Loss")
    plt.legend(loc = 'best')
    plt.draw()
    plt.pause(0.1)  # Add a small delay to allow for rendering


def plot_board(bitmap):
    title = {0: "Rooks", 1: "Knights", 2: "Bishops", 3: "Queen", 4: "King", 5: "Pawn", 6: "Player's Turn", 7: "Player's Turn", 8: "Player's Turn"}
    fig, axs = plt.subplots(1, 9, figsize=(18, 2))
    for i in range(9):
        axs[i].imshow(bitmap[i])
        axs[i].axis('off')
        axs[i].set_title(title[i])
    plt.show()